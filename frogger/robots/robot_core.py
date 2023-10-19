from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numba import jit
from pydrake.geometry import (
    AddContactMaterial,
    AddRigidHydroelasticProperties,
    CollisionFilterDeclaration,
    GeometrySet,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    ProximityProperties,
    QueryObject,
    Rgba,
    Role,
    StartMeshcat,
)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.multibody.tree import (
    Body,
    Frame,
    JacobianWrtVariable,
    SpatialInertia,
    UnitInertia,
)
from pydrake.systems.controllers import InverseDynamics
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import Adder, Demultiplexer, Multiplexer
from qpth.qp import QPFunction
from quantecon.optimize.linprog_simplex import linprog_simplex as linprog

from frogger import ROOT
from frogger.grasping import (
    compute_gOCs,
    compute_grasp_matrix,
    compute_primitive_forces,
    wedge,
)
from frogger.objects import ObjectDescription


@dataclass
class RobotModelConfig:
    """A configuration for a robot model."""


class RobotModel(ABC):
    """Abstract class for robot models.

    This class serves as a container for caching computation efficiently so that
    repeated calculations are minimized.

    [NOTE] any geometries belonging to the fingertip that are ALLOWED to collide with
    the object should have the substring "FROGGERCOL" in their *.sdf file.
    """

    def __init__(
        self,
        narm: int,
        nhand: int,
        nuarm: int,
        nuhand: int,
        nc: int,
        obj: ObjectDescription,
        settings: dict | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the robot model.

        Parameters
        ----------
        narm : int
            The number of configuration variables associated with the arm.
        nhand : int
            The number of configuration variables associated with the hand.
        nuarm : int
            The number of actuated DOFs associated with the arm.
        nuhand : int
            The number of actuated DOFs associated with the hand.
        nu : int
            The number of degrees of actuation of the system.
        nc : int
            The number of contact points of the manipulator.
        obj : ObjectDescription
            A description of the object to be manipulated.
        settings : dict | None, default=None
            A dictionary of settings formulated as key/value pairs for loading a
            model. The settings will be model-specific.

            The dictionary should always have the key/value pairs:
            * "ns" : int Number of sides of the desired pyramidal approximation to the
              friction cone. Must be >= 3.
            * "mu" : float. The coefficient of friction. Must be > 0.0.
        verbose : bool, default=False
            Indicates whether to print verbose outputs.
        """
        # assigning variables and checking settings
        self.narm = narm
        self.nhand = nhand
        self.nuarm = nuarm
        self.nuhand = nuhand
        self.n = narm + nhand
        self.nu = nuarm + nuhand
        self.nc = nc
        self.obj = obj
        self.name = "robot"  # the name of the robot - should be overriden by children
        if settings is None:
            self.settings = {"ns": 4, "mu": 0.5}
        else:
            if "ns" not in settings:
                settings["ns"] = 4
            else:
                assert isinstance(settings["ns"], int)
                assert settings["ns"] >= 3
            if "mu" not in settings:
                settings["mu"] = 0.5
            else:
                assert isinstance(settings["mu"], float)
                assert settings["mu"] > 0.0
            self.settings = settings
        self.settings["nc"] = nc
        self.ns = self.settings["ns"]
        self.mu = self.settings["mu"]
        self.l_cutoff = self.settings.get("l_cutoff", 1e-6)
        assert self.l_cutoff >= 1e-6 and self.l_cutoff <= 1.0
        self.baseline = self.settings.get("baseline", False)
        self.viz = self.settings.get("viz", True)

        # boilerplate Drake code
        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=0.001
        )
        self.parser = Parser(self.plant, self.scene_graph)
        self.parser.package_map().Add("frogger", ROOT)
        self.preload_model()

        # adding a table to the system
        self.parser.AddModels(ROOT + "/models/station/tabletop.sdf")
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("tabletop_base"),
            RigidTransform(
                RotationMatrix(),
                np.array([0.34, 0.0, -0.05]),
            ),
        )

        self.arm_instance = self.plant.GetModelInstanceByName(f"{self.arm_name}")
        self.hand_instance = self.plant.GetModelInstanceByName(
            f"{self.hand_name}_simplified"
        )

        # registering object's visual/collision geoms with the plant
        assert None not in [self.obj.shape_visual, self.obj.shape_collision_list]
        self.obj_instance = self.plant.AddModelInstance("obj")
        I_o = obj.inertia / obj.mass
        self.obj_body = self.plant.AddRigidBody(
            "obj",
            self.obj_instance,
            SpatialInertia(
                obj.mass,
                np.zeros(3),
                UnitInertia(
                    I_o[0, 0], I_o[1, 1], I_o[2, 2], I_o[0, 1], I_o[0, 2], I_o[1, 2]
                ),
            ),
        )

        # visual geoms
        self.plant.RegisterVisualGeometry(
            self.obj_body,
            RigidTransform(),
            obj.shape_visual,
            "obj_visual",
            np.array([0.9, 0.9, 0.9, 0.5]),  # make object transparent
        )
        prox_properties = ProximityProperties()
        fric = CoulombFriction(static_friction=0.7, dynamic_friction=0.7)
        dissipation = 0.1  # hunt-crossley dissipation
        stiffness = 1e6  # point contact stiffness
        resolution_hint = 0.01  # res hint for the rigid body
        AddContactMaterial(dissipation, stiffness, fric, prox_properties)
        AddRigidHydroelasticProperties(resolution_hint, prox_properties)

        # collision geometries
        for i, shape_col in enumerate(obj.shape_collision_list):
            self.plant.RegisterCollisionGeometry(
                self.obj_body,
                RigidTransform(),
                shape_col,
                f"obj_collision_{i}",
                prox_properties,
            )
        self.plant.Finalize()

        # constructing system diagram with gravity compensation.
        # takes in feedback torque and adds it to the feedforward gravity comp.
        # [NOTE] for this controller to work properly, we want a plant that ONLY has
        # the robot in it (or the inverse dynamics controller will also "see" the
        # object and treat its states as part of the robot state).
        gc_mode = InverseDynamics.InverseDynamicsMode.kGravityCompensation
        gc_ctrl = self.builder.AddSystem(InverseDynamics(self.arm_hand_plant, gc_mode))
        adder_tau = self.builder.AddSystem(Adder(2, self.nu))  # adds gc tau to fb tau

        # demux and mux to rearrange robot/object states appropriately
        # the 7 and 6 here refer to the object's position (position + quat) and its
        # generalized velocities (translational and angular velocity)
        demux = self.builder.AddSystem(Demultiplexer([self.n, 7, self.n, 6]))
        mux = self.builder.AddSystem(Multiplexer([self.n, self.n]))

        # connecting diagram
        self.builder.Connect(gc_ctrl.get_output_port(), adder_tau.get_input_port(0))
        self.builder.Connect(self.plant.get_state_output_port(), demux.get_input_port())
        self.builder.Connect(demux.get_output_port(0), mux.get_input_port(0))
        self.builder.Connect(demux.get_output_port(2), mux.get_input_port(1))
        self.builder.Connect(mux.get_output_port(), gc_ctrl.get_input_port(0))
        self.builder.Connect(
            adder_tau.get_output_port(), self.plant.get_actuation_input_port()
        )
        self.builder.ExportInput(adder_tau.get_input_port(1), "tau_command_in")
        self.builder.ExportOutput(
            self.scene_graph.get_query_output_port(), "sim_query_object"
        )
        self.builder.ExportOutput(mux.get_output_port(), "model_state_out")
        self.builder.ExportOutput(
            self.plant.get_body_poses_output_port(), "body_poses_out"
        )

        # exporting arm state for differential IK
        arm_demux = self.builder.AddSystem(
            Demultiplexer([self.narm, self.nhand, self.narm, self.nhand])
        )
        arm_mux = self.builder.AddSystem(Multiplexer([self.narm, self.narm]))
        self.builder.Connect(mux.get_output_port(0), arm_demux.get_input_port(0))
        self.builder.Connect(arm_demux.get_output_port(0), arm_mux.get_input_port(0))
        self.builder.Connect(arm_demux.get_output_port(2), arm_mux.get_input_port(1))
        self.builder.ExportOutput(arm_mux.get_output_port(0), "arm_state_out")

        # initializes visualization objects. For the joint slider, we set step=1e-12
        # because the slider resolution also rounds the commanded configuration.
        if self.viz:
            self.meshcat = StartMeshcat()
            self.visualizer = MeshcatVisualizer.AddToBuilder(
                self.builder,
                self.scene_graph,
                self.meshcat,
                MeshcatVisualizerParams(role=Role.kPerception, prefix="visual"),
            )
            MeshcatVisualizer.AddToBuilder(
                self.builder,
                self.scene_graph,
                self.meshcat,
                MeshcatVisualizerParams(
                    role=Role.kProximity,
                    prefix="collision",
                    default_color=Rgba(0.9, 0.9, 0.9, 0.5),
                ),
            )
            self.meshcat.SetProperty("collision", "visible", False)
            self.sliders = self.builder.AddSystem(
                JointSliders(self.meshcat, self.plant, step=1e-12)
            )

        self.diagram = self.builder.Build()
        self.plant = self.diagram.GetSubsystemByName("plant")
        self.scene_graph = self.diagram.GetSubsystemByName("scene_graph")

        # storing diagram context and subsystem contexts
        _diag_context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(_diag_context)
        self.sg_context = self.scene_graph.GetMyMutableContextFromRoot(_diag_context)
        self._qo_port = self.scene_graph.get_query_output_port()

        # setting the pose of the object body in the world and collision filtering the
        # object with the tabletop
        self.plant.SetFreeBodyPose(self.plant_context, self.obj_body, obj.X_WO)

        cfm = self.scene_graph.collision_filter_manager(self.sg_context)
        inspector = self.query_object.inspector()

        # collision filtering for the coarse internal representation of collision geom
        self.obj_geoms = []
        tabletop_geom = None
        for g in inspector.GetAllGeometryIds():
            name = inspector.GetName(g)
            if "obj" in name and "visual" not in name:
                self.obj_geoms.append(g)

            elif "tabletop_collision" in name:
                tabletop_geom = g

        tabletop_set = GeometrySet(tabletop_geom)
        obj_set = GeometrySet(self.obj_geoms)
        cfm.Apply(CollisionFilterDeclaration().ExcludeBetween(tabletop_set, obj_set))

        # contact frames and locations
        _, self.contact_frames, self.contact_locs = self.compute_fingertip_contacts()
        assert len(self.contact_frames) == nc
        assert len(self.contact_locs) == nc

        # defining linear box constraint matrices
        lb_q, ub_q = self._q_bounds()
        I = np.eye(self.n)
        self.A_box = np.concatenate((-I, I), axis=-2)  # (2*n, n)
        self.b_box = np.concatenate((lb_q, -ub_q), axis=-1)  # (2*n,)

        # all the possible hand-obj collision pairs, which are always checked
        self.hand_obj_pairs = []
        inspector = self.query_object.inspector()
        col_cands = list(inspector.GetCollisionCandidates())
        for c in col_cands:
            id_A, id_B = c[0], c[1]
            name_A, name_B = inspector.GetName(id_A), inspector.GetName(id_B)
            has_tip = "FROGGERCOL" in name_A or "FROGGERCOL" in name_B
            has_obj = "obj" in name_A or "obj" in name_B
            if has_tip and has_obj:
                self.hand_obj_pairs.append((id_A, id_B))

        # cached values. all Jacobians wrt q when unspecified.
        self.q = None  # cached value of the last unique configuration
        self.p_tips = None  # cached values of the fingertip FK and Jacobians
        self.J_tips = None
        self.P_OF = None  # cached values of tips and contact normals in object frame
        self.n_O = None
        self.Ds_p = None  # cached value of Ds evaluated at fingertips
        self.s_fk = None  # cached values of s composed with FK and its Jacobian
        self.Ds_fk = None  # this is D[s(FK)](q), NOT Ds(q) like Ds_p
        self.h = None  # cached values of the eq constraint function and Jacobian
        self.Dh = None
        self.g = None  # cached values of the ineq constraint function and Jacobian
        self.Dg = None
        self.G = None  # cached values of the grasp map and Jacobian
        self.DG = None
        self.W = None  # cached values of wrench basis matrix and Jacobian
        self.DW = None
        self.l = None  # cached values of min convex weight LP and Jacobian
        self.Dl = None
        self.gid_pair_inds = {}  # dict ordering collision geometry id pairs
        self.sdp_normals = {}  # dict caching signed distance pair normal vectors
        self.hand_obj_cols = {}  # dict recording most penetrating hand/obj collisions

        # [BASELINE] Wu 2022 CVAE Bilevel Optimization paper
        self._init_baseline_cons()

        # warm-starting the cache
        if verbose:
            print("Warming up computation cache...")
        self.F = compute_primitive_forces(self.ns, self.mu)  # primitive force matrix
        self.compute_l(np.zeros(self.n))

    def _init_baseline_cons(self) -> None:
        """Initializes the baseline constraint from Wu 2022.

        Equation (5) of the paper:
        "Learning Diverse and Physically Feasible Dexterous Grasps with Generative
        Model and Bilevel Optimization"
        """
        # setting up the feasibility QP
        # (i) friction cone constraint
        Lambda_i = np.vstack(
            (
                np.append(np.cos(2 * np.pi * np.arange(self.ns) / self.ns), 0.0),
                np.append(np.sin(2 * np.pi * np.arange(self.ns) / self.ns), 0.0),
                -np.append(self.mu * np.cos(np.pi * np.ones(self.ns) / self.ns), 1.0),
            )
        ).T  # pyramidal friction cone approx in contact frame + min normal force

        # Ain has shape((ns + 1) * nc, 3 * nc)
        fn_min = 1.0  # hard code min normal force to 1, that's what they use
        A_in = torch.tensor(np.kron(np.eye(self.nc), Lambda_i))
        b_in = torch.zeros((self.ns + 1) * self.nc).double()
        b_in[self.ns :: self.ns + 1] = -fn_min

        A_eq = torch.Tensor().double()  # empty
        b_eq = torch.Tensor().double()

        # (ii) setting up QP constraint function
        def bilevel_constraint_func(G: torch.Tensor) -> torch.Tensor:
            """Constraint function for bilevel optimization.

            Takes in the grasp map and returns a cost.
            """
            if G.dtype == torch.float32:
                G = G.double()
            Q = G.T @ G + 1e-7 * torch.eye(3 * self.nc).double()
            f_opt = QPFunction(verbose=-1, check_Q_spd=False)(
                Q, torch.zeros(3 * self.nc).double(), A_in, b_in, A_eq, b_eq
            ).squeeze()
            obj_opt = f_opt @ Q @ f_opt
            return obj_opt

        self._bilevel_cons = bilevel_constraint_func

    def set_X_WO(self, X_WO_new: RigidTransform) -> None:
        """Sets a new object pose."""
        self.obj.set_X_WO(X_WO_new)
        self.obj.X_WO = RigidTransform(X_WO_new)
        self.obj.settings["X_WO"] = RigidTransform(X_WO_new)
        self.plant.SetFreeBodyPose(self.plant_context, self.obj_body, X_WO_new)

    def reset(self) -> None:
        """Resets any internal variables."""

    @property
    def joint_coupling_constraints(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint coupling constraints.

        If the implemented system has joint coupling this MUST be overloaded.
        Constraints are represented as a linear system of the form:
            A_couple @ q + b_couple = 0
        This is how mimic joints are represented in URDF formats and also how coupler
        constraints are represented in Drake.

        By default, all joints are assumed to be independent (i.e., n_couple=0).

        Returns
        -------
        A_couple : np.ndarray, shape=(n_couple, n)
            The gear ratio matrix.
        b_couple : np.ndarray, shape=(n_couple,)
            The offset vector.
        """
        A_couple = np.zeros((0, self.n))
        b_couple = np.zeros(0)
        return A_couple, b_couple

    def _q_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounds on the configuration.

        Returns
        -------
        lb_q : np.ndarray, shape=(n,)
            Lower bounds.
        ub_q : np.ndarray, shape=(n,)
            Upper bounds.
        """
        return (
            self.plant.GetPositionLowerLimits()[:self.n],
            self.plant.GetPositionUpperLimits()[:self.n],
        )

    def _tau_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounds on the motor torques.

        Returns
        -------
        lb_tau : np.ndarray, shape=(nu,)
            Lower bounds.
        ub_tau : np.ndarray, shape=(nu,)
            Upper bounds.
        """
        return (
            self.plant.GetEffortLowerLimits(),
            self.plant.GetEffortUpperLimits(),
        )

    @abstractmethod
    def preload_model(self) -> None:
        """Loads the robot description.

        Should use self.parser to add the robot model and weld all the necessary
        frames. The default RobotModel constructor will automatically process the
        object and finalize the plant. The arm is expected to be welded to the origin.
        It should "face" the positive x-axis, as that is where the table extends.
        """

    @abstractmethod
    def compute_fingertip_contacts(
        self,
    ) -> tuple[list[Body], list[Frame], list[np.ndarray]]:
        """Computes fixed fingertip contact locations used for grasp synthesis.

        Returns
        -------
        contact_bodies : list[Body]
            A list of the Body objects associated with each fingertip.
        contact_frames : list[Frame]
            A list of the Frames associated with each fingertip making contact.
        contact_locs : list[np.ndarray]
            A list of the contact locations on each finger wrt the fingertip frames.
        """

    @property
    @abstractmethod
    def pregrasp_hand_config(self) -> np.ndarray:
        """A heuristically-motivated pregrasp hand configuration.

        Used to help sample initial grasps by pre-shaping the hand.

        Returns
        -------
        q_hand : np.ndarray, shape=(n_hand,)
            The configuration of ONLY the hand pre-grasp.
        """

    @property
    @abstractmethod
    def X_BasePalmcenter(self) -> RigidTransform:
        """The pose of the palm center relative to the hand base frame.

        Used to help sample initial grasps by placing palm relative to the object.

        Returns
        -------
        X_BasePalmcenter : RigidTransform
            The pose.
        """

    @property
    def query_object(self) -> QueryObject:
        """The query object."""
        return self._qo_port.Eval(self.sg_context)

    # ######################## #
    # CACHED VALUE COMPUTATION #
    # ######################## #

    def _process_collisions(self, q: np.ndarray) -> None:
        """Computes all information related to collision constraints."""
        n = self.n

        # checking if first iteration
        # if so, sort all possible collision pairs and initialize constraint info
        if self.g is None:
            # gidps means geometry id pairs
            col_cands = list(self.query_object.inspector().GetCollisionCandidates())
            gidps = [(c[0].get_value(), c[1].get_value()) for c in col_cands]
            _, col_cands = zip(*sorted(zip(gidps, col_cands)))  # sort cands by gidps

            # the first 2n constraints are box constraints, followed by collision
            # constraints. if using the min-weight metric, then bound it
            self.n_pairs = len(col_cands)
            n_minweight = 0 if self.baseline else 1
            self.g = np.zeros(2 * n + self.n_pairs + n_minweight)
            self.Dg = np.zeros((2 * n + self.n_pairs + n_minweight, n))
            self.Dg[: (2 * n), :] = self.A_box

            # initializing a dictionary that orders gid pairs
            # this lets us update specific signed distances in a well-ordered way
            for i, c in enumerate(col_cands):
                id_A = c[0]
                id_B = c[1]
                self.gid_pair_inds[(id_A, id_B)] = i

        # update joint limit constraint values
        self.g[: (2 * n)] = self.A_box @ q + self.b_box

        # collisions get computed conditionally after culling w/max_distance
        # everything sufficiently far away we ignore and set the gradient to 0
        # d_min is the desired safety margin, a minimum distance enforced between geoms
        d_min = self.settings.get("d_min", 0.0)
        sdps = self.query_object.ComputeSignedDistancePairwiseClosestPoints(
            max_distance=(d_min + 0.001)
        )
        if len(sdps) > 0:
            gidps = [(sdp.id_A.get_value(), sdp.id_B.get_value()) for sdp in sdps]
            _, sdps = zip(*sorted(zip(gidps, sdps)))  # sort sdps by gidps
            sdps = list(sdps)  # tuple -> list
        else:
            gidps = []

        # manually adds hand-obj collisions to the list of collision pairs
        for ho_pair in self.hand_obj_pairs:
            if ho_pair not in gidps:
                sdps.append(
                    self.query_object.ComputeSignedDistancePairClosestPoints(
                        ho_pair[0], ho_pair[1]
                    )
                )
        inspector = self.query_object.inspector()  # model inspector for geometries

        def get_bf(gid):
            """From a geometry ID, return the body frame."""
            fid = inspector.GetFrameId(gid)
            return self.plant.GetBodyFromFrameId(fid).body_frame()

        J_witness_dict = {}  # Jacobians of witness points

        self.g[2 * n : (2 * n + self.n_pairs)] = -1.0  # setting "far" points to 1.0m
        self.Dg[2 * n : (2 * n + self.n_pairs), :] = 0.0  # resetting col gradients

        # loop through unculled collision pairs
        self.hand_obj_cols = {}  # reset the hand-obj dictionary
        for sdp in sdps:
            id_A = sdp.id_A  # geometry IDs
            id_B = sdp.id_B
            sd = sdp.distance  # signed distance
            nrml = sdp.nhat_BA_W  # direction of fastest increase outward from point B
            pA = sdp.p_ACa  # witness points A and B
            pB = sdp.p_BCb
            fA = get_bf(id_A)  # frames associated with points A and B
            fB = get_bf(id_B)

            # computing/retrieving cached Jacobians, shape=(3, n)
            try:
                J_A = J_witness_dict[id_A]
            except KeyError:
                J_A = self.plant.CalcJacobianTranslationalVelocity(
                    self.plant_context,
                    JacobianWrtVariable.kQDot,
                    fA,
                    pA,
                    self.plant.world_frame(),  # velocity measured in the world frame
                    self.plant.world_frame(),  # velocity expressed in the world frame
                )[..., : self.n]
                J_witness_dict[id_A] = J_A
            try:
                J_B = J_witness_dict[id_B]
            except KeyError:
                J_B = self.plant.CalcJacobianTranslationalVelocity(
                    self.plant_context,
                    JacobianWrtVariable.kQDot,
                    fB,
                    pB,
                    self.plant.world_frame(),  # velocity measured in the world frame
                    self.plant.world_frame(),  # velocity expressed in the world frame
                )[..., : self.n]
                J_witness_dict[id_B] = J_B

            # handling case when nrml is nan due to gradient exactly on surface
            # if nan, use the cached value. if no cached value, set equal to [0, 0, 1]
            if np.any(np.isnan(nrml)):
                try:
                    nrml = self.sdp_normals[(id_A, id_B)]
                except KeyError:
                    nrml = np.array([0.0, 0.0, 1.0])  # arbitrary choice
            self.sdp_normals[(id_A, id_B)] = nrml

            # caching distance constraint computations
            # [NOTE] the formula for Dg is like this because nhat_BA_W is always
            # pointing OUTWARD from the surface of object B. It is NOT the vector that
            # always points from B to A. If this were the case, we would need to take
            # the sign of the SDF into account when computing the gradient. Therefore,
            # the gradient formula is unchanging whether or not there is contact.
            # drake.mit.edu/doxygen_cxx/structdrake_1_1geometry_1_1_signed_distance_pair.html
            i = self.gid_pair_inds[(id_A, id_B)]

            # only allow tip/obj collision w/ small penetration
            names = [inspector.GetName(id_A), inspector.GetName(id_B)]
            has_tip = "FROGGERCOL" in names[0] or "FROGGERCOL" in names[1]
            has_obj = "obj_collision" in names[0] or "obj_collision" in names[1]
            d_pen = self.settings.get("d_pen", 0.0)
            if has_tip and has_obj:
                self.g[2 * n + i] = -sd - d_pen  # allow fingertip to penetrate obj
            else:
                self.g[2 * n + i] = d_min - sd  # other pairs must respect d_min
            Dgi = -(J_A - J_B).T @ nrml
            self.Dg[2 * n + i, :] = Dgi

            # updating the most interpenetrating pairs for each link allowing collision
            if has_tip and has_obj:
                bA = fA.body()  # bodies associated with collision geoms
                bB = fB.body()
                body_name_A = bA.name()
                body_name_B = bB.name()

                if "FROGGERCOL" in names[0]:
                    p_tip_W = self.plant.CalcPointsPositions(
                        self.plant_context,
                        fA,
                        pA,
                        self.plant.world_frame(),
                    ).squeeze(-1)
                    p_tip_C = pA
                    f_tip = fA
                    key = (body_name_A, body_name_B)
                else:
                    p_tip_W = self.plant.CalcPointsPositions(
                        self.plant_context,
                        fB,
                        pB,
                        self.plant.world_frame(),
                    ).squeeze(-1)
                    p_tip_C = pB
                    f_tip = fB
                    key = (body_name_B, body_name_A)

                if not key in self.hand_obj_cols:
                    self.hand_obj_cols[key] = (sd, -Dgi, p_tip_W, p_tip_C, f_tip)
                else:
                    if sd < self.hand_obj_cols[key][0]:
                        self.hand_obj_cols[key] = (sd, -Dgi, p_tip_W, p_tip_C, f_tip)

        # updating p_tips and J_tips
        h_tip = []
        Dh_tip = []
        p_tips = []
        J_tips = []
        for k, v in sorted(self.hand_obj_cols.items()):
            h_tip.append(v[0])
            Dh_tip.append(v[1])
            p_tips.append(v[2])
            J_tips.append(
                self.plant.CalcJacobianTranslationalVelocity(
                    self.plant_context,
                    JacobianWrtVariable.kQDot,
                    v[4],
                    v[3],
                    self.plant.world_frame(),
                    self.plant.world_frame(),
                )[..., :self.n]
            )
        self.h_tip = np.array(h_tip)
        self.Dh_tip = np.array(Dh_tip)
        self.p_tips = np.array(p_tips)
        self.J_tips = np.array(J_tips)

    def _compute_s_fk(self) -> None:
        """Computes the composition of s with the forward kinematics and Jacobian."""
        self.s_fk = self.obj.s_W(self.p_tips, batched=True)  # (nc,)
        self.Ds_p = self.obj.Ds_W(self.p_tips, batched=True)  # Ds(p)
        J_T = np.swapaxes(self.J_tips, -1, -2)  # (nc, n, 3)
        self.Ds_fk = (J_T @ self.Ds_p[..., None]).squeeze(-1)  # (nc, n), D[s(FK)](q)

    def _compute_eq_cons(self, q: np.ndarray) -> None:
        """Computes equality constraints and their gradients."""
        A_couple, b_couple = self.joint_coupling_constraints
        if self.baseline:
            # computing the value and gradient of the bilevel constraint wrt q
            G_torch = torch.tensor(self.G)
            feas = self._bilevel_cons(G_torch).detach().numpy()[..., None]
            Dfeas_G = (
                torch.autograd.functional.jacobian(
                    self._bilevel_cons, G_torch, create_graph=True, strict=True
                )
                .detach()
                .numpy()
                .astype(np.float64)
            )
            DG = self.DG
            Dfeas = (Dfeas_G.reshape(-1) @ DG.reshape((-1, DG.shape[-1])))[None, ...]

            if len(b_couple) == 0:
                self.h = np.concatenate((self.h_tip, feas))
                self.Dh = np.concatenate((self.Dh_tip, Dfeas), axis=0)
            else:
                h_s_fk = self.h_tip
                h_couple = A_couple @ q + b_couple
                self.h = np.concatenate((h_s_fk, h_couple, feas))
                self.Dh = np.concatenate((self.Dh_tip, A_couple, Dfeas), axis=0)
        else:
            # if not using the baseline, simply proceed
            if len(b_couple) == 0:
                self.h = self.h_tip
                self.Dh = self.Dh_tip
            else:
                h_s_fk = self.h_tip
                h_couple = A_couple @ q + b_couple
                self.h = np.concatenate((h_s_fk, h_couple))
                self.Dh = np.concatenate((self.Dh_tip, A_couple), axis=0)

    def _finish_ineq_cons(self) -> None:
        """Finieshes computing inequality constraints and their gradients."""
        # force closure inequality constraint
        if not self.baseline:
            idx_minw = self.n_pairs + 2 * self.n
            self.g[idx_minw] = -self.l + self.l_cutoff / (self.ns * self.nc)
            self.Dg[idx_minw, :] = -self.Dl

    def _compute_G_and_W(self) -> None:
        """Computes the grasp and wrench matrices."""
        nc = self.nc

        # computing the grasp map in the object frame using known pose
        X_WO = self.obj.X_WO  # pose of obj wrt world
        X_OW = X_WO.inverse()
        R_OW = X_WO.inverse().rotation().matrix()

        # contact points and normals in the object frame
        self.P_OF = X_OW @ self.p_tips.T  # (3, nc)
        _normals = -self.Ds_p.T / np.linalg.norm(self.Ds_p, axis=1)  # (3, nc)
        self.n_O = R_OW @ _normals  # INWARD pointing normals

        self.gOCs = compute_gOCs(self.P_OF, self.n_O)  # (nc, 4, 4)
        self.G = compute_grasp_matrix(self.gOCs)  # (6, 3 * nc), object frame
        self.W = self.G @ np.kron(np.eye(nc), self.F)  # cols are primitive wrenches

    def _compute_DG_and_DW(self) -> None:
        """Computes the Jacobians of the grasp and wrench matrices."""
        J_T = np.swapaxes(self.J_tips, -1, -2)  # (nc, n, 3)
        X_WO = self.obj.X_WO  # pose of obj wrt world
        R_OW = X_WO.inverse().rotation().matrix()

        Ds_O_ps = self.obj.Ds_O(self.P_OF.T, batched=True)
        D2s_O_ps = self.obj.D2s_O(self.P_OF.T, batched=True)
        self.DG, self.DW = self._DG_DW_helper(
            J_T, Ds_O_ps, D2s_O_ps, self.P_OF, self.n_O, R_OW, self.F
        )

    @staticmethod
    @jit(nopython=True, fastmath=True, cache=True)
    def _DG_DW_helper(
        J_T: np.ndarray,
        Ds_O_ps: np.ndarray,
        D2s_O_ps: np.ndarray,
        P_OF: np.ndarray,
        n_O: np.ndarray,
        R_OW: np.ndarray,
        F: np.ndarray,
    ) -> np.ndarray:
        """Numba jit-compiled helper for computing DG and DW.

        I'm a maniac who decided to compute this analytically.

        This was refactored into a very hard-to-read way to be compatible with numba.
        Apologies in advance for any future readers.
        """
        n = J_T.shape[1]
        nc = J_T.shape[0]
        ns = F.shape[-1]

        # need to consider the change from object to world frame
        DG = np.empty((6, 3 * nc, n))  # (6, 3 * nc, n)
        DW = np.empty((6, nc * ns, n))  # (6, nc * ns, n)

        # pre-computing non-contiguous products beforehand for numba performance
        _zeta = np.array([1.2, 2.3, 3.4])
        _zeta_prime = np.array([3.4, 2.3, 1.2])
        zetas = np.empty((nc, 3))
        for i in range(nc):
            nrml = n_O.T[i, :]  # n_O.T has shape (nc, 3), select ith row
            if np.linalg.norm(nrml - _zeta) <= 1e-6:
                zeta = _zeta_prime
            else:
                zeta = _zeta
            zetas[i, :] = zeta
        n_O_zeta = np.diag(n_O.T @ zetas.T)  # (nc,)
        R_OW_J_Ts = (np.ascontiguousarray(J_T).reshape((-1, 3)) @ R_OW.T).reshape(
            J_T.shape
        )
        summing_matrix = np.kron(np.eye(nc), np.ones((1, 3)))
        gs_inners = summing_matrix @ (np.ascontiguousarray(Ds_O_ps).reshape(-1) ** 2)

        for i in range(nc):
            p = P_OF.T[i, :]  # P_OF.T has shape (nc, 3), select ith row
            nrml = n_O.T[i, :]  # n_O.T has shape (nc, 3), select ith row
            R_OW_J = R_OW_J_Ts[i].T  # (3, n)

            # compute R, the rotation matrix for this contact frame
            z = zeta - n_O_zeta[i] * nrml
            zz = z @ z
            tx = z / np.sqrt(zz)
            ty = np.cross(nrml, tx)
            R = np.stack((tx, ty, nrml)).T

            # compute DR_p, Jacobian of rotation matrix wrt p
            gs = Ds_O_ps[i]  # compute this in the object frame
            gsgs = gs_inners[i]
            factor1 = (np.eye(3) - np.outer(gs, gs) / gsgs) / np.sqrt(gsgs)
            factor2 = D2s_O_ps[i]  # compute this in the object frame
            Dn_p = -factor1 @ factor2  # (3, 3)

            factor1 = np.eye(3) - np.outer(z, z) / zz
            factor2 = np.outer(nrml, zeta) + n_O_zeta[i] * np.eye(3)
            Dtx_n = -factor1 @ factor2 / np.sqrt(zz)
            Dty_n = wedge(nrml) @ Dtx_n - wedge(z / np.sqrt(zz))

            Dtx_p = Dtx_n @ Dn_p
            Dty_p = Dty_n @ Dn_p

            DR_p = np.stack((Dtx_p, Dty_p, Dn_p), axis=1)  # (3, 3, 3)

            # compute DphatR_p, Jacobian of skew(p) @ R
            # note that here p is in the object frame, so later, when we try to
            # differentiate wrt q, we need a factor that rotates to the world frame
            Dskew_p1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
            Dskew_p2 = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
            Dskew_p3 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
            Dskew_p = (Dskew_p1, Dskew_p2, Dskew_p3)

            term1 = np.swapaxes((R.T @ np.hstack(Dskew_p)).reshape((3, 3, 3)), 0, 1)
            term2 = (wedge(p) @ DR_p.reshape((3, -1))).reshape(DR_p.shape)
            DphatR_p = term1 + term2  # (3, 3, 3)

            # compute DG_i, Jacobian of G_i wrt q
            # note the R_OW factor that rotates back to the world frame
            DG_i_p = np.concatenate((DR_p, DphatR_p), axis=0)  # (6, 3, 3)
            _shape = (DG_i_p.shape[0], 3, n)  # original shape
            DG_i = (DG_i_p.reshape((-1, 3)) @ R_OW_J).reshape(_shape)  # (6, 3, n)

            intermediate = np.copy(np.swapaxes(DG_i, 0, 1)).reshape((3, -1))  # (3, 6n)
            DW_i = np.swapaxes(
                (F.T @ intermediate).reshape((ns, 6, n)), 0, 1
            )  # (6, ns, n)

            DG[:, (3 * i) : (3 * (i + 1)), :] = DG_i
            DW[:, (ns * i) : (ns * (i + 1)), :] = DW_i
        return DG, DW

    def _compute_l(self) -> None:
        """Computes the min-weight metric and its gradient."""
        x_opt, lamb_opt, nu_opt = self._l_helper(self.W)
        self.l = x_opt[-1]
        self.Dl = self._Dl_helper(x_opt, lamb_opt, nu_opt, self.W, self.DW)

    @staticmethod
    @jit(nopython=True, fastmath=True, cache=True)
    def _l_helper(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numba jit-compiled helper method for computing l."""
        m = W.shape[-1]

        Im = np.eye(m)
        ones_m = np.ones((m, 1))
        w_bar = np.zeros(6)  # wrench whose convex hull membership is checked

        # numba-compiled quantecon simplex method
        # [NOTE] we must convert the LP to standard form to use quantecon's linprog.
        # Thus, we add slack variables for the non-negativity constraints on the
        # standard form decision variables
        c = np.concatenate((np.zeros(2 * m), np.array([1, -1])))
        top = np.concatenate((W, -W, np.zeros((6, 2))), axis=1)
        bot = np.concatenate((ones_m.T, -ones_m.T, np.zeros((1, 2))), axis=1)
        Aeq = np.concatenate((top, bot), axis=0)
        beq = np.concatenate((w_bar, np.array([1.0])))
        Ain = np.concatenate((-Im, Im, ones_m, -ones_m), axis=1)
        res = linprog(c, A_ub=Ain, b_ub=np.zeros(m), A_eq=Aeq, b_eq=beq)
        x_ = res.x
        x_p = np.concatenate((x_[:m], np.array([x_[-2]])))
        x_m = np.concatenate((x_[m : 2 * m], np.array([x_[-1]])))

        x_opt = x_p - x_m
        lamb_opt = res.lambd[:m]
        nu_opt = res.lambd[m:]

        return x_opt, lamb_opt, nu_opt

    @staticmethod
    @jit(nopython=True, fastmath=True, cache=True)
    def _Dl_helper(
        x_opt: np.ndarray,
        lamb_opt: np.ndarray,
        nu_opt: np.ndarray,
        W: np.ndarray,
        DW: np.ndarray,
    ) -> np.ndarray:
        """Numba jit-compiled helper method for computing Dl."""
        m = lamb_opt.shape[0]
        Im = np.eye(m)
        ones_m = np.ones((m, 1))

        # [NOTE] these are the inequality and equality constraint matrices in the
        # original formulation. We use these to compute the gradient, not to set up
        # the LP, which must be put in standard form for quantecon's linprog method.
        Ain = np.concatenate((-Im, ones_m), axis=1)
        Aeq = np.zeros((7, m + 1))
        Aeq[:-1, :-1] = W
        Aeq[-1, :-1] = 1.0

        # jacobian of KKT wrt W
        _DH_W1_T = np.kron(Im, nu_opt[:6]).reshape((m, m, 6))
        _DH_W1 = np.swapaxes(_DH_W1_T, -1, -2)
        _DH_W2 = np.kron(np.eye(6), x_opt[:m]).reshape((6, 6, m))
        DH_W = np.concatenate(
            (_DH_W1, np.zeros((m + 1, 6, m)), _DH_W2, np.zeros((1, 6, m)))
        )
        C = np.vstack((Ain * np.expand_dims(lamb_opt, -1), Aeq))
        _DH_W = DH_W.reshape((2 * m + 8, 6 * m))  # 3d -> 2d for lstsq

        # computing Dl explicitly using chain rule + sparsity exploit
        RHS2 = _DH_W[m + 1 :, :]  # bottom block of _DH_W
        Dl_W = -np.linalg.lstsq(C, RHS2)[0][-1, :].reshape((6, m))
        Dl = Dl_W.reshape(-1) @ DW.reshape((-1, DW.shape[-1]))
        return Dl

    def _compute_cost_func(self) -> None:
        """Computes the cost function and its gradient."""
        # cost function
        if self.baseline:
            f = 0.0
            Df = np.zeros(self.n)
        else:
            self._compute_l()
            f = -self.l
            Df = -self.Dl
        self.f = f
        self.Df = Df

    def compute_all(self, q: np.ndarray) -> None:
        """Computes and caches calculations for the robot.

        Parameters
        ----------
        q : np.ndarray, shape=(n,)
            The configuration of the system.
        """
        # updating plant context
        n = self.n
        assert q.shape == (n,)
        q_arm = q[: self.narm]
        q_hand = q[self.narm :]
        self.plant.SetPositions(self.plant_context, self.arm_instance, q_arm)
        self.plant.SetPositions(self.plant_context, self.hand_instance, q_hand)

        # computing all cached values
        self._process_collisions(q)
        self._compute_s_fk()
        self._compute_G_and_W()
        self._compute_DG_and_DW()
        self._compute_eq_cons(q)
        self._compute_cost_func()
        self._finish_ineq_cons()

    # ###################### #
    # CACHED VALUE RETRIEVAL #
    # ###################### #

    def compute_p_tips(self, q: np.ndarray) -> np.ndarray:
        """Computes the forward kinematics, p_tips."""
        if self.p_tips is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.p_tips  # (nc, 3)

    def compute_J_tips(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobians of the fingertips, J_tips."""
        if self.J_tips is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.J_tips  # (nc, 3, n)

    def compute_g(self, q: np.ndarray) -> np.ndarray:
        """Computes the inequality constraints g."""
        if self.g is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.g  # (n_ineq_cons,)

    def compute_Dg(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobian of the inequality constraints, Dg."""
        if self.Dg is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.Dg  # (n_ineq_cons, n)

    def compute_h(self, q: np.ndarray) -> np.ndarray:
        """Computes the equality constraints h."""
        if self.h is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.h  # (n_eq_cons,)

    def compute_Dh(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobian of the equality constraints, Dh."""
        if self.Dh is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.Dh  # (n_eq_cons, n)

    def compute_s_fk(self, q: np.ndarray) -> np.ndarray:
        """Computes the composition of s with FK, s_fk.

        The 0-level set of s_W is a 2D implicit surface embedded in R^3. FK is the
        forward kinematics of the robot.
        """
        if self.s_fk is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.s_fk  # (nc,)

    def compute_Ds(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobian of the SDF evaluated at the forward kinematics.

        Specifically, this computes Ds(p), where p = FK(q).
        """
        if self.Ds_p is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.Ds_p  # (nc, 3)

    def compute_Ds_fk(self, q: np.ndarray) -> np.ndarray:
        """Computes the gradient of the composition of s_W with FK wrt q, Ds_fk.

        The 0-level set of s_W is a 2D implicit surface embedded in R^3. FK is the
        forward kinematics of the robot.

        Specifically, this computes D[s(FK)](q) in contrast with Ds_p above.
        """
        if self.Ds_fk is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.Ds_fk  # (nc, n)

    def compute_gOCs(self, q: np.ndarray) -> np.ndarray:
        """Computes the transformations from contacts to object frames."""
        if self.gOCs is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.gOCs  # (nc, 4, 4)

    def compute_G(self, q: np.ndarray) -> np.ndarray:
        """Computes the grasp map, G."""
        if self.G is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.G  # (6, 3 * nc)

    def compute_DG(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobian of the grasp map, DG."""
        if self.DG is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.DG  # (6, 3 * nc, n)

    def compute_W(self, q: np.ndarray) -> np.ndarray:
        """Computes the wrench matrix whose columns are the primitive wrenches, W."""
        if self.W is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.W  # (6, ns * nc)

    def compute_DW(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobian of the wrench matrix, Dw."""
        if self.DW is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.DW  # (6, ns * nc, n)

    def compute_l(self, q: np.ndarray) -> float:
        """Computes the optimal minimum convex weight l."""
        if self.l is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.l

    def compute_Dl(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobian of l, Dl."""
        if self.Dl is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.Dl  # (n,)

    def compute_f(self, q: np.ndarray) -> float:
        """Computes the cost function f."""
        if self.f is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.f

    def compute_Df(self, q: np.ndarray) -> np.ndarray:
        """Computes the gradient of the cost, Df."""
        if self.Df is None or np.any(q != self.q):
            self.compute_all(q)
            self.q = np.copy(q)
        return self.Df  # (n,)

    # ##### #
    # UTILS #
    # ##### #

    def viz_config(self, q: np.ndarray) -> None:
        """Visualizes a configuration q in meshcat."""
        assert q.shape == (self.n,)
        assert self.viz
        q_arm = q[: self.narm]
        q_hand = q[self.narm :]
        X_WO = self.obj.X_WO
        obj_quat = X_WO.rotation().ToQuaternion().wxyz()
        obj_pos = X_WO.translation()
        q_obj = np.concatenate((obj_quat, obj_pos))
        q_all = np.concatenate((q, q_obj))
        self.plant.SetPositions(self.plant_context, self.arm_instance, q_arm)
        self.plant.SetPositions(self.plant_context, self.hand_instance, q_hand)
        self.plant.SetPositions(self.plant_context, self.obj_instance, q_obj)
        self.sliders.SetPositions(q_all)
        self.sliders.Run(self.diagram, None)
