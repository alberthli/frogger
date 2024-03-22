from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from numba import jit
from pydrake.geometry import (
    AddContactMaterial,
    AddRigidHydroelasticProperties,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    ProximityProperties,
    QueryObject,
    Rgba,
    Role,
    StartMeshcat,
)
from pydrake.math import RigidTransform
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph,
    CoulombFriction,
    DiscreteContactSolver,
)
from pydrake.multibody.tree import JacobianWrtVariable, SpatialInertia, UnitInertia
from pydrake.systems.framework import DiagramBuilder
from quantecon.optimize.linprog_simplex import linprog_simplex as linprog

from frogger import ROOT
from frogger.grasping import (
    compute_gOCs,
    compute_grasp_matrix,
    compute_primitive_forces,
    wedge,
)
from frogger.metrics import min_weight_gradient, min_weight_lp
from frogger.objects import ObjectDescription


class RobotModel:
    """Base class for robot models.

    This class serves as a container for caching computation efficiently so that
    repeated calculations are minimized.

    [NOTE] any geometries belonging to the fingertip that are ALLOWED to collide with
    the object should have the substring "FROGGERCOL" in their *.sdf file.
    """

    # ############## #
    # INITIALIZATION #
    # ############## #

    def __init__(self, cfg: "RobotModelConfig") -> None:
        """Initialize the robot model.

        Parameters
        ----------
        cfg : RobotModelConfig
            The configuration of the robot model.
        """
        # cfg variables
        model_path = cfg.model_path
        self.obj = cfg.obj
        self.ns = cfg.ns
        self.mu = cfg.mu
        self.d_min = cfg.d_min
        self.d_pen = cfg.d_pen
        self.l_bar_cutoff = cfg.l_bar_cutoff
        self.viz = cfg.viz
        self.custom_compute_l = (
            cfg.__class__.custom_compute_l
        )  # these will be class functions
        self.custom_compute_g = cfg.__class__.custom_compute_g
        self.custom_compute_h = cfg.__class__.custom_compute_h
        self.finger_level_set = cfg.finger_level_set
        self.n_g_extra = cfg.n_g_extra
        self.n_h_extra = cfg.n_h_extra
        self.cfg = cfg

        # boilerplate drake code + initializing the robot
        self.builder = DiagramBuilder()
        self.plant, self.scene_graph = AddMultibodyPlantSceneGraph(
            self.builder, time_step=0.001
        )
        self.parser = Parser(self.plant, self.scene_graph)
        self.parser.package_map().Add("frogger", ROOT)

        # adding robot + object
        self.robot_instance = self.parser.AddModels(f"{ROOT}/models/{model_path}")[0]
        self._load_object()

        # additional plant setup - if you want additional modifications to plants in
        # derived classes, overwrite this method
        self._plant_additions()
        self.plant.Finalize()

        # visualization settings
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
                JointSliders(self.meshcat, self.plant, step=1e-2)
            )

        # creating diagrams, subsystems, and subcontexts
        self.diagram = self.builder.Build()
        self.plant = self.diagram.GetSubsystemByName("plant")
        self.scene_graph = self.diagram.GetSubsystemByName("scene_graph")

        self.diag_context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.diag_context)
        self.sg_context = self.scene_graph.GetMyMutableContextFromRoot(
            self.diag_context
        )
        self._qo_port = self.scene_graph.get_query_output_port()

        # setting initial object pose
        self.plant.SetFreeBodyPose(self.plant_context, self.obj_body, self.obj.X_WO)

        # all the possible hand-obj collision pairs, which are always checked
        _robot_collision_bodies = []
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
                if "FROGGERCOL" in name_A:
                    fid = inspector.GetFrameId(id_A)
                else:
                    fid = inspector.GetFrameId(id_B)
                robot_body = self.plant.GetBodyFromFrameId(fid).body_frame().body()
                if robot_body not in _robot_collision_bodies:
                    _robot_collision_bodies.append(robot_body)

        # internal dimensions
        self.nc = len(_robot_collision_bodies)  # number of contact points
        self.n = self.plant.num_positions() - 7  # exclude the object pose states
        self.n_couple = cfg.n_couple

        # additional setup for constraints
        self._create_bound_cons()
        self.F = compute_primitive_forces(self.ns, self.mu)  # primitive force matrix
        self._add_coupling_constraints()

        # cached values. all Jacobians wrt q when unspecified
        self.q = None  # cached value of the last unique configuration
        self.p_tips = None  # cached values of the fingertip FK and Jacobians
        self.J_tips = None
        self.P_OF = (
            None  # cached values of tips and contact normals in object/world frame
        )
        self.n_O = None
        self.n_W = None
        self.R_cf_O = None  # contact frames of each finger expressed in obj frame
        self.DR_cf_O = None
        self.Ds_p = None  # cached value of Ds evaluated at fingertips
        self.h = None  # cached values of the eq constraint function and Jacobian
        self.Dh = None
        self.g = None  # cached values of the ineq constraint function and Jacobian
        self.Dg = None
        self.f = None  # cached values of the cost function and Jacobian
        self.Df = None
        self.G = None  # cached values of the grasp map and Jacobian
        self.DG = None
        self.W = None  # cached values of wrench basis matrix and Jacobian
        self.DW = None
        self.l = None  # cached values of min convex weight LP and Jacobian
        self.Dl = None
        self.gid_pair_inds = {}  # dict ordering collision geometry id pairs
        self.sdp_normals = {}  # dict caching signed distance pair normal vectors
        self.hand_obj_cols = {}  # dict recording most penetrating hand/obj collisions

    def warm_start(self) -> None:
        """Convenience method to warm start the compilation.

        If you want to warm start the system, this must be manually called after it is
        initialized OR the create function will automatically call it.
        """
        self.compute_l(self.plant.GetPositions(self.plant_context, self.robot_instance))

    def _plant_additions(self) -> None:
        """Additional modifications to the plant in derived classes."""

    def _add_coupling_constraints(self) -> None:
        """Adds coupling constraints in derived classes."""

    def _load_object(self) -> None:
        """Loads the object into the plant."""
        # registering object's visual/collision geoms with the plant
        assert None not in [self.obj.shape_visual, self.obj.shape_collision_list]
        self.obj_instance = self.plant.AddModelInstance("obj")
        I_o = self.obj.inertia / self.obj.mass
        self.obj_body = self.plant.AddRigidBody(
            "obj",
            self.obj_instance,
            SpatialInertia(
                self.obj.mass,
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
            self.obj.shape_visual,
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
        for i, shape_col in enumerate(self.obj.shape_collision_list):
            self.plant.RegisterCollisionGeometry(
                self.obj_body,
                RigidTransform(),
                shape_col,
                f"obj_collision_{i}",
                prox_properties,
            )

    def _create_bound_cons(self) -> None:
        """Creates the lower and upper bound constraints."""
        lb_q, ub_q = self.q_bounds

        # finite lower and upper bounds
        lb_inds = ~np.isinf(lb_q)
        ub_inds = ~np.isinf(ub_q)

        # creating constraint matrices
        self.n_bounds = np.sum(lb_inds) + np.sum(ub_inds)
        _A_box_lb = np.diag(lb_inds).astype(float)
        _A_box_ub = np.diag(ub_inds).astype(float)
        A_box_lb = _A_box_lb[~np.all(_A_box_lb == 0.0, axis=1)]
        A_box_ub = _A_box_ub[~np.all(_A_box_ub == 0.0, axis=1)]
        b_box_lb = lb_q[lb_inds]
        b_box_ub = ub_q[ub_inds]
        self.A_box = np.concatenate((-A_box_lb, A_box_ub))
        self.b_box = np.concatenate((b_box_lb, -b_box_ub))

    def _init_ineq_cons(self) -> None:
        """Initializes the inequality constraint data structures."""
        n = self.n

        # sort all possible collision pairs and initialize constraint info
        # gidps means geometry id pairs
        col_cands = list(self.query_object.inspector().GetCollisionCandidates())
        gidps = [(c[0].get_value(), c[1].get_value()) for c in col_cands]
        _, col_cands = zip(*sorted(zip(gidps, col_cands)))  # sort cands by gidps

        # the first 2n constraints are box constraints, followed by collision
        # constraints, and minimum min-weight metric value
        self.n_pairs = len(col_cands)
        self.g = np.zeros(self.n_bounds + self.n_pairs + self.n_g_extra)
        self.Dg = np.zeros((self.n_bounds + self.n_pairs + self.n_g_extra, n))
        self.Dg[: self.n_bounds, :] = self.A_box

        # initializing a dictionary that orders gid pairs
        # this lets us update specific signed distances in a well-ordered way
        for i, c in enumerate(col_cands):
            id_A = c[0]
            id_B = c[1]
            self.gid_pair_inds[(id_A, id_B)] = i

    def _init_eq_cons(self) -> None:
        """Initializes the equality constraint data structures."""
        self.h = np.zeros(self.n_couple + self.nc + self.n_h_extra)
        self.Dh = np.zeros((self.n_couple + self.nc + self.n_h_extra, self.n))

    # ########## #
    # PROPERTIES #
    # ########## #

    @property
    def q_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounds on the configuration from the description file.

        Returns
        -------
        lb_q : np.ndarray, shape=(n,)
            Lower bounds.
        ub_q : np.ndarray, shape=(n,)
            Upper bounds.
        """
        return (
            self.plant.GetPositionLowerLimits()[: self.n],
            self.plant.GetPositionUpperLimits()[: self.n],
        )

    @property
    def query_object(self) -> QueryObject:
        """The query object."""
        return self._qo_port.Eval(self.sg_context)

    # ####### #
    # SETTERS #
    # ####### #

    def set_X_WO(self, X_WO_new: RigidTransform) -> None:
        """Sets a new object pose."""
        self.obj.set_X_WO(X_WO_new)
        self.plant.SetFreeBodyPose(self.plant_context, self.obj_body, X_WO_new)

    def set_q(self, q: np.ndarray) -> None:
        """Sets the robot state."""
        assert len(q) == self.n
        self.plant.SetPositions(self.plant_context, self.robot_instance, q)
        self.q = np.copy(q)

    # ######################## #
    # CACHED VALUE COMPUTATION #
    # ######################## #

    def _process_collisions(self) -> None:
        """Computes all information related to collision constraints."""
        # we initialize the inequality constraints here to allow derived RobotModels
        # to modify the collision geometries during initialization
        if self.g is None:
            self._init_ineq_cons()

        # update joint limit constraint values
        self.g[: self.n_bounds] = self.A_box @ self.q + self.b_box

        # collisions get computed conditionally after culling w/max_distance.
        # everything sufficiently far away we ignore and set the gradient to 0
        # d_min is the desired safety margin, a minimum distance enforced between geoms
        d_min = self.d_min
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

        self.g[
            self.n_bounds : (self.n_bounds + self.n_pairs)
        ] = -1.0  # setting "far" points to 1.0m
        self.Dg[
            self.n_bounds : (self.n_bounds + self.n_pairs), :
        ] = 0.0  # resetting col gradients

        # loop through unculled collision pairs
        self.hand_obj_cols = {}  # reset the hand-obj dictionary
        for sdp in sdps:
            id_A, id_B = sdp.id_A, sdp.id_B  # geometry IDs
            sd = sdp.distance  # signed distance
            nrml = sdp.nhat_BA_W  # direction of fastest increase outward from point B
            pA, pB = sdp.p_ACa, sdp.p_BCb  # witness points A and B
            fA, fB = get_bf(id_A), get_bf(id_B)  # frames associated with points A and B

            # computing/retrieving cached Jacobians, shape=(3, n)
            J_A = self.plant.CalcJacobianTranslationalVelocity(
                self.plant_context,
                JacobianWrtVariable.kQDot,
                fA,
                pA,
                self.plant.world_frame(),
                self.plant.world_frame(),
            )[..., : self.n]
            J_B = self.plant.CalcJacobianTranslationalVelocity(
                self.plant_context,
                JacobianWrtVariable.kQDot,
                fB,
                pB,
                self.plant.world_frame(),
                self.plant.world_frame(),
            )[..., : self.n]

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
            d_pen = self.d_pen
            if has_tip and has_obj:
                self.g[self.n_bounds + i] = -sd - d_pen  # allow tips to penetrate obj
            else:
                self.g[self.n_bounds + i] = d_min - sd  # other pairs must respect d_min
            Dgi = -(J_A - J_B).T @ nrml
            self.Dg[self.n_bounds + i, :] = Dgi

            # updating the most interpenetrating pairs for each link allowing collision
            if has_tip and has_obj:
                bA, bB = fA.body(), fB.body()  # bodies associated with collision geoms
                body_name_A, body_name_B = bA.name(), bB.name()

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

                if key not in self.hand_obj_cols or sd < self.hand_obj_cols[key][0]:
                    self.hand_obj_cols[key] = (sd, -Dgi, p_tip_W, p_tip_C, f_tip)

        # updating p_tips and J_tips
        h = []
        Dh = []
        p_tips = []
        J_tips = []
        for _, v in sorted(self.hand_obj_cols.items()):
            h.append(
                v[0] - self.finger_level_set
            )  # enforce that the tips lie on a level set
            Dh.append(v[1])
            p_tips.append(v[2])
            J_tips.append(
                self.plant.CalcJacobianTranslationalVelocity(
                    self.plant_context,
                    JacobianWrtVariable.kQDot,
                    v[4],
                    v[3],
                    self.plant.world_frame(),
                    self.plant.world_frame(),
                )[..., : self.n]
            )
        self.h_tip = np.array(h)
        self.Dh_tip = np.array(Dh)
        self.p_tips = np.array(p_tips)
        self.J_tips = np.array(J_tips)

    def _finish_ineq_cons(self) -> None:
        """Finishes computing inequality constraints and their gradients."""
        if self.custom_compute_g is None:
            # the default behavior is to enforce a minimum l* value
            if self.n_g_extra == 1:
                idx_minw = self.n_pairs + self.n_bounds
                self.g[idx_minw] = -self.l + self.l_bar_cutoff / (self.ns * self.nc)
                self.Dg[idx_minw, :] = -self.Dl

        else:
            g_extra, Dg_extra = self.custom_compute_g(self)
            if isinstance(g_extra, float):
                assert self.n_g_extra == 1
            elif isinstance(g_extra, np.ndarray):
                assert len(g_extra) == self.n_g_extra
            self.g[-self.n_g_extra :] = g_extra
            self.Dg[-self.n_g_extra :, :] = Dg_extra

    def _compute_G_and_W(self) -> None:
        """Computes the grasp and wrench matrices."""
        # computing the grasp map in the object frame using known pose
        X_WO = self.obj.X_WO  # pose of obj wrt world
        X_OW = X_WO.inverse()
        R_OW = X_WO.inverse().rotation().matrix()

        # contact points and normals in the object frame
        self.P_OF = X_OW @ self.p_tips.T  # (3, nc)
        Ds_p = self.obj.Ds_W(self.p_tips, batched=True)  # Ds(p)
        self.n_W = -Ds_p.T / np.linalg.norm(Ds_p, axis=1)  # (3, nc)
        self.n_O = R_OW @ self.n_W  # INWARD pointing normals

        self.gOCs = compute_gOCs(self.P_OF, self.n_O)  # (nc, 4, 4)
        self.G = compute_grasp_matrix(self.gOCs)  # (6, 3 * nc), object frame
        self.W = self.G @ np.kron(np.eye(self.nc), self.F)  # cols = primitive wrenches

    def _compute_DG_and_DW(self) -> None:
        """Computes the Jacobians of the grasp and wrench matrices."""
        J_T = np.swapaxes(self.J_tips, -1, -2)  # (nc, n, 3)
        X_WO = self.obj.X_WO  # pose of obj wrt world
        R_OW = X_WO.inverse().rotation().matrix()

        Ds_O_ps = self.obj.Ds_O(self.P_OF.T, batched=True)
        D2s_O_ps = self.obj.D2s_O(self.P_OF.T, batched=True)
        self.DG, self.DW, self.R_cf_O, self.DR_cf_O = self._DG_DW_helper(
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

        This was refactored into a very hard-to-read way to be compatible with numba.
        Apologies in advance for any future readers.
        """
        nc, n = J_T.shape[:2]
        ns = F.shape[-1]

        # need to consider the change from object to world frame
        DG = np.empty((6, 3 * nc, n))
        DW = np.empty((6, nc * ns, n))

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

        # caching the contact frames and their Jacobians wrt q
        R_cf_O = np.zeros((nc, 3, 3))
        DR_cf_O = np.zeros((nc, 3, 3, n))

        for i in range(nc):
            p = P_OF.T[i, :]  # P_OF.T has shape (nc, 3), select ith row
            nrml = n_O.T[i, :]  # n_O.T has shape (nc, 3), select ith row
            R_OW_J = R_OW_J_Ts[i].T  # (3, n)

            # compute R, the rotation matrix for this contact frame
            z = zeta - n_O_zeta[i] * nrml
            zz = z @ z
            tx = z / np.sqrt(zz)
            ty = np.cross(nrml, tx)
            R = np.stack((tx, ty, nrml)).T  # contact frame in object frame
            R_cf_O[i, ...] = R

            # compute DR_p, Jacobian of rotation matrix wrt p in object frame
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

            # computing the Jacobian of R wrt q - useful for nerf grasping
            DR = (DR_p.reshape((9, -1)) @ R_OW_J).reshape((3, 3, n))  # (3, 3, n)
            DR_cf_O[i, ...] = DR

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

        return DG, DW, R_cf_O, DR_cf_O

    def _compute_l(self) -> None:
        """Computes the min-weight metric and its gradient."""
        if self.custom_compute_l is None:
            x_opt, lamb_opt, nu_opt = min_weight_lp(self.W)
            self.l = x_opt[-1]
            self.Dl = min_weight_gradient(x_opt, lamb_opt, nu_opt, self.W, self.DW)
        else:
            self.l, self.Dl = self.custom_compute_l(self)

    def _compute_cost_func(self) -> None:
        """Computes the cost function and its gradient."""
        self.f = -self.l
        self.Df = -self.Dl

    def _compute_eq_cons(self) -> None:
        """Computes the equality constraints."""
        # computing coupling and contact constraints
        if self.n_couple != 0:
            h_couple = self.A_couple @ self.q + self.b_couple
            Dh_couple = self.A_couple
            self.h[: self.n_couple + self.nc] = np.concatenate((self.h_tip, h_couple))
            self.Dh[: self.n_couple + self.nc, :] = np.concatenate(
                (self.Dh_tip, Dh_couple), axis=0
            )
        else:
            self.h[: self.nc] = self.h_tip
            self.Dh[: self.nc, :] = self.Dh_tip

        # computing the extra equality constraints
        if self.custom_compute_h is not None:
            h_extra, Dh_extra = self.custom_compute_h(self)
            assert len(h_extra) == self.n_h_extra
            self.h[self.n_couple + self.nc :] = h_extra
            self.Dh[self.n_couple + self.nc :, :] = Dh_extra

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
        self.set_q(q)  # also caches self.q

        # computing all cached values
        if self.h is None:
            self._init_eq_cons()
        self._process_collisions()
        self._compute_G_and_W()
        self._compute_DG_and_DW()
        self._compute_l()
        self._compute_cost_func()
        self._finish_ineq_cons()
        self._compute_eq_cons()

    # ###################### #
    # CACHED VALUE RETRIEVAL #
    # ###################### #

    def compute_p_tips(self, q: np.ndarray) -> np.ndarray:
        """Computes the forward kinematics, p_tips."""
        if self.p_tips is None or np.any(q != self.q):
            self.compute_all(q)
        return self.p_tips  # (nc, 3)

    def compute_J_tips(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobians of the fingertips, J_tips."""
        if self.J_tips is None or np.any(q != self.q):
            self.compute_all(q)
        return self.J_tips  # (nc, 3, n)

    def compute_n_O(self, q: np.ndarray) -> np.ndarray:
        """Computes the invward contact normals in the object frame, n_O."""
        if self.n_O is None or np.any(q != self.q):
            self.compute_all(q)
        return self.n_O.T  # (nc, 3)

    def compute_n_W(self, q: np.ndarray) -> np.ndarray:
        """Computes the inward contact normals in the world frame, n_W."""
        if self.n_W is None or np.any(q != self.q):
            self.compute_all(q)
        return self.n_W.T  # (nc, 3)

    def compute_R_cf_O(self, q: np.ndarray) -> np.ndarray:
        """Computes the contact frames of each finger expressed in the object frame."""
        if self.R_cf_O is None or np.any(q != self.q):
            self.compute_all(q)
        return self.R_cf_O

    def compute_DR_cf_O(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobian of the contact frames in the object frame."""
        if self.DR_cf_O is None or np.any(q != self.q):
            self.compute_all(q)
        return self.DR_cf_O

    def compute_g(self, q: np.ndarray) -> np.ndarray:
        """Computes the inequality constraints g."""
        if self.g is None or np.any(q != self.q):
            self.compute_all(q)
        return self.g  # (n_ineq_cons,)

    def compute_Dg(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobian of the inequality constraints, Dg."""
        if self.Dg is None or np.any(q != self.q):
            self.compute_all(q)
        return self.Dg  # (n_ineq_cons, n)

    def compute_h(self, q: np.ndarray) -> np.ndarray:
        """Computes the equality constraints h."""
        if self.h is None or np.any(q != self.q):
            self.compute_all(q)
        return self.h  # (n_eq_cons,)

    def compute_Dh(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobian of the equality constraints, Dh."""
        if self.Dh is None or np.any(q != self.q):
            self.compute_all(q)
        return self.Dh  # (n_eq_cons, n)

    def compute_Ds(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobian of the SDF evaluated at the forward kinematics.

        Specifically, this computes Ds(p), where p = FK(q).
        """
        if self.Ds_p is None or np.any(q != self.q):
            self.compute_all(q)
        return self.Ds_p  # (nc, 3)

    def compute_gOCs(self, q: np.ndarray) -> np.ndarray:
        """Computes the transformations from contacts to object frames."""
        if self.gOCs is None or np.any(q != self.q):
            self.compute_all(q)
        return self.gOCs  # (nc, 4, 4)

    def compute_G(self, q: np.ndarray) -> np.ndarray:
        """Computes the grasp map, G."""
        if self.G is None or np.any(q != self.q):
            self.compute_all(q)
        return self.G  # (6, 3 * nc)

    def compute_DG(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobian of the grasp map, DG."""
        if self.DG is None or np.any(q != self.q):
            self.compute_all(q)
        return self.DG  # (6, 3 * nc, n)

    def compute_W(self, q: np.ndarray) -> np.ndarray:
        """Computes the wrench matrix whose columns are the primitive wrenches, W."""
        if self.W is None or np.any(q != self.q):
            self.compute_all(q)
        return self.W  # (6, ns * nc)

    def compute_DW(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobian of the wrench matrix, Dw."""
        if self.DW is None or np.any(q != self.q):
            self.compute_all(q)
        return self.DW  # (6, ns * nc, n)

    def compute_l(self, q: np.ndarray) -> float:
        """Computes the optimal minimum convex weight l (or other metric value)."""
        if self.l is None or np.any(q != self.q):
            self.compute_all(q)
        return self.l

    def compute_Dl(self, q: np.ndarray) -> np.ndarray:
        """Computes the Jacobian of l, Dl."""
        if self.Dl is None or np.any(q != self.q):
            self.compute_all(q)
        return self.Dl  # (n,)

    def compute_f(self, q: np.ndarray) -> float:
        """Computes the cost function f. Negative of the metric l."""
        if self.f is None or np.any(q != self.q):
            self.compute_all(q)
        return self.f

    def compute_Df(self, q: np.ndarray) -> np.ndarray:
        """Computes the gradient of the cost, Df."""
        if self.Df is None or np.any(q != self.q):
            self.compute_all(q)
        return self.Df  # (n,)

    # ##### #
    # UTILS #
    # ##### #

    def viz_config(self, q: np.ndarray) -> None:
        """Visualizes a configuration q in meshcat."""
        assert q.shape == (self.n,)
        assert self.viz
        X_WO = self.obj.X_WO
        obj_quat = X_WO.rotation().ToQuaternion().wxyz()
        obj_pos = X_WO.translation()
        q_obj = np.concatenate((obj_quat, obj_pos))
        q_all = np.concatenate((q, q_obj))
        self.plant.SetPositions(self.plant_context, q_all)
        self.sliders.SetPositions(q_all)
        self.sliders.Run(self.diagram, None)

    def introspect_collisions(
        self, q: np.ndarray | None = None, level: float = -1e-6
    ) -> None:
        """Introspects collisions for most recently processed configuration.

        Parameters
        ----------
        q : np.ndarray | None, default=None
            The configuration to introspect. If None, uses the most recent configuration.
        level : float, default=-1e-6
            The level of penetration to introspect. Any pair with signed distance less than
            level will be printed.
        """
        if q is not None:
            self.compute_all(q)

        # print the names of the collision pairs in violation
        d_min = self.d_min
        sdps = self.query_object.ComputeSignedDistancePairwiseClosestPoints(
            max_distance=(d_min + 0.001)
        )
        inspector = self.query_object.inspector()  # model inspector for geometries

        # loop through unculled collision pairs
        for sdp in sdps:
            id_A, id_B = sdp.id_A, sdp.id_B  # geometry IDs
            sd = sdp.distance  # signed distance
            names = [inspector.GetName(id_A), inspector.GetName(id_B)]
            if sd <= level:
                print(names)


@dataclass(kw_only=True)
class RobotModelConfig:
    """A configuration for a robot model.

    Parameters
    ----------
    model_path : str
        The path of the description relative to the ROOT/models directory in the repo.
    obj : ObjectDescription
        The description of the object.
    ns : int, default=4
        The number of sides of the pyramidal friction cone approximation.
    mu : float, default=0.7
        The coefficient of friction.
    d_min : float, default=0.001
        The minimum distance between any two collision geometries.
    d_pen : float, default=0.001
        The allowable penetration between the fingertips and the object.
    l_bar_cutoff : float, default=1e-6
        The minimum allowable value of l_bar.
    n_couple : int, default=0
        The number of coupling constraints.
    finger_level_set : float, default=0.0
        The object level set that the fingertips should lie on.
    name : str | None, default=None
        The name of the robot.
    viz : bool, default=True
        Whether to visualize the robot.
    custom_compute_l : Callable[[RobotModel], Tuple[np.ndarray, np.ndarray]] | None
        A custom cost function that takes in the robot and returns l and Dl. This ensures that
        the callback has access to whatever info might be necessary.
    custom_compute_g : Callable[[RobotModel], Tuple[np.ndarray, np.ndarray]] | None
        A custom inequality constraint function that returns extra inequality constraints.
    custom_compute_h : Callable[[RobotModel], Tuple[np.ndarray, np.ndarray]] | None
        A custom equality constraint function that returns extra equality constraints.
    n_g_extra : int, default=1
        The number of "extra" inequality constraints. Extra is defined as anything that is
        not a box constraint or a collision constraint.
    n_h_extra : int, default=0
        The number of "extra" equality constraints. Extra is defined as anything that is
        not a coupling or contact constraint.

    WARNING
    -------
    When defining the custom callbacks, it is important to note the order that they
    execute, in case they rely on cached values. The function `compute_all` shows that first,
    l is computed, then the extra inequality constraints, then the equality constraints. The only
    caveat to this is that the non-extra inequality constraints (collision and joint limits) are
    computed before l.
    """

    # required
    model_path: str | None = None
    obj: ObjectDescription | None = None

    # optional
    ns: int = 4
    mu: float = 0.7
    d_min: float = 0.001
    d_pen: float = 0.001
    l_bar_cutoff: float = 1e-6
    n_couple: int = 0
    finger_level_set: float = 0.0
    name: str | None = None
    viz: bool = True
    custom_compute_l: Callable[
        [RobotModel], Tuple[np.ndarray, np.ndarray]
    ] | None = None
    custom_compute_g: Callable[
        [RobotModel], Tuple[np.ndarray, np.ndarray]
    ] | None = None
    custom_compute_h: Callable[
        [RobotModel], Tuple[np.ndarray, np.ndarray]
    ] | None = None
    n_g_extra: int = 1
    n_h_extra: int = 0

    # model type to create - shouldn't override during initialization, only when subclassing
    model_class: type = RobotModel

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        # enforcing required parameters manually, since model_path is provided by
        # derived classes
        if self.model_path is None:
            raise AttributeError
        if self.obj is None:
            raise AttributeError

        # handling default values manually, since derived classes have different ones
        if self.name is None:
            self.name = "robot"

    def create_pre_warmstart(self, model: RobotModel) -> None:
        """Entrypoint into the create() function before the warm start."""

    def create(self) -> RobotModel:
        """Creates the robot model."""
        model = self.model_class(self)
        self.create_pre_warmstart(model)
        model.warm_start()
        return model
