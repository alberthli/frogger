from pathlib import Path

import numpy as np
import scipy
from pydrake.common.eigen_geometry import Quaternion
from pydrake.common.value import AbstractValue
from pydrake.geometry import MeshcatVisualizer, StartMeshcat
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
)
from pydrake.math import RigidTransform
from pydrake.solvers import MathematicalProgram, MosekSolver, SnoptSolver
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder, EventStatus, LeafSystem
from pydrake.systems.primitives import StateInterpolatorWithDiscreteDerivative
from pydrake.trajectories import PiecewisePose

from core.grasping import vee
from core.robots.robot_core import RobotModel

current_dir = Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1] == "manipulation"][0]
ROOT = str(project_dir)


class PickupController(LeafSystem):
    """Controller for picking an object.

    For simplicity, assumes the robot is initialized in a configuration where it can
    immediately execute a grasp on the object.
    """

    def __init__(
        self,
        model: RobotModel,
        t_lift: float,
        lift_duration: float,
        lift_height: float,
    ) -> None:
        """Initialize the controller.

        Parameters
        ----------
        model : RobotModel
            The robot model, which includes the entire scene and already has gravity
            compensation automatically loaded.
        t_lift : float
            The time at which to lift the arm.
        lift_duration : float
            The desired duration of the lifting motion. Used for trajectory planning.
        lift_height : float
            The desired height above the tabletop to reach.
        """
        super().__init__()

        # extracting key parameters for convenience
        self.model = model
        self.t_lift = t_lift
        self.lift_duration = lift_duration
        self.lift_height = lift_height

        self.ns = model.ns
        self.nc = model.nc
        self.mu = model.mu
        self.n = model.n
        self.narm = model.narm
        self.nhand = model.nhand
        self.nu = model.nu
        self.nuhand = model.nuhand
        self.obj = model.obj
        self.obj_mass = self.obj.mass
        self.arm_instance = model.arm_instance
        self.hand_instance = model.hand_instance
        self.obj_instance = model.obj_instance
        self.lb_tau, self.ub_tau = self.model._tau_bounds()

        # initializing controller's internal model of the plant
        self.model_plant = model.plant
        self.arm_plant = model.arm_plant  # only the arm - used for diffIK

        # cached values
        self.q0 = None  # initial configuration
        self.P_OCo = None  # desired contact points on the object, (nc, 3)
        self.n_O = None  # desired outward contact normals on the object, (nc, 3)
        self.f_sol_cached = None

        # pre-computing quantities used for optimization + setting up solver
        # use MOSEK if available, otherwise default to SNOPT
        self.prog = MathematicalProgram()
        _solver = MosekSolver()
        if _solver.available():
            self.solver = _solver
        else:
            self.solver = SnoptSolver()
        self.f = self.prog.NewContinuousVariables(3 * self.nc, "f")

        # constraint 1: linearized friction cone (unchanging w/new data)
        # minimum normal force
        if self.obj.mass <= 0.01:
            fn_min = 0.25  # for very light objects, don't require as much force
        else:
            fn_min = 1.0
        Lambda_i = np.vstack(
            (
                -np.append(np.cos(2 * np.pi * np.arange(self.ns) / self.ns), 0.0),
                -np.append(np.sin(2 * np.pi * np.arange(self.ns) / self.ns), 0.0),
                np.append(self.mu * np.cos(np.pi * np.ones(self.ns) / self.ns), 1.0),
            )
        ).T  # pyramidal friction cone approx in contact frame + min normal force
        self.Lambda = np.kron(np.eye(self.nc), Lambda_i)  # ((ns + 1) * nc, 3 * nc)
        lb = np.zeros(((self.ns + 1) * self.nc, 1))
        lb[self.ns :: self.ns + 1] = fn_min
        ub = np.full(((self.ns + 1) * self.nc, 1), np.inf)
        self.prog.AddLinearConstraint(self.Lambda, lb, ub, self.f)

        # constraint 2: torque limits (changes with measurements)
        A_dum = np.zeros((self.nuhand, self.nc * 3))
        self.lb_hand = self.lb_tau[-self.nuhand :]  # unchanging
        self.ub_hand = self.ub_tau[-self.nuhand :]
        self.cons_tau = self.prog.AddLinearConstraint(
            A_dum, self.lb_hand, self.ub_hand, self.f
        )

        # cost function (changes with measurements)
        Q_dum = np.zeros((3 * self.nc, 3 * self.nc))
        b_dum = np.zeros(3 * self.nc)
        self.cost = self.prog.AddQuadraticCost(Q_dum, b_dum, self.f)

        # input/output ports to the controller block
        self.DeclareVectorInputPort("model_state_in", 2 * self.n)
        self.DeclareVectorInputPort("x_arm_des_in", 2 * self.narm)
        self.DeclareAbstractInputPort(
            "body_poses_in", AbstractValue.Make([RigidTransform()])
        )  # assume an external system measures the object pose perfectly
        self.DeclareVectorOutputPort("tau_command_out", self.nu, self.compute_tau)
        self.DeclareAbstractOutputPort(
            "X_WE_des_out",
            lambda: AbstractValue.Make(RigidTransform()),
            self.compute_X_WE_des,
        )

    def initialize_ctrl(self, q0: np.ndarray) -> None:
        """Sets the initial configuration. Used for early stabilization of arm.

        Further, when q0 is changed, a new trajectory is computed for the pickup
        sequence from that initial condition.
        """
        # compute the initial wrist position
        self.q0 = q0
        arm_plant = self.model.arm_plant
        arm_plant_context = self.model.arm_plant.CreateDefaultContext()
        arm_plant.SetPositions(arm_plant_context, q0[: self.narm])

        # compute the current object pose relative to the palm center, set to desired
        self.model_plant.SetPositions(
            self.model.plant_context, self.arm_instance, q0[: self.narm]
        )
        self.model_plant.SetPositions(
            self.model.plant_context,
            self.hand_instance,
            q0[self.narm : (self.narm + self.nhand)],
        )

        # computing end-effector trajectory for diffik to track
        ee_body_index = self.model_plant.GetBodyIndices(self.arm_instance)[-1]
        ee_body = self.model_plant.get_body(ee_body_index)
        X_WE = self.model_plant.EvalBodyPoseInWorld(self.model.plant_context, ee_body)

        X_WE_0 = X_WE
        X_WE_f = RigidTransform(
            X_WE_0.rotation(),
            X_WE_0.translation() + np.array([0.0, 0.0, self.lift_height]),
        )

        t_lift = self.t_lift
        lift_duration = self.lift_duration
        t_list = np.linspace(0, t_lift + lift_duration, 101)
        X_WE_list = []
        for t in t_list:
            if t <= t_lift:
                X_WE_list.append(X_WE_0)
            else:
                # after t_lift, linearly interpolate the desired height
                # also, inject a high-frequency sinusoidal disturbance a short time
                # after the lift begins
                trans0 = X_WE_0.translation()
                transf = X_WE_f.translation()
                transt = (transf - trans0) * ((t - t_lift) / lift_duration) + trans0
                if t >= t_lift + 0.25:
                    transt[0] = transt[0] + 0.003 * np.sin(5.0 * t)
                    transt[1] = transt[1] + 0.003 * np.cos(4.0 * t)
                    transt[2] = transt[2] + 0.003 * np.cos(3.0 * t)
                X_WE_list.append(RigidTransform(X_WE_0.rotation(), transt))

        self.X_WE_traj = PiecewisePose.MakeCubicLinearWithEndLinearVelocity(
            t_list,
            X_WE_list,
            np.zeros(3),
            np.zeros(3),
        )

        # computing object pose for pose regulation
        obj_body_index = self.model_plant.GetBodyIndices(self.obj_instance)[0]
        obj_body = self.model_plant.get_body(obj_body_index)
        X_WO = self.model_plant.EvalBodyPoseInWorld(self.model.plant_context, obj_body)
        self.X_WO_last = None
        self.X_EO_des = X_WE_0.inverse() @ X_WO

    def set_pn_contact(self, P_OCo: np.ndarray, n_O: np.ndarray) -> None:
        """Sets the contact positions and normals in the object frame."""
        self.P_OCo = P_OCo
        self.n_O = n_O

    def compute_X_WE_des(self, context, X_WE_des) -> None:
        """Drake desired end effector pose callback function."""
        t = context.get_time()
        X_WE_des.set_value(self.X_WE_traj.GetPose(t))

    def compute_tau(self, context, tau_command) -> None:
        """Drake torque calculation callback function."""
        # unpacking states
        t = context.get_time()
        body_poses = self.GetInputPort("body_poses_in").Eval(context)
        x_arm_des = self.GetInputPort("x_arm_des_in").Eval(context)
        q_arm_des = x_arm_des[: self.narm]
        dq_arm_des = x_arm_des[self.narm :]
        x_all = self.GetInputPort("model_state_in").Eval(context)

        q_all = x_all[: self.n]
        q_arm = q_all[: self.narm]
        q_hand = q_all[self.narm :]
        dq_all = x_all[self.n :]
        dq_arm = dq_all[: self.narm]
        # dq_hand = dq_all[self.narm :]

        # [NOTE] these lines simulate the system that estimates the object pose.
        # Here, we assume we have access to the ground truth pose.
        obj_body_index = self.model_plant.GetBodyIndices(self.obj_instance)[0]
        X_WO = body_poses[obj_body_index]
        self.model.set_X_WO(X_WO)  # updating model/object's stored value of X_WO

        # computing the hand Jacobian
        # G = self.model.compute_G(q_all)  # unused grasp matrix retrieval
        J_all = self.model.compute_J_tips(q_all, skip_ineqs=True)  # (nc, 3, n)
        J_hand = J_all[..., self.narm :].reshape((-1, self.nhand))  # (nc * 3, nhand)

        # actuation limits
        lb_arm = self.lb_tau[: self.narm]
        ub_arm = self.ub_tau[: self.narm]
        lb_hand = self.lb_tau[self.narm :]
        ub_hand = self.ub_tau[self.narm :]

        # (i) computing arm torques
        kp_arm = 500.0
        kd_arm = 0.01 * np.diag(np.array([1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1]))
        q_arm_des[-1] = self.q0[self.narm - 1]  # maintain wrist position
        tau_arm = -kp_arm * (q_arm - q_arm_des) - kd_arm @ (dq_arm - dq_arm_des)

        # (ii) null space controller to regulate hand configuration
        k_pjoint = 5.0
        _tau_joint = -k_pjoint * (q_hand - self.q0[self.narm :])
        tau_joint = _tau_joint - J_hand.T @ np.linalg.pinv(J_hand.T) @ _tau_joint

        # (iiia) computing error in object pose, converting to an external wrench
        X_WE_des = self.X_WE_traj.GetPose(t)
        X_WO_des = X_WE_des @ self.X_EO_des

        p = X_WO.translation()
        pd = X_WO_des.translation()
        R = X_WO.rotation().matrix()
        Rd = X_WO_des.rotation().matrix()

        if self.X_WO_last is None:
            dp = np.zeros(3)
            dR = np.zeros(3)
        else:
            # gut check: https://gamedev.stackexchange.com/questions/189950/calculate-angular-velocity-from-rotation-matrix-difference
            dp = (p - self.X_WO_last.translation()) / self.model.plant.time_step()
            R_diff = X_WO.rotation() @ self.X_WO_last.rotation().inverse()
            angle_axis_diff = R_diff.ToAngleAxis()
            dR = (
                angle_axis_diff.axis() * angle_axis_diff.angle()
            ) / self.model.plant.time_step()
        self.X_WO_last = RigidTransform(X_WO)

        # errors (position + velocity)
        e_p = p - pd
        e_dp = dp
        e_R = 0.5 * vee(Rd.T @ R - R.T @ Rd)
        e_dR = dR

        # gains
        k_pp = 50.0  # PD gains for position
        k_dp = 5.0
        k_pR = 50.0  # PD gains for rotation
        k_dR = 5.0

        # computing induced "error wrench"
        w_error = np.hstack(
            (k_pp * e_p + k_dp * e_dp, k_pR * e_R + k_dR * e_dR)
        )  # this wrench will attempt to be counteracted in the QP

        # (iiib) contact force optimization
        _a_frame = self.X_WE_traj.GetAcceleration(t)  # (a_rot, a_trans)
        a_frame = np.zeros(6)  # the acceleration of the inertial frame from IK
        a_frame[:3] = _a_frame[3:]
        a_frame[3:] = _a_frame[:3]
        a_grav = np.array([0, 0, -9.81, 0, 0, 0])
        w_ext = self.obj_mass * a_grav + w_error  # adding all wrenches
        f_opt = self.compute_optimal_forces(q_all, dq_all, X_WO, w_ext, tau_joint)

        # adding (i), (ii), and (iii) for final controller
        tau_hand = J_hand.T @ f_opt + tau_joint

        # thresholding torques to comply with actuation limits
        tau_all = np.concatenate((tau_arm, tau_hand))
        tau_all[: self.narm] = np.where(tau_arm > ub_arm, ub_arm, tau_arm)
        tau_all[: self.narm] = np.where(tau_arm < lb_arm, lb_arm, tau_arm)
        tau_all[self.narm :] = np.where(tau_hand > ub_hand, ub_hand, tau_hand)
        tau_all[self.narm :] = np.where(tau_hand < lb_hand, lb_hand, tau_hand)
        tau_command.SetFromVector(tau_all)

    def compute_optimal_forces(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        X_WO: RigidTransform,
        w_ext: np.ndarray,
        tau_joint: np.ndarray,
    ) -> np.ndarray:
        """Computes the optimal contact forces.

        Parameters
        ----------
        q : np.ndarray, shape=(n,)
            The current robot configuration.
        dq : np.ndarray, shape=(n,)
            The current robot generalized velocities.
        X_WO : RigidTransform
            The current pose of the object relative to the world.
        w_ext : np.ndarray, shape=(6,)
            The external wrench.

        Returns
        -------
        f_opt : np.ndarray, shape=(nc * 3,)
            The stack of optimal contact forces in the WORLD FRAME.
        """
        # useful quantities
        G = self.model.compute_G(q, skip_ineqs=True)  # grasp map
        g_OCs = self.model.compute_gOCs(q, skip_ineqs=True)  # contact to obj frame
        J_all = self.model.compute_J_tips(q, skip_ineqs=True)  # (nc, 3, n)
        J_hand = J_all[..., self.narm :].reshape((-1, self.nhand))  # (nc * 3, nhand)

        R_OCs = g_OCs[:, :3, :3]  # (nc, 3, 3)
        R_OC_blkdiag = scipy.linalg.block_diag(*R_OCs)  # (3*nc, 3*nc)

        R_WO = X_WO.rotation().matrix()  # (3, 3)
        R_WO_blkdiag = scipy.linalg.block_diag(*(self.nc * [R_WO]))  # (3*nc, 3*nc)

        # map from contact frame forces to world frame torques
        A = J_hand.T @ R_WO_blkdiag @ R_OC_blkdiag

        # updating constraint 2: torque limits, subtracting off null space control
        self.cons_tau.evaluator().UpdateCoefficients(
            A,
            self.lb_hand - tau_joint,
            self.ub_hand - tau_joint,
        )

        # updating cost: penalize differences between applied and external wrench
        Q = G.T @ G + 1e-6 * np.eye(3 * self.nc)
        w_ext_right_frame = np.zeros(6)
        w_ext_right_frame[:3] = R_WO.T @ w_ext[:3]
        w_ext_right_frame[3:] = R_WO.T @ w_ext[3:]
        b = G.T @ w_ext_right_frame
        self.cost.evaluator().UpdateCoefficients(Q, b)

        # solve the QP and check feasibility
        f_guess = np.zeros(3 * self.nc)
        result = self.solver.Solve(self.prog, f_guess)

        num_infeas = len(result.GetInfeasibleConstraintNames(self.prog))
        if result.is_success() and num_infeas == 0:
            f_sol = result.get_x_val()  # in contact frame
            self.f_sol_cached = f_sol  # cache the last feasible solution
            return R_WO_blkdiag @ R_OC_blkdiag @ f_sol
        else:
            # if infeasible, return the cached forces
            if self.f_sol_cached is None:
                self.f_sol_cached = np.zeros(3 * self.nc)
            return self.f_sol_cached


class PickupSystem:
    """A container class for executing object pickups."""

    def __init__(
        self,
        model: RobotModel,
        t_lift: float = 0.5,
        hold_duration: float = 1.0,
        lift_duration: float = 2.0,
        lift_height: float = 0.2,
        visualize: bool = False,
    ) -> None:
        """Initialize the PickupSystem.

        Parameters
        ----------
        model : RobotModel
            The robot model.
        t_lift : float, default=0.5
            The time at which to lift the arm.
        hold_duration : float, default=1.0
            The desired duration of holding the object post-lift.
        lift_duration : float, default=2.0
            The desired duration of the lifting motion. Used for trajectory planning.
        lift_height : float, default=0.2
            The desired height above the tabletop to reach.
        visualize : bool, default=False
            Whether to visualize the pickup in a new Meshcat window.
        """
        self.obj = model.obj
        self.model = model
        self.t_lift = t_lift
        self.hold_duration = hold_duration
        self.lift_duration = lift_duration
        self.lift_height = lift_height
        self.visualize = visualize

        self.n = model.n
        self.narm = model.narm
        self.nhand = model.nhand
        self.arm_instance = model.arm_instance
        self.hand_instance = model.hand_instance
        self.obj_instance = model.obj_instance

        # connecting the robot model to the pickup controller
        builder = DiagramBuilder()

        # adding subsystems
        system_model = builder.AddSystem(model.diagram)  # model w/o controllers
        self.ctrl = builder.AddSystem(
            PickupController(model, t_lift, lift_duration, lift_height)
        )

        ee_body_index = model.arm_plant.GetBodyIndices(self.arm_instance)[-1]
        self.ee_body = model.arm_plant.get_body(ee_body_index)
        ik_params = DifferentialInverseKinematicsParameters(self.narm, self.narm)
        ik_params.set_joint_position_limits(
            (
                model.arm_plant.GetPositionLowerLimits(),
                model.arm_plant.GetPositionUpperLimits(),
            )
        )
        ik_params.set_joint_velocity_limits(
            (
                model.arm_plant.GetVelocityLowerLimits(),
                model.arm_plant.GetVelocityUpperLimits(),
            )
        )
        ik_params.set_end_effector_translational_velocity_limits(
            -0.1 * np.ones(3), 0.1 * np.ones(3)
        )
        self.diffik = builder.AddSystem(  # diffik for arm control
            DifferentialInverseKinematicsIntegrator(
                model.arm_plant,
                self.ee_body.body_frame(),
                self.model.plant.time_step(),  # [TODO] get this from the model
                ik_params,
            )
        )
        self.interpolator = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                model.arm_plant.num_positions(),
                self.model.plant.time_step(),  # [TODO] get this from the model
                suppress_initial_transient=True,
            )
        )

        # connecting subsystems
        builder.Connect(
            system_model.GetOutputPort("arm_state_out"),
            self.diffik.GetInputPort("robot_state"),
        )
        builder.Connect(
            self.ctrl.GetOutputPort("X_WE_des_out"),
            self.diffik.GetInputPort("X_WE_desired"),
        )
        builder.Connect(
            self.diffik.GetOutputPort("joint_positions"),
            self.interpolator.GetInputPort("position"),
        )
        builder.Connect(
            self.interpolator.GetOutputPort("state"),
            self.ctrl.GetInputPort("x_arm_des_in"),
        )

        builder.Connect(
            self.ctrl.GetOutputPort("tau_command_out"),
            system_model.GetInputPort("tau_command_in"),
        )
        builder.Connect(
            system_model.GetOutputPort("model_state_out"),
            self.ctrl.GetInputPort("model_state_in"),
        )
        builder.Connect(
            system_model.GetOutputPort("body_poses_out"),
            self.ctrl.GetInputPort("body_poses_in"),
        )

        # initializing the visualizer
        if visualize:
            meshcat = StartMeshcat()
            self.visualizer = MeshcatVisualizer.AddToBuilder(
                builder, system_model.GetOutputPort("sim_query_object"), meshcat
            )
        self.diagram = builder.Build()

        # recovering the plant and setting the object pose
        self.plant = self.diagram.GetSystems()[0].GetSubsystemByName("plant")
        self.simulator = Simulator(self.diagram)

        def monitor_fn(root_context):
            """A function that determines whether to stop integration.

            Terminates if the object rotates too much/becomes too far away from hand.
            Thresholds: (1) 7.5cm away from desired hand pose or (2) 30deg too rotated.
            """
            # recoving the object pose and desired pose
            t = root_context.get_time()
            ctrl_context = self.ctrl.GetMyContextFromRoot(root_context)
            body_poses = self.ctrl.GetInputPort("body_poses_in").Eval(ctrl_context)
            obj_body_index = self.plant.GetBodyIndices(self.obj_instance)[0]
            X_WO = body_poses[obj_body_index]
            X_WE_des = self.ctrl.X_WE_traj.GetPose(t)
            X_WO_des = X_WE_des @ self.ctrl.X_EO_des

            z_dist = np.abs(X_WO.translation()[-1] - X_WO_des.translation()[-1])

            R = X_WO.rotation()
            Rd = X_WO_des.rotation()

            R_diff = R @ Rd.inverse()
            angle_axis_diff = R_diff.ToAngleAxis()

            plant_context = self.plant.GetMyContextFromRoot(root_context)
            velocities = self.plant.GetVelocities(plant_context)
            _v_obj = velocities[self.n :]
            v_obj = np.zeros(6)
            v_obj[:3] = _v_obj[3:]  # linear velocities on top
            v_obj[3:] = _v_obj[:3]

            if z_dist >= 0.075 or np.abs(angle_axis_diff.angle()) >= np.pi / 6:
                return EventStatus.ReachedTermination(
                    self.ctrl,
                    f"    Integration terminated at t={ctrl_context.get_time()}",
                )
            return EventStatus.DidNothing()

        self.simulator.set_monitor(monitor_fn)

        self.sim_context = self.simulator.get_mutable_context()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.sim_context)
        self.plant.SetFreeBodyPose(
            self.plant_context, self.plant.GetBodyByName("obj"), self.obj.X_WO
        )

    def set_q0(self, q: np.ndarray) -> None:
        """Set the configuration of the robot.

        Also sets the contact positions and normals by assuming q is the desired grasp
        configuration being passed into the simulator.

        Parameters
        ----------
        q : np.ndarray, shape=(n,)
            The new configuration.
        """
        q_arm = q[: self.narm]
        q_hand = q[self.narm :]

        # updating the original plant
        X_WO = self.obj.X_WO
        obj_quat = X_WO.rotation().ToQuaternion().wxyz()
        obj_pos = X_WO.translation()
        q_obj = np.concatenate((obj_quat, obj_pos))
        self.plant.SetPositions(self.plant_context, self.arm_instance, q_arm)
        self.plant.SetPositions(self.plant_context, self.hand_instance, q_hand)
        self.plant.SetPositions(self.plant_context, self.obj_instance, q_obj)
        self.ctrl.initialize_ctrl(q)

        # updating the initial wrist position
        self.P_WWrist_0 = self.plant.CalcPointsPositions(
            self.plant_context,
            self.ee_body.body_frame(),
            np.zeros(3),
            self.plant.world_frame(),
        ).squeeze()

        # updating diffik
        diffik_context = self.diffik.CreateDefaultContext()
        arm_plant_context = self.model.arm_plant.CreateDefaultContext()
        self.diffik.SetPositions(diffik_context, q[: self.model.narm])
        self.model.arm_plant.SetPositions(arm_plant_context, q[: self.model.narm])

        X_OW = self.obj.X_WO.inverse()
        P_WCo = self.model.compute_p_tips(q)  # (nc, 3)
        n_W = self.model.compute_Ds(q)  # (nc, 3)
        P_OCo = (X_OW @ P_WCo.T).T
        n_O = (X_OW @ n_W.T).T
        self.set_pn_contact(P_OCo, n_O)

    def set_pn_contact(self, P_OCo: np.ndarray, n_O: np.ndarray) -> None:
        """Sets the contact positions and normals in the object frame."""
        self.ctrl.set_pn_contact(P_OCo, n_O)

    def set_X_WO(self, X_WO: RigidTransform) -> None:
        """Set the pose of the object."""
        X_WO = self.obj.X_WO
        obj_quat = X_WO.rotation().ToQuaternion().wxyz()
        obj_pos = X_WO.translation()
        q_obj = np.concatenate((obj_quat, obj_pos))
        self.plant.SetPositions(self.plant_context, self.obj_instance, q_obj)

    def simulate(
        self,
        q0: np.ndarray,
        X_WO_0: RigidTransform | None = None,
    ) -> tuple[np.ndarray, np.ndarray, RigidTransform, np.ndarray, bool]:
        """Simulates the system.

        Parameters
        ----------
        q0 : np.ndarray, shape=(n,)
            The initial configuration of the robot. The desired object contacts are
            assumed from q0. q0 must therefore be a valid grasp configuration or the
            commanded grasp will be nonsense.
        X_WO_0 : RigidTransform
            The initial pose of the object.

        Returns
        -------
        qf : np.ndarray, shape=(n,)
            The final configuration of the robot.
        P_WWrist_f : np.ndarray, shape=(3,)
            The final position of the wrist in the world frame.
        X_WO_f : np.ndarray, RigidTransform
            The final pose of the object.
        v_obj_f : np.ndarray, shape=(6,)
            The final velocity of the object in the world frame. The linear velocities
            are the first three components and the angular velocities are the last 3.
        success : bool
            Whether the pickup was a success. Checks that the velocity of the object
            at the end of simulation is "low" and that its height exceeds the change
            in the z-coordinate of the wrist by at least 5cm. It is up to the end user
            to simulate long enough past the stopping of the wrist to ensure the
            checks pass consistently.
        """
        # reset initial condition and time
        self.set_q0(q0)
        if X_WO_0 is not None:
            self.set_X_WO(X_WO_0)
        self.simulator.get_mutable_context().SetTime(0.0)
        self.simulator.Initialize()

        # run the simulation
        t_sim = self.t_lift + self.lift_duration + self.hold_duration
        if self.visualize:
            self.visualizer.StartRecording()
            self.simulator.AdvanceTo(t_sim)
            self.visualizer.PublishRecording()
        else:
            self.simulator.AdvanceTo(t_sim)

        # retrieve the final conditions
        final_context = self.simulator.get_context()
        final_plant_context = self.plant.GetMyContextFromRoot(final_context)
        final_positions = self.plant.GetPositions(final_plant_context)
        final_velocities = self.plant.GetVelocities(final_plant_context)

        # the final position of the wrist
        # used to check if diffik got stuck, modify grasp success condition
        ee_body_index = self.plant.GetBodyIndices(self.arm_instance)[-1]
        ee_body = self.plant.get_body(ee_body_index)
        P_WWrist_f = self.plant.CalcPointsPositions(
            final_plant_context,
            ee_body.body_frame(),
            np.zeros(3),
            self.plant.world_frame(),
        ).squeeze()

        q_robot_f = final_positions[: self.n]
        q_obj_f = final_positions[self.n :]
        quat = q_obj_f[:4]
        quat /= np.linalg.norm(quat)  # `Quaternion` complains about precision
        X_WO_f = RigidTransform(Quaternion(quat), q_obj_f[4:])
        # p_obj_f = X_WO_f.translation()

        _v_obj_f = final_velocities[self.n :]
        v_obj_f = np.zeros(6)
        v_obj_f[:3] = _v_obj_f[3:]  # linear velocities on top
        v_obj_f[3:] = _v_obj_f[:3]

        # checking pickup success
        # pickup fails if any of the following are satisfied:
        # (i) the final height of the object is >= 5cm from desired
        # (ii) the final angle of the object is >= 45deg from desired
        X_WE_des = self.ctrl.X_WE_traj.GetPose(final_context.get_time())
        X_WO_des = X_WE_des @ self.ctrl.X_EO_des

        R = X_WO_f.rotation()
        Rd = X_WO_des.rotation()

        z_dist = np.abs(X_WO_f.translation()[-1] - X_WO_des.translation()[-1])
        R_diff = R @ Rd.inverse()
        angle_axis_diff = R_diff.ToAngleAxis()

        success = True
        if (z_dist >= 0.075) or (np.abs(angle_axis_diff.angle()) >= np.pi / 6):
            success = False
            print("    FAILURE")  # [DEBUG]
        return q_robot_f, P_WWrist_f, X_WO_f, v_obj_f, success
