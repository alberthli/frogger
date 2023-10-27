from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.tree import Frame
from pydrake.solvers import Solve

from frogger import ROOT
from frogger.grasping import wedge
from frogger.robots.robot_core import RobotModel
from frogger.robots.robots import AlgrModel, BH280Model, FR3AlgrModel


# ########### #
# IC SAMPLERS #
# ########### #

class ICSampler(ABC):
    """An interface for defining samplers for initial grasps q0.

    Any custom samplers only need to implement the `sample_configuration` function.
    """

    def __init__(self, model: RobotModel) -> None:
        """Initialize the sampler.

        Parameters
        ----------
        model : RobotModel
            The model of the robot and system.
        """
        self.model = model

    @abstractmethod
    def sample_configuration(self) -> np.ndarray:
        """Samples a grasp configuration q0.

        Returns
        -------
        q_star : np.ndarray, shape=(n,)
            The optimized grasp configuration.
        """


class HeuristicICSampler(ICSampler):
    """Abstract heuristic sampler for general manipulators.

    High-level idea: implement a way to sample a palm pose in the world frame, then
    solve a simple inverse kinematics problem to place the palm there.

    If desired, additional constraints can also be added.
    """

    def __init__(self, model: RobotModel) -> None:
        """Initialize the heuristic sampler.

        Parameters
        ----------
        model : RobotModel
            The model of the robot and system.
        """
        super().__init__(model)
        self.palm_frame = None  # palm frame used for sampling.

    def get_palm_frame(self) -> Frame:
        """Gets the palm frame.

        This MUST be specified in the URDF/SDF by adding a dummy link with the substring
        "FROGGERSAMPLE" in the link name. You can only have one such link.

        Recommended axis conventions:
        * The x-axis should point outwards from the palm.
        * If the hand morphology is such that it can act like a parallel-jaw gripper,
          then the y-axis should be perpendicular to the close/open direction.

        For example, for the Allegro right hand, the x-axis points out, the y-axis
        points to the right towards the thumb, and the z-axis points towards the
        remaining 3 fingers.

        Returns
        -------
        f_palm : Frame
            The palm frame.

        Raises
        ------
        ValueError
            If there are 0 or multiple links identified as a sampling palm frame.
        """
        if self.palm_frame is None:
            inspector = self.model.query_object.inspector()
            frames = []
            for fid in inspector.GetAllFrameIds():
                body = self.model.plant.GetBodyFromFrameId(fid)
                frame = body.body_frame()
                name = body.name()
                if "FROGGERSAMPLE" in name:
                    frames.append(frame)
            if len(frames) != 1:
                raise ValueError("There must be exactly 1 sampling frame!")
            self.palm_frame = frames[0]
        return self.palm_frame

    def sample_palm_pose(self) -> RigidTransform:
        """Samples the pose of the palm in the world frame for the IK problem.

        Overview of strategy:
        (1) align the y-axis of the palm wrt the object
        (2) align the x-axis of the palm wrt the object
        (3) place the palm near the object

        Returns
        -------
        X_WPalm : RigidTransform
            The pose of the palm wrt the world.
        """
        obj = self.model.obj
        X_WO = obj.X_WO  # the pose of the object in the world frame
        X_OBB = obj.X_OBB  # the oriented bounding box expressed in the obj frame
        axis_lengths_O = obj.ub_oriented - obj.lb_oriented

        # (1) pick a direction to the align the y-axis
        # the alignment probability is proportional to the length of the BB sides

        # (1a) picking using simple CDF sampling
        prob_1 = axis_lengths_O[0] / np.sum(axis_lengths_O)
        prob_2 = axis_lengths_O[1] / np.sum(axis_lengths_O)
        num = np.random.rand()
        if num <= prob_1:
            hand_y_axis_ind = 0
        elif num > prob_1 and num <= prob_1 + prob_2:
            hand_y_axis_ind = 1
        else:
            hand_y_axis_ind = 2

        # (1b) representing the axis in the world frame
        hand_y_axis_O = np.zeros(3)
        hand_y_axis_O[hand_y_axis_ind] = 1.0
        hand_y_axis_W = (X_WO @ X_OBB).rotation().matrix() @ hand_y_axis_O

        # (1c) randomly choosing the direction of the axis
        rand_sign = 1 if np.random.random() < 0.5 else -1
        _y_hat = rand_sign * hand_y_axis_W
        _y_hat = _y_hat / np.linalg.norm(_y_hat)
        y_hat = np.random.vonmises(mu=_y_hat, kappa=20)  # add von mises noise
        y_hat = y_hat / np.linalg.norm(y_hat)

        # (2) picking a direction to align the x-axis
        # the alignment probability is proportional to the length of the remaining BB
        # sides after accounting for the one picked by the y-axis

        # (2a) CDF sampling
        ax_remaining = np.array([0, 1, 2])[np.arange(3) != hand_y_axis_ind]
        prob = axis_lengths_O[ax_remaining[0]] / np.sum(axis_lengths_O[ax_remaining])
        num = np.random.rand()
        if num <= prob:
            hand_x_axis_ind = ax_remaining[0]
        else:
            hand_x_axis_ind = ax_remaining[1]

        # (2b) representing the axis in the world frame
        hand_x_axis_O = np.zeros(3)
        hand_x_axis_O[hand_x_axis_ind] = 1.0
        hand_x_axis_W = (X_WO @ X_OBB).rotation().matrix() @ hand_x_axis_O

        # (2c) randomly choosing the direction of the axis
        rand_sign = 1 if np.random.random() < 0.5 else -1
        _x_hat = rand_sign * hand_x_axis_W
        _x_hat = _x_hat / np.linalg.norm(_x_hat)
        R_pert = _rodrigues(_y_hat, y_hat)
        x_hat = R_pert @ _x_hat
        x_hat = x_hat - (np.inner(y_hat, x_hat) / np.inner(y_hat, y_hat)) * y_hat

        # computing the rotation matrix associated with the choice of x and y-axis
        z_hat = np.cross(x_hat, y_hat)
        R_WPalm = np.stack((x_hat, y_hat, z_hat), axis=1)

        # (3) compute a palm position by bisecting with the object bounding box

        # (3a) center of object in world frame
        lb_W = obj.lb_W
        ub_W = obj.ub_W
        c_W = obj.X_WO @ obj.center_mass

        # (3b) bisect
        d_offset = 0.05
        P_WPalm = _bisect_on_box(c_W, -x_hat, d_offset, lb_W, ub_W)

        # return palm pose
        X_WPalm = RigidTransform(RotationMatrix(R_WPalm), P_WPalm)
        return X_WPalm

    def add_additional_constraints(
        self, ik: InverseKinematics, X_WPalm_des: RigidTransform
    ) -> None:
        """Adds any additional constraints to the IK problem.

        Useful for more involved heuristics, e.g., if fingers should also be constrained
        somehow with respect to the object geometry. Typically, the solver is very
        sensitive to the initial guess, so there should be more constraints than just
        the wrist position if you want reasonable solve times.

        Parameters
        ----------
        ik : InverseKinematics
            The IK problem.
        X_WPalm_des : RigidTransform
            The desired palm pose in the world frame.
        """

    def sample_configuration(
        self, tol_ang: float = 0.0, tol_pos: float = 0.0, seed: int | None = None
    ) -> tuple[np.ndarray, int]:
        """Sample a grasp.

        Parameters
        ----------
        tol_ang : float, default=0.0
            The positive tolerance on the orientation of the palm in radians.
        tol_pos : float, default=0.0
            The (double-sided) tolerance on the position of the palm in meters.
        seed : int | None, default=None
            Random seed.

        Returns
        -------
        q_sample : np.ndarray, shape=(n,)
            The sample of the grasp.
        num_attempts : int
            The number of attempts to produce a successful sample.
        """
        assert tol_ang >= 0.0
        assert tol_pos >= 0.0
        if seed is not None:
            np.random.seed(seed)
        plant = self.model.plant

        # repeatedly tries to solve an IK problem until a feasible sample is found
        success = False
        num_attempts = 0
        while not success:
            num_attempts += 1

            # sampling a desired palm pose
            f_palm = self.get_palm_frame()
            X_WPalm_des = self.sample_palm_pose()
            p_WPalm_des = X_WPalm_des.translation()
            R_WPalm_des = X_WPalm_des.rotation()

            ik = InverseKinematics(plant, self.model.plant_context)

            # palm pose constraints
            ik.AddOrientationConstraint(
                plant.world_frame(),
                R_WPalm_des,
                f_palm,
                RotationMatrix(),
                tol_ang,
            )  # constraint: align palm frame with desired palm frame
            ik.AddPositionConstraint(
                plant.world_frame(),
                p_WPalm_des,
                f_palm,
                -tol_pos * np.ones(3),
                tol_pos * np.ones(3),
            )  # constraint: place palm frame at desired position

            # restrict object state for IK, since it's a free body
            q_guess = plant.GetPositions(self.model.plant_context)
            q_obj = q_guess[-7:]  # quaternion + position of object
            ik.get_mutable_prog().AddBoundingBoxConstraint(q_obj, q_obj, ik.q()[-7:])

            # any additional constraints - if the action in this function has a chance
            # at failing, then we catch a ValueError to allow continuation
            try:
                self.add_additional_constraints(ik, X_WPalm_des)
            except ValueError:
                continue

            # solve IK
            result = Solve(ik.prog(), q_guess)
            if result.is_success():
                q_sample = plant.GetPositions(ik.context())[:self.model.n]
                return q_sample, num_attempts


class HeuristicFR3AlgrICSampler(HeuristicICSampler):
    """Heuristic sampler for the FR3 Allegro system.

    The heuristic operates by sampling a palm pose in the world frame, then solving an
    inverse kinematics problem to place the palm there. Based on the width of the
    object, the fingers are heuristically placed to determine the initial guess.
    """

    def __init__(self, model: FR3AlgrModel) -> None:
        """Initialize the IC sampler."""
        super().__init__(model)

    def add_additional_constraints(
        self, ik: InverseKinematics, X_WPalm_des: RigidTransform
    ) -> None:
        """Adds additional constraints.

        Parameters
        ----------
        ik : InverseKinematics
            The IK problem.
        X_WPalm_des : RigidTransform
            The desired palm pose in the world frame.
        """
        # set guess for hand
        q_imr = np.array([0.0, 0.5, 0.5, 0.5])  # if, mf, rf
        q_th = np.array([0.8, 1.0, 0.5, 0.5])  # th
        q_hand = np.concatenate((q_imr, q_imr, q_imr, q_th))
        q_curr = self.model.plant.GetPositions(self.model.plant_context)
        q_curr[7:23] = q_hand
        self.model.plant.SetPositions(self.model.plant_context, q_curr)

        # getting object axis lengths
        obj = self.model.obj
        X_WO = obj.X_WO
        X_OBB = obj.X_OBB
        R_OBB = X_OBB.rotation()
        R_WBB = X_WO.rotation() @ R_OBB
        axis_lengths_O = obj.ub_oriented - obj.lb_oriented

        # computing reasonable width of fingers
        z_pc = X_WPalm_des.rotation().matrix()[:, -1]  # z axis of palm center
        z_alignment = np.argmax(
            np.abs(R_WBB.inverse() @ z_pc)
        )  # which axis z_pc is most similar to
        w = axis_lengths_O[z_alignment]
        w = min(w, 0.1)  # consider a max width for feasibility reasons

        # computing desired fingertip positions
        x_pc = X_WPalm_des.rotation().matrix()[:, 0]  # x axis of palm center
        x_alignment = np.argmax(
            np.abs(R_WBB.inverse() @ x_pc)
        )  # which axis x_pc is most similar to

        # finger extension in the x direction (outward from palm)
        if x_alignment == 2:
            f_ext = min(X_WPalm_des.translation()[-1], 0.08)
        else:
            f_ext = 0.08

        # allegro-specific heuristic for the width of the fingertips
        assert self.model.nc == 4
        hand = self.model.hand
        if hand == "rh":
            p_if = X_WPalm_des @ np.array([f_ext, 0.04, w / 2])
            p_mf = X_WPalm_des @ np.array([f_ext, 0.0, w / 2])
            p_rf = X_WPalm_des @ np.array([f_ext, -0.04, w / 2])
            p_th = X_WPalm_des @ np.array([f_ext, 0.02, -w / 2])
        else:
            p_if = X_WPalm_des @ np.array([f_ext, -0.04, w / 2])
            p_mf = X_WPalm_des @ np.array([f_ext, 0.0, w / 2])
            p_rf = X_WPalm_des @ np.array([f_ext, 0.04, w / 2])
            p_th = X_WPalm_des @ np.array([f_ext, -0.02, -w / 2])
        P_WFs = np.stack((p_if, p_mf, p_rf, p_th))

        # defining frames and positions of initial guess for contacts
        contact_bodies = [
            self.model.plant.GetBodyByName(f"algr_{hand}_if_ds"),  # index
            self.model.plant.GetBodyByName(f"algr_{hand}_mf_ds"),  # middle
            self.model.plant.GetBodyByName(f"algr_{hand}_rf_ds"),  # ring
            self.model.plant.GetBodyByName(f"algr_{hand}_th_ds"),  # thumb
        ]
        contact_frames = [
            contact_bodies[0].body_frame(),  # index
            contact_bodies[1].body_frame(),  # middle
            contact_bodies[2].body_frame(),  # ring
            contact_bodies[3].body_frame(),  # thumb
        ]
        th_t = np.pi / 6.0  # tilt angle
        r_f = 0.012  # radius of fingertip
        contact_locs = [
            np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # if
            np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # mf
            np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # rf
            np.array([r_f * np.sin(th_t), 0.0, 0.0423 + r_f * np.cos(th_t)]),  # th
        ]

        for i in range(P_WFs.shape[0]):
            ik.AddPositionConstraint(
                contact_frames[i],
                contact_locs[i],
                self.model.plant.world_frame(),
                P_WFs[i, :] - 1e-4,
                P_WFs[i, :] + 1e-4,
            ) 

class HeuristicAlgrICSampler(HeuristicFR3AlgrICSampler):
    """Convenience class for sampling with the disembodied Allegro hand.

    The implementation turns out to be identical to the FR3-Allegro sampler.
    """

    def __init__(self, model: AlgrModel) -> None:
        """Initialize the IC sampler."""
        super().__init__(model)

class HeuristicBHICSampler(HeuristicFR3AlgrICSampler):
    """Heuristic sampler for the disembodied Barrett Hand.

    See description in HeuristicFR3AlgrICSampler.
    """

    def __init__(self, model: BH280Model) -> None:
        """Initialize the IC sampler."""
        super().__init__(model)

    def add_additional_constraints(
        self, ik: InverseKinematics, X_WPalm_des: RigidTransform
    ) -> None:
        """Adds additional constraints.

        Parameters
        ----------
        ik : InverseKinematics
            The IK problem.
        X_WPalm_des : RigidTransform
            The desired palm pose in the world frame.
        """
        # set guess for hand
        q_32 = 0.5
        q_33 = 0.3442622950819672 * q_32
        q_11 = 0.0
        q_12 = 0.5
        q_13 = 0.3442622950819672 * q_12
        q_21 = 0.0
        q_22 = 0.5
        q_23 = 0.3442622950819672 * q_22
        q_hand = np.array([q_32, q_33, q_11, q_12, q_13, q_21, q_22, q_23])
        q_curr = self.model.plant.GetPositions(self.model.plant_context)
        q_curr[7:15] = q_hand
        self.model.plant.SetPositions(self.model.plant_context, q_curr)

        # getting object axis lengths
        obj = self.model.obj
        X_WO = obj.X_WO
        X_OBB = obj.X_OBB
        R_OBB = X_OBB.rotation()
        R_WBB = X_WO.rotation() @ R_OBB
        axis_lengths_O = obj.ub_oriented - obj.lb_oriented

        # computing reasonable width of fingers
        z_pc = X_WPalm_des.rotation().matrix()[:, -1]  # z axis of palm center
        z_alignment = np.argmax(
            np.abs(R_WBB.inverse() @ z_pc)
        )  # which axis z_pc is most similar to
        w = axis_lengths_O[z_alignment]
        w = min(w, 0.1)  # consider a max width for feasibility reasons

        # computing desired fingertip positions
        x_pc = X_WPalm_des.rotation().matrix()[:, 0]  # x axis of palm center
        x_alignment = np.argmax(
            np.abs(R_WBB.inverse() @ x_pc)
        )  # which axis x_pc is most similar to

        # finger extension in the x direction (outward from palm)
        if x_alignment == 2:
            f_ext = min(X_WPalm_des.translation()[-1], 0.08)
        else:
            f_ext = 0.08

        # [TODO] barrett-specific heuristic for the width of the fingertips
        assert self.model.nc == 3
        p1 = X_WPalm_des @ np.array([f_ext, 0.025, w / 2])
        p2 = X_WPalm_des @ np.array([f_ext, -0.025, w / 2])
        p3 = X_WPalm_des @ np.array([f_ext, 0.0, -w / 2])
        P_WFs = np.stack((p1, p2, p3))

        # defining frames and positions of initial guess for contacts
        contact_bodies = [
            self.model.plant.GetBodyByName(f"bh_finger_13"),
            self.model.plant.GetBodyByName(f"bh_finger_23"),
            self.model.plant.GetBodyByName(f"bh_finger_33"),
        ]
        contact_frames = [
            contact_bodies[0].body_frame(),
            contact_bodies[1].body_frame(),
            contact_bodies[2].body_frame(),
        ]
        contact_locs = [
            np.array([-0.0375, 0.04, 0.0]),
            np.array([-0.0375, 0.04, 0.0]),
            np.array([-0.0375, 0.04, 0.0]),
        ]

        for i in range(P_WFs.shape[0]):
            ik.AddPositionCost(
                contact_frames[i],
                contact_locs[i],
                self.model.plant.world_frame(),
                P_WFs[i, :],
                np.eye(3),
            )

        # adding coupling constraints
        ik.get_mutable_prog().AddLinearEqualityConstraint(
            self.model.A_couple,
            -self.model.b_couple,
            ik.q()[:self.model.n],
        )

# ##### #
# UTILS #
# ##### #

def _rodrigues(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Computes the rotation matrix from v1 to v2 using Rodrigues' formula."""
    v = np.cross(v1, v2)
    v_wedge = wedge(v)
    s = np.linalg.norm(v)
    c = np.dot(v1, v2)
    R = np.eye(3) + v_wedge + (v_wedge @ v_wedge) * ((1 - c) / s ** 2)
    return R

def _sdf_box(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> float:
    """The signed distance function associated with a box.

    Used for the bisection procedure in the heuristic sampler.

    Parameters
    ----------
    x : np.ndarray, shape=(3,)
        The query point.
    lb : np.ndarray, shape=(3,)
        The lower bounds of the box dimensions.
    ub : np.ndarray, shape=(3,)
        The upper bounds of the box dimensions.
    """
    c = (ub + lb) / 2.0  # the center of the box
    l = (ub - lb) / 2.0  # the half side lengths of the box
    y = np.abs(x - c) - l  # center x, place in the first octant, measure wrt l
    return np.linalg.norm(np.maximum(y, 0.0)) + min(np.max(y), 0.0)

def _bisect_on_box(
    o: np.ndarray, d: np.ndarray, l: float, lb: np.ndarray, ub: np.ndarray
) -> np.ndarray:
    """Uses bisection to compute the intersection between a ray and a box level set.

    Parameters
    ----------
    o : np.ndarray, shape=(3,)
        The ray origin.
    d : np.ndarray, shape=(3,)
        The direction of the ray.
    l : float
        The level of the box.
    lb : np.ndarray, shape=(3,)
        The lower bound of the box.
    ub : np.ndarray, shape=(3,)
        The upper bound of the box.

    Returns
    -------
    p : np.ndarray, shape=(3,)
        The point on the ray.
    """
    # determining the bounds for the bisection
    sdf_val_o = _sdf_box(o, lb, ub)
    inside = sdf_val_o <= 0.0
    gamma = 2.0
    p1 = o + 0.01 * gamma * d
    sdf_val_p1 = _sdf_box(p1, lb, ub)

    # if d points inwards, invert it
    if sdf_val_p1 < sdf_val_o:
        d *= -1

    # find loose upper and lower bounds with crappy line search
    alphah = 0.01 * gamma
    p1 = o + alphah * d
    sdf_val_p1 = _sdf_box(p1, lb, ub)
    while sdf_val_p1 < l:
        alphah *= gamma
        p1 = o + alphah * d
        sdf_val_p1 = _sdf_box(p1, lb, ub)

    alphal = -0.01 * gamma
    p2 = o + alphal * d
    sdf_val_p2 = _sdf_box(p2, lb, ub)
    while sdf_val_p2 > l:
        alphal *= gamma
        p2 = o + alphal * d
        sdf_val_p2 = _sdf_box(p2, lb, ub)

    # bisect in the interval
    alpha = alphah
    sdf_val = _sdf_box(o + alpha * d, lb, ub)
    while np.abs(sdf_val - l) > 1e-3:
        if sdf_val > l:
            alphah = alpha
        else:
            alphal = alpha
        alpha = (alphal + alphah) / 2.0
        sdf_val = _sdf_box(o + alpha * d, lb, ub)
    p = o + alpha * d
    return p
