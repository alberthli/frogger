from pathlib import Path

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve

from core.grasping import wedge
from core.robots.robot_core import RobotModel

current_dir = Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1] == "manipulation"][0]
ROOT = str(project_dir)


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


def sample_uniformly_from_S2(num_samples: int = 1, seed: int = None) -> np.ndarray:
    """Samples num_samples points in R^3 uniformly randomly from the sphere S^2.

    Parameters
    ----------
    num_samples : int, default=1
        The number of samples to generate.
    seed : int, default=None
        Random seed.

    Returns
    -------
    samples : np.ndarray, shape=(num_samples, 3)
        The samples.
    """
    # stats.stackexchange.com/questions/7977
    if seed is not None:
        np.random.seed(seed)
    z = 2 * np.random.rand(num_samples) - 1  # U[-1, 1]
    th = 2 * np.pi * np.random.rand(num_samples) - np.pi  # U[-pi, pi]
    x = np.sin(th) * np.sqrt(1 - z**2)
    y = np.cos(th) * np.sqrt(1 - z**2)
    samples = np.stack((x, y, z))
    return samples


def sample_configuration(
    model: RobotModel,
    X_WPc_des: RigidTransform | None = None,
    seed: int = None,
    sampler: str = "heuristic",
    ik_type: str = "partial",
) -> np.ndarray | None:
    """Samples a configuration for the robot.

    Will run either a "partial" or "full" IK with either "heuristic" or "full" setting.
    * partial means that collisions and finger-to-surface constraints are not taken
    into consideration, but a heuristic wrist position is chosen.
    * full means that collisions and finger-to-surface constraints are taken into
    consideration, but the wrist pose is free.

    The public release of our code only supports the "heuristic" setting.

    Parameters
    ----------
    model : RobotModel
        The model of the robot and system.
    X_WPc_des : RigidTransform | None, default=None
        Desired pose of the palm center relative to the world.
    seed : int, default=None
        Random seed.
    sampler : str
        Whether to use the "heuristic" or "cvae" sampler/IK.
    ik_type : str
        Whether to use "partial" or "full" IK.

    Returns
    -------
    q_sample : np.ndarray | None, shape=(n,)
        The configuration sample. Returns None if IK was infeasible.
    ik_counter : int
        The number of IK calls made
    """
    assert ik_type in ["partial", "full"]
    if ik_type == "partial":
        if sampler == "heuristic":
            return sample_configuration_heuristic_partial(
                model, X_WPc_des=X_WPc_des, seed=seed
            )
        elif sampler == "cvae":
            # not included in public release
            raise NotImplementedError
    elif ik_type == "full":
        # IK used in Wu 2022 baseline. Not included in public release.
        raise NotImplementedError
    else:
        raise NotImplementedError


def sample_configuration_heuristic_partial(
    model: RobotModel, X_WPc_des: RigidTransform | None = None, seed: int = None
) -> np.ndarray | None:
    """Samples a configuration for the robot (config 1).

    Uses the "partial" IK scheme and "heuristic" sampling strategy.

    If no target pose for the palm center frame is provided, then this function
    assumes that the frame of the palm center of the robot model has its x-axis
    pointing out of the palm, which will be used to orient the palm with respect to
    the object.

    Otherwise, IK will try to compute a feasible sample satisfying the queried pose.

    Parameters
    ----------
    model : RobotModel
        The model of the robot and system.
    X_WPc_des : RigidTransform | None, default=None
        Desired pose of the palm center relative to the world.
    seed : int, default=None
        Random seed.

    Returns
    -------
    q_sample : np.ndarray | None, shape=(n,)
        The configuration sample. Returns None if IK was infeasible.
    ik_counter : int
        The number of IK calls made.
    """
    if seed is not None:
        np.random.seed(seed)
    n = model.n  # total DOFs
    obj = model.obj

    # compute a valid configuration using IK
    plant = model.plant
    X_BPc = model.X_BasePalmcenter  # pose of palm center wrt hand base frame
    ik = InverseKinematics(plant)

    # if no palm center pose is specified, sample one
    if X_WPc_des is None:
        h_obj = obj.ub_W[-1]  # height of object

        # aligning the axis of the hand with the axes of the object with
        # probability proportional to the square of the lengths of the obj axes
        X_WO = obj.settings["X_WO"]

        X_OBB = obj.X_OBB  # pose of oriented bounding box wrt O
        axis_lengths_O = obj.ub_oriented - obj.lb_oriented

        # choosing which axis to align the hand y axis with randomly with weights
        # favoring long axis-aligned bounding box lengths
        while True:
            # prob_1 = axis_lengths_O[0] ** 2 / np.sum(axis_lengths_O**2)
            # prob_2 = axis_lengths_O[1] ** 2 / np.sum(axis_lengths_O**2)
            prob_1 = axis_lengths_O[0] / np.sum(axis_lengths_O)  # [DEBUG]
            prob_2 = axis_lengths_O[1] / np.sum(axis_lengths_O)

            num = np.random.rand()
            if num <= prob_1:
                hand_y_axis_ind = 0
            elif num > prob_1 and num <= prob_1 + prob_2:
                hand_y_axis_ind = 1
            else:
                hand_y_axis_ind = 2

            hand_y_axis_O = np.zeros(3)
            hand_y_axis_O[hand_y_axis_ind] = 1.0
            hand_y_axis_W = (X_WO @ X_OBB).rotation().matrix() @ hand_y_axis_O
            rand_sign = 1 if np.random.random() < 0.5 else -1
            _y_hat = rand_sign * hand_y_axis_W
            _y_hat = _y_hat / np.linalg.norm(_y_hat)
            y_hat = np.random.vonmises(mu=_y_hat, kappa=20)  # add von mises noise
            y_hat = y_hat / np.linalg.norm(y_hat)

            # computing the rotation matrix from _y_hat to y_hat w/rodrigues
            v = np.cross(_y_hat, y_hat)
            v_wedge = wedge(v)
            s = np.linalg.norm(v)
            c = np.dot(_y_hat, y_hat)
            R_axes = np.eye(3) + v_wedge + (v_wedge @ v_wedge) * ((1 - c) / s**2)

            # choosing which axis to align the hand x axis with
            axes_remaining = np.array([0, 1, 2])[np.arange(3) != hand_y_axis_ind]
            # prob = axis_lengths_O[axes_remaining[0]] ** 2 / np.sum(
            #     axis_lengths_O[axes_remaining] ** 2
            # )
            prob = axis_lengths_O[axes_remaining[0]] / np.sum(
                axis_lengths_O[axes_remaining]
            )  # [DEBUG]
            num = np.random.rand()
            if num <= prob:
                hand_x_axis_ind = axes_remaining[0]
            else:
                hand_x_axis_ind = axes_remaining[1]
            hand_x_axis_O = np.zeros(3)
            hand_x_axis_O[hand_x_axis_ind] = 1.0
            hand_x_axis_W = (X_WO @ X_OBB).rotation().matrix() @ hand_x_axis_O
            rand_sign = 1 if np.random.random() < 0.5 else -1
            _x_hat = rand_sign * hand_x_axis_W
            _x_hat = _x_hat / np.linalg.norm(_x_hat)
            x_hat = R_axes @ _x_hat
            x_hat = x_hat - (np.inner(y_hat, x_hat) / np.inner(y_hat, y_hat)) * y_hat

            # if object is short, only approach from above
            if h_obj <= 0.1 and np.arccos(-x_hat[-1]) > np.pi / 6.0:
                continue
            break

        ik.AddAngleBetweenVectorsConstraint(
            plant.world_frame(),
            y_hat,  # perpendicular to closed fingers (up to sign)
            plant.GetBodyByName(f"{model.hand_name}_palm").body_frame(),
            X_BPc.rotation().matrix() @ np.array([0.0, 1.0, 0.0]),  # palm yaxis
            0.0,  # lower angle
            0.0,  # upper angle
        )

        # we assume the object bounds are close to tight, so we place the palm outside
        # the bounds by some amount
        lb_W = obj.lb_W  # bounding box for the object
        ub_W = obj.ub_W
        c_W = obj.X_WO @ obj.center_mass  # center about which we locate the palm

        # deal with very flat objects
        if ub_W[-1] - lb_W[-1] <= 0.03:
            d_offset = 0.07
        else:
            extra = max(0.1 - 0.04 - h_obj, 0)
            d_offset = extra + 0.04

        # compute the position by rough bisection w/sdf bounding box boundary
        alpha = 1.0
        alphah = alpha
        alphal = 0.0
        sdf_val = _sdf_box(c_W - alpha * x_hat, lb_W, ub_W)
        while np.abs(sdf_val - d_offset) > 1e-3:
            if sdf_val > d_offset:
                alphah = alpha
            else:
                alphal = alpha
            alpha = (alphal + alphah) / 2.0
            sdf_val = _sdf_box(c_W - alpha * x_hat, lb_W, ub_W)
        P_WPalmcenter = c_W - alpha * x_hat

        ik.AddAngleBetweenVectorsConstraint(
            plant.world_frame(),
            x_hat,  # normal of the palm
            plant.GetBodyByName(f"{model.hand_name}_palm").body_frame(),
            X_BPc.rotation().matrix() @ np.array([1.0, 0.0, 0.0]),  # palm center xaxis
            0.0,  # lower angle
            1e-2,  # upper angle
        )  # [DEBUG]
        ik.AddPositionConstraint(
            plant.GetBodyByName(f"{model.hand_name}_palm").body_frame(),
            X_BPc.translation(),  # origin of palm center wrt palm
            plant.world_frame(),
            P_WPalmcenter - 1e-4,
            P_WPalmcenter + 1e-4,
        )

    # if the palm center pose is specified (e.g. by a dataset), constrain it
    else:
        raise NotImplementedError  # [NOTE] this path is deprecated now
        p_des = X_WPc_des.translation()
        R_des = X_WPc_des.rotation()
        ik.AddPositionConstraint(
            plant.GetBodyByName(f"{model.hand_name}_palm").body_frame(),
            X_BPc.translation(),  # origin of palm center wrt palm
            plant.world_frame(),
            p_des - 1e-4,
            p_des + 1e-4,
        )
        ik.AddOrientationConstraint(
            plant.GetBodyByName(f"{model.hand_name}_palm").body_frame(),
            X_BPc.rotation(),
            plant.world_frame(),
            R_des,
            1e-2,
        )

    # try to initialize the hand width to something reasonable using bounding boxes
    z_hat = np.cross(x_hat, y_hat)
    R_WPalmcenter = RotationMatrix(np.stack((x_hat, y_hat, z_hat), axis=1))
    X_WPalmcenter = RigidTransform(R_WPalmcenter, P_WPalmcenter)

    P_WFs = model.compute_candidate_fingertip_positions(obj, X_WPalmcenter)  # (nc, 3)
    fingertip_frames, contact_locs = model.compute_fingertip_contacts()[-2:]
    for i in range(P_WFs.shape[0]):
        ik.AddPositionConstraint(
            fingertip_frames[i],
            contact_locs[i],
            plant.world_frame(),
            P_WFs[i, :] - 1e-4,
            P_WFs[i, :] + 1e-4,
        )

    # initial guess for the arm-hand system
    q_init = np.zeros(n)
    q_hand = model.pregrasp_hand_config  # default hand config
    q_init[-len(q_hand) :] = q_hand  # assume arm DOFs before hand DOFs  # [DEBUG]

    # must append object states onto the initial guess
    X_WO = model.obj.X_WO
    obj_quat = X_WO.rotation().ToQuaternion().wxyz()
    obj_pos = X_WO.translation()
    q_obj = np.concatenate((obj_quat, obj_pos))
    q_all = np.concatenate((q_init, q_obj))

    # attempt to solve IK
    result = Solve(ik.prog(), q_all)
    if result.is_success():
        q_sample = plant.GetPositions(ik.context())[: model.n]
        return q_sample, 1
    else:
        return None, 1
