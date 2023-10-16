from pathlib import Path

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import Body, Frame

from frogger.objects import ObjectDescription
from frogger.robots.robot_core import RobotModel

current_dir = Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1] == "manipulation"][0]
ROOT = str(project_dir)


class FR3AlgrModel(RobotModel):
    """The FR3-Allegro model."""

    def __init__(self, obj: ObjectDescription, settings: dict) -> None:
        """Initialize the FR3-Allegro model."""
        # standard parameters
        narm = 7  # DOFs for the arm
        nuarm = 7  # actuated DOFs for the arm
        nc = settings.get("nc", 4)
        assert nc == 3 or nc == 4
        nhand = 16  # DOFs for the hand
        nuhand = 16  # actuated DOFs for the hand
        assert "hand" in settings
        settings["hand_name"] = f"algr_{settings['hand']}"
        settings["arm_name"] = "fr3_arm"
        settings["name"] = f"{settings['arm_name']}_{settings['hand_name']}"
        self.settings = settings
        self.hand_name = settings["hand_name"]
        self.arm_name = settings["arm_name"]
        self.name = settings["name"]

        # store a submodel of just the arm - used for end-effector control
        self.arm_plant = MultibodyPlant(time_step=0.0)
        _parser = Parser(self.arm_plant)
        _parser.package_map().Add("manipulation", ROOT)
        _parser.AddModelFromFile(ROOT + "/models/fr3/fr3_arm_simplified.sdf")
        self.arm_plant.WeldFrames(
            self.arm_plant.world_frame(),
            self.arm_plant.GetFrameByName("fr3_link0"),
        )
        self.arm_plant.Finalize()

        # store a submodel of just the arm and hand - used for gravity compensation
        hand = self.settings["hand"]
        self.arm_hand_plant = MultibodyPlant(time_step=0.001)
        _parser = Parser(self.arm_hand_plant)
        _parser.package_map().Add("manipulation", ROOT)
        _parser.AddModelFromFile(ROOT + "/models/fr3/fr3_arm_simplified.sdf")
        _parser.AddModelFromFile(
            ROOT + f"/models/allegro/allegro_{hand}_simplified.sdf"
        )
        self.arm_hand_plant.WeldFrames(
            self.arm_hand_plant.GetFrameByName("fr3_link7"),
            self.arm_hand_plant.GetFrameByName(f"algr_{hand}_palm"),
            RigidTransform(
                RotationMatrix.MakeZRotation(-0.7853981633974484),
                np.array([0.0, 0.0, 0.12]),
            ),
        )
        self.arm_hand_plant.WeldFrames(
            self.arm_hand_plant.world_frame(),
            self.arm_hand_plant.GetFrameByName("fr3_link0"),
        )
        self.arm_hand_plant.Finalize()

        super().__init__(narm, nhand, nuarm, nuhand, nc, obj, settings)

    def _q_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounds on the FR3-Algr configuration.

        Returns
        -------
        lb_q : np.ndarray, shape=(n,)
            Lower bounds.
        ub_q : np.ndarray, shape=(n,)
            Upper bounds.
        """
        # manufacturer angle limits
        lb_q_fr3 = np.array(
            [-2.3093, -1.5133, -2.4937, -2.7478, -2.4800, 0.8521, -2.6895]
        )
        lb_q_imr = np.array([-0.47, -0.196, -0.174, -0.227])  # if, mf, rf
        lb_q_th = np.array([0.263, -0.105, -0.189, -0.162])  # th
        _q_lower = np.concatenate((lb_q_fr3, lb_q_imr, lb_q_imr, lb_q_imr, lb_q_th))

        ub_q_fr3 = np.array([2.3093, 1.5133, 2.4937, -0.4461, 2.4800, 4.2094, 2.6895])
        ub_q_imr = np.array([0.47, 1.61, 1.709, 1.618])  # if, mf, rf
        ub_q_th = np.array([1.396, 1.163, 1.644, 1.719])  # th
        _q_upper = np.concatenate((ub_q_fr3, ub_q_imr, ub_q_imr, ub_q_imr, ub_q_th))

        _q_diff = _q_upper - _q_lower

        # (1) constrain axial joints + FR3 joints to middle 90% of range
        lb_q = _q_lower + 0.05 * _q_diff
        ub_q = _q_upper - 0.05 * _q_diff

        # (2) enforce on most non-axial joints a lower limit of 0.15
        ind_non_axial = [8, 9, 10, 12, 13, 14, 16, 17, 18, 21, 22]
        lb_q[ind_non_axial] = 0.15

        # (3) thumb has custom lower bounds
        lb_q[19] = 0.4  # thumb cmc joint
        lb_q[20] = 0.5  # thumb axl joint

        return lb_q, ub_q

    def _tau_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Bounds on the motor torques.

        Returns
        -------
        lb_tau : np.ndarray, shape=(nu,)
            Lower bounds.
        ub_tau : np.ndarray, shape=(nu,)
            Upper bounds.
        """
        lb_tau = -0.7 * np.ones(self.nu)
        ub_tau = 0.7 * np.ones(self.nu)
        lb_tau[:4] = -87.0
        ub_tau[:4] = 87.0
        lb_tau[4:7] = -12.0
        ub_tau[4:7] = 12.0
        return lb_tau, ub_tau

    def preload_model(self) -> None:
        """Loads the FR3-Algr description."""
        hand = self.settings["hand"]
        simplified = self.settings.get("simplified", True)
        assert hand in ["lh", "rh"]

        # loading the correct arm and hand model
        if simplified:
            self.parser.AddModelFromFile(ROOT + "/models/fr3/fr3_arm_simplified.sdf")
            self.parser.AddModelFromFile(
                ROOT + f"/models/allegro/allegro_{hand}_simplified.sdf"
            )
        else:
            self.parser.AddModelFromFile(ROOT + "/models/fr3/fr3_arm.sdf")
            self.parser.AddModelFromFile(ROOT + f"/models/allegro/allegro_{hand}.sdf")

        # welding the FR3-Algr system together and to the world frame origin
        # the Z rotation below is taken from the fr3_arm_and_hand sdf
        # the Z offset is arbitrary - there will be a fixture between the arm and hand
        # that isn't designed yet, so something will fill the gap
        self.plant.WeldFrames(
            self.plant.GetFrameByName("fr3_link7"),
            self.plant.GetFrameByName(f"algr_{hand}_palm"),
            RigidTransform(
                RotationMatrix.MakeZRotation(-0.7853981633974484),
                np.array([0.0, 0.0, 0.12]),
            ),
        )
        self.plant.WeldFrames(
            self.plant.world_frame(), self.plant.GetFrameByName("fr3_link0")
        )

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
        hand = self.settings["hand"]
        assert hand in ["lh", "rh"]

        # 4-finger grasping
        if self.nc == 4:
            contact_bodies = [
                self.plant.GetBodyByName(f"algr_{hand}_if_ds"),  # index
                self.plant.GetBodyByName(f"algr_{hand}_mf_ds"),  # middle
                self.plant.GetBodyByName(f"algr_{hand}_rf_ds"),  # ring
                self.plant.GetBodyByName(f"algr_{hand}_th_ds"),  # thumb
            ]
            contact_frames = [
                contact_bodies[0].body_frame(),  # index
                contact_bodies[1].body_frame(),  # middle
                contact_bodies[2].body_frame(),  # ring
                contact_bodies[3].body_frame(),  # thumb
            ]

            # computing the contact location, can move this using settings
            # th_t is an angle from the axis pointing out of the fingertip rotating
            # towards the palm. th_t = 0 corresponds to the desired contact being
            # precisely on the tip and th_t = pi/2 corresponds to the desired contact
            # being where the "pad of the finger" would be on a human.
            th_t = self.settings.get("th_t", 0.0)  # tilt angle
            r_f = 0.012  # radius of fingertip
            contact_locs = [
                np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # if
                np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # mf
                np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # rf
                np.array([r_f * np.sin(th_t), 0.0, 0.0423 + r_f * np.cos(th_t)]),  # th
            ]

        # 3-finger grasping
        elif self.nc == 3:
            contact_bodies = [
                self.plant.GetBodyByName(f"algr_{hand}_if_ds"),  # index
                self.plant.GetBodyByName(f"algr_{hand}_mf_ds"),  # middle
                self.plant.GetBodyByName(f"algr_{hand}_th_ds"),  # thumb
            ]
            contact_frames = [
                contact_bodies[0].body_frame(),  # index
                contact_bodies[1].body_frame(),  # middle
                contact_bodies[2].body_frame(),  # thumb
            ]
            th_t = self.settings.get("th_t", 0.0)  # tilt angle
            r_f = 0.012  # radius of fingertip
            contact_locs = [
                np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # if
                np.array([r_f * np.sin(th_t), 0.0, 0.0267 + r_f * np.cos(th_t)]),  # mf
                np.array([r_f * np.sin(th_t), 0.0, 0.0423 + r_f * np.cos(th_t)]),  # th
            ]
        return contact_bodies, contact_frames, contact_locs

    @property
    def pregrasp_hand_config(self) -> np.ndarray:
        """A heuristically-motivated pregrasp hand configuration.

        Used to help sample initial grasps by pre-shaping the hand.

        On the Allegro hand, we define "open" and "closed" pregrasp configurations.
        "Narrow" means the hand is completely open (so far, we've found this isn't
        very useful), while "closed" means the hand is partially closed initially.

        Returns
        -------
        q_hand : np.ndarray, shape=(16,)
            The configuration of ONLY the hand pre-grasp.
        """
        # regardless of number of fingers, this is the pregrasp
        q_imr = np.array([0.0, 0.5, 0.5, 0.5])  # if, mf, rf
        q_th = np.array([0.8, 1.0, 0.5, 0.5])  # th
        q_hand = np.concatenate((q_imr, q_imr, q_imr, q_th))
        return q_hand

    def compute_candidate_fingertip_positions(
        self,
        obj: ObjectDescription,
        X_WPalmcenter: RigidTransform,
    ) -> np.ndarray:
        """Computes candidate fingertip positions for IK.

        Parameters
        ----------
        obj : ObjectDescription
            An object.
        X_WPalmcenter : RigidTransform
            The pose of the palm center with respect to the world frame.

        Returns
        -------
        P_WFs : np.ndarray, shape=(nc, 3)
            The approximate fingertip positions used to guide IK.
        """
        X_WO = obj.X_WO
        X_OBB = obj.X_OBB
        R_OBB = X_OBB.rotation()
        R_WBB = X_WO.rotation() @ R_OBB
        axis_lengths_O = obj.ub_oriented - obj.lb_oriented

        z_pc = X_WPalmcenter.rotation().matrix()[:, -1]  # z axis of palm center
        z_alignment = np.argmax(
            np.abs(R_WBB.inverse() @ z_pc)
        )  # which axis z_pc is most similar to
        w = axis_lengths_O[z_alignment]
        w = min(w, 0.1)  # consider a max width for feasibility reasons

        # setting the finger positions
        x_pc = X_WPalmcenter.rotation().matrix()[:, 0]  # x axis of palm center
        x_alignment = np.argmax(
            np.abs(R_WBB.inverse() @ x_pc)
        )  # which axis x_pc is most similar to
        if x_alignment == 2:
            f_ext = min(X_WPalmcenter.translation()[-1], 0.08)  # fing extension
        else:
            f_ext = 0.08
        if self.nc == 4:
            if self.settings["hand"] == "rh":
                p_if = X_WPalmcenter @ np.array([f_ext, 0.04 - 0.02, w / 2])
                p_mf = X_WPalmcenter @ np.array([f_ext, 0.0 - 0.02, w / 2])
                p_rf = X_WPalmcenter @ np.array([f_ext, -0.04 - 0.02, w / 2])
                p_th = X_WPalmcenter @ np.array([f_ext, 0.02 - 0.02, -w / 2])
            else:
                p_if = X_WPalmcenter @ np.array([f_ext, -0.04 + 0.02, w / 2])
                p_mf = X_WPalmcenter @ np.array([f_ext, 0.0 + 0.02, w / 2])
                p_rf = X_WPalmcenter @ np.array([f_ext, 0.04 + 0.02, w / 2])
                p_th = X_WPalmcenter @ np.array([f_ext, -0.02 + 0.02, -w / 2])

            P_WFs = np.stack((p_if, p_mf, p_rf, p_th))
        elif self.nc == 3:
            if self.settings["hand"] == "rh":
                p_if = X_WPalmcenter @ np.array([f_ext, 0.04 - 0.02, w / 2])
                p_mf = X_WPalmcenter @ np.array([f_ext, 0.0 - 0.02, w / 2])
                p_th = X_WPalmcenter @ np.array([f_ext, 0.02 - 0.02, -w / 2])
            else:
                p_if = X_WPalmcenter @ np.array([f_ext, -0.04 + 0.02, w / 2])
                p_mf = X_WPalmcenter @ np.array([f_ext, 0.0 + 0.02, w / 2])
                p_th = X_WPalmcenter @ np.array([f_ext, -0.02 + 0.02, -w / 2])

            P_WFs = np.stack((p_if, p_mf, p_th))
        else:
            raise NotImplementedError
        return P_WFs

    @property
    def X_BasePalmcenter(self) -> RigidTransform:
        """The pose of the palm center relative to the hand base frame.

        Used to help sample initial grasps by placing palm relative to the object.

        The Allegro base frame is located at the bottom of the wrist with the x-axis
        pointing outward from the palm and the z-axis pointing towards the fingers.
        The center of the palm here is defined to be 9.5cm up from the base of the
        palm, which is located at the base of the middle finger with the x-axis
        pointing out and then additionally 2cm towards the index finger.

        Returns
        -------
        X_BasePalmcenter : RigidTransform
            The pose.
        """
        if self.settings["hand"] == "rh":
            return RigidTransform(RotationMatrix(), np.array([0.0, 0.02, 0.095]))
        else:
            return RigidTransform(RotationMatrix(), np.array([0.0, -0.02, 0.095]))
