from pathlib import Path

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import Body, Frame

from frogger import ROOT
from frogger.objects import ObjectDescription
from frogger.robots.robot_core import RobotModel


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
        _parser.package_map().Add("frogger", ROOT)
        _parser.AddModels(ROOT + "/models/fr3/fr3_arm_simplified.sdf")
        self.arm_plant.WeldFrames(
            self.arm_plant.world_frame(),
            self.arm_plant.GetFrameByName("fr3_link0"),
        )
        self.arm_plant.Finalize()

        # store a submodel of just the arm and hand - used for gravity compensation
        hand = self.settings["hand"]
        self.arm_hand_plant = MultibodyPlant(time_step=0.001)
        _parser = Parser(self.arm_hand_plant)
        _parser.package_map().Add("frogger", ROOT)
        _parser.AddModels(ROOT + "/models/fr3/fr3_arm_simplified.sdf")
        _parser.AddModels(
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

    def preload_model(self) -> None:
        """Loads the FR3-Algr description."""
        hand = self.settings["hand"]
        simplified = self.settings.get("simplified", True)
        assert hand in ["lh", "rh"]

        # loading the correct arm and hand model
        if simplified:
            self.parser.AddModels(ROOT + "/models/fr3/fr3_arm_simplified.sdf")
            self.parser.AddModels(
                ROOT + f"/models/allegro/allegro_{hand}_simplified.sdf"
            )
        else:
            self.parser.AddModels(ROOT + "/models/fr3/fr3_arm.sdf")
            self.parser.AddModels(ROOT + f"/models/allegro/allegro_{hand}.sdf")

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
