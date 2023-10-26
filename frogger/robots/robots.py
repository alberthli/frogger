from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydrake.geometry import CollisionFilterDeclaration, GeometrySet
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import Body, Frame

from frogger import ROOT
from frogger.objects import ObjectDescription
from frogger.robots.robot_core import RobotModel, RobotModelConfig


@dataclass(kw_only=True)
class FR3AlgrModelConfig(RobotModelConfig):
    """Configuration of the FR3Algr robot model."""
    hand: str = "rh"

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        assert self.hand in ["lh", "rh"]
        self.model_path = f"fr3_algr/fr3_algr_{self.hand}.sdf"
        if self.name is None:
            self.name = f"fr3_algr_{self.hand}"

class FR3AlgrModel(RobotModel):
    """The FR3-Allegro model."""

    def __init__(self, cfg: FR3AlgrModelConfig) -> None:
        """Initialize the FR3-Allegro model."""
        self.cfg = cfg
        self.hand = cfg.hand
        super().__init__(cfg)

        # collision filtering the table with the object
        cfm = self.scene_graph.collision_filter_manager(self.sg_context)
        inspector = self.query_object.inspector()
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

@dataclass(kw_only=True)
class AlgrModelConfig(RobotModelConfig):
    """Configuration of the Algr robot model."""
    hand: str = "rh"

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        assert self.hand in ["lh", "rh"]
        self.model_path = f"allegro/allegro_{self.hand}.sdf"
        if self.name is None:
            self.name = f"algr_{self.hand}"

class AlgrModel(RobotModel):
    """The Allegro model."""

    def __init__(self, cfg: AlgrModelConfig) -> None:
        """Initialize the Allegro model."""
        self.cfg = cfg
        self.hand = cfg.hand
        super().__init__(cfg)
