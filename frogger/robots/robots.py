from dataclasses import dataclass

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
    """Configuration of the FR3Algr robot model.

    Attributes
    ----------
    hand : str, default="rh"
        The hand to use. Can be "rh" or "lh".
    """

    hand: str = "rh"

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        assert self.hand in ["lh", "rh"]
        self.model_path = f"fr3_algr/fr3_algr_{self.hand}.sdf"
        if self.name is None:
            self.name = f"fr3_algr_{self.hand}"

    def create(self) -> "FR3AlgrModel":
        """Creates the model."""
        model = FR3AlgrModel(self)
        return model


class FR3AlgrModel(RobotModel):
    """The FR3-Allegro model.

    [NOTE] This model comes with a fixed table.
    """

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
    """Configuration of the Algr robot model.

    Attributes
    ----------
    hand : str, default="rh"
        The hand to use. Can be "rh" or "lh".
    """

    hand: str = "rh"

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        assert self.hand in ["lh", "rh"]
        self.model_path = f"allegro/allegro_{self.hand}.sdf"
        if self.name is None:
            self.name = f"algr_{self.hand}"

    def create(self) -> "AlgrModel":
        """Creates the model."""
        model = AlgrModel(self)
        model.warm_start()
        return model


class AlgrModel(RobotModel):
    """The Allegro model."""

    def __init__(self, cfg: AlgrModelConfig) -> None:
        """Initialize the Allegro model."""
        self.cfg = cfg
        self.hand = cfg.hand
        super().__init__(cfg)


@dataclass(kw_only=True)
class BH280ModelConfig(RobotModelConfig):
    """Configuration of the Barrett Hand robot model.

    Attributes
    ----------
    n_couple : int, default=4
        The number of coupling constraints to add manually. Note: must actually be 4,
        this value is only specified here to overwrite the value in the parent class.
    """

    n_couple: int = 4

    def __post_init__(self) -> None:
        """Post-initialization checks."""
        self.model_path = "barrett_hand/bh280.urdf"
        assert self.n_couple == 4
        if self.name is None:
            self.name = "bh280"

    def create(self) -> "BH280Model":
        """Creates the model."""
        model = BH280Model(self)
        model.warm_start()
        return model


class BH280Model(RobotModel):
    """The Barrett Hand model."""

    def __init__(self, cfg: BH280ModelConfig) -> None:
        """Initialize the Barrett Hand model."""
        self.cfg = cfg
        super().__init__(cfg)

    def _add_coupling_constraints(self) -> None:
        """Adds coupler constraints manually for the mimic joints.

        In Drake 1.22.0, these are not automatically enforced as kinematic constraints.
        """
        names = self.plant.GetPositionNames()

        def get_index(name):
            return [i for (i, _name) in enumerate(names) if name in _name][0]

        # linear equality constraint: A_couple @ q + b_couple == 0
        A_couple = []
        self.b_couple = np.zeros(4)

        # coupling joint 32 and 33
        # q_33 = 0.3442622950819672 * q_32
        i_32 = get_index("bh_j32_joint")
        i_33 = get_index("bh_j33_joint")
        row = np.zeros(self.n)
        row[i_32] = 0.3442622950819672
        row[i_33] = -1.0
        A_couple.append(row)

        # coupling joint 12 and 13
        # q_13 = 0.3442622950819672 * q_12
        i_12 = get_index("bh_j12_joint")
        i_13 = get_index("bh_j13_joint")
        row = np.zeros(self.n)
        row[i_12] = 0.3442622950819672
        row[i_13] = -1.0
        A_couple.append(row)

        # coupling joint 22 and 23
        # q_23 = 0.3442622950819672 * q_22
        i_22 = get_index("bh_j22_joint")
        i_23 = get_index("bh_j23_joint")
        row = np.zeros(self.n)
        row[i_22] = 0.3442622950819672
        row[i_23] = -1.0
        A_couple.append(row)

        # coupling joint 11 and 21
        # q_11 = q_21
        i_11 = get_index("bh_j11_joint")
        i_21 = get_index("bh_j21_joint")
        row = np.zeros(self.n)
        row[i_11] = 1.0
        row[i_21] = -1.0
        A_couple.append(row)

        self.A_couple = np.array(A_couple)
