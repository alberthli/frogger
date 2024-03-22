import numpy as np
import torch
import trimesh
from nerf_grasping.config.grasp_metric_config import GraspMetricConfig
from nerf_grasping.optimizer_utils import GraspMetric
from pydrake.all import RigidTransform, RotationMatrix

from frogger import ROOT
from frogger.baselines.ng import NerfGraspingBaselineConfig
from frogger.objects import MeshObjectConfig
from frogger.robots.robots import FR3AlgrZed2iModelConfig
from frogger.sampling import HeuristicFR3AlgrICSampler
from frogger.solvers import FroggerConfig

# loading the object
mesh = trimesh.load(ROOT + "/data/test/bottle3/coacd/decomposed.obj")
mesh.apply_scale(0.0915)  # scale down the size of the mesh
bounds = mesh.bounds
lb_O = bounds[0, :]
ub_O = bounds[1, :]
X_WO = RigidTransform(
    RotationMatrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    ),  # make the bottle right side up
    np.array([0.7, 0.0, -lb_O[-2]]),
)
obj = MeshObjectConfig(
    X_WO=X_WO,
    mesh=mesh,
    name="bottle3",
    clean=False,
).create()

# loading the model
ModelConfig = NerfGraspingBaselineConfig.from_cfg(FR3AlgrZed2iModelConfig)
model = ModelConfig(
    obj=obj,
    ns=4,
    mu=0.7,
    d_min=0.001,
    d_pen=0.005,
    min_success_prob=0.05,
    finger_level_set=0.01,
    viz=True,
).create()

# loading the sampler
sampler = HeuristicFR3AlgrICSampler(
    model,
    z_axis_fwd=True,
)
frogger = FroggerConfig(
    model=model,
    sampler=sampler,
    tol_surf=1e-3,
    tol_joint=1e-2,
    tol_col=1e-3,
    tol_fclosure=1e-5,
    xtol_rel=1e-6,
    xtol_abs=1e-6,
    maxeval=1000,
    maxtime=60.0,
).create()

q_star = frogger.generate_grasp()
model.viz_config(q_star)
