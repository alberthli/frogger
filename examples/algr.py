import numpy as np
import trimesh
from pydrake.math import RigidTransform, RotationMatrix

from frogger import ROOT
from frogger.objects import MeshObject, MeshObjectConfig
from frogger.robots.robots import AlgrModelConfig, BH280ModelConfig, FR3AlgrModelConfig
from frogger.sampling import (
    HeuristicAlgrICSampler,
    HeuristicBH280ICSampler,
    HeuristicFR3AlgrICSampler,
)
from frogger.solvers import Frogger, FroggerConfig

# loading object
obj_name = "001_chips_can"
mesh = trimesh.load(ROOT + f"/data/{obj_name}/{obj_name}_clean.obj")
bounds = mesh.bounds
lb_O = bounds[0, :]
ub_O = bounds[1, :]
X_WO = RigidTransform(
    RotationMatrix(),
    np.array([0.0, 0.0, -lb_O[-1]]),
)
obj = MeshObjectConfig(X_WO=X_WO, mesh=mesh, name=obj_name, clean=False).create()

# loading model
model = AlgrModelConfig(
    obj=obj,
    ns=4,
    mu=0.7,
    d_min=0.001,
    d_pen=0.005,
    l_bar_cutoff=0.3,
    hand="rh",
).create()
frogger = FroggerConfig(
    model=model,
    sampler=HeuristicAlgrICSampler(model),
    tol_surf=1e-3,
    tol_joint=1e-2,
    tol_col=1e-3,
    tol_fclosure=1e-5,
    xtol_rel=1e-6,
    xtol_abs=1e-6,
    maxeval=1000,
    maxtime=60.0,
).create()
print("Model compiled! Generating grasp...")
q_star = frogger.generate_grasp()
print("Grasp generated!")
model.viz_config(q_star)