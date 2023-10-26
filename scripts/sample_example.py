import trimesh
from pydrake.math import RigidTransform, RotationMatrix

from frogger import ROOT
from frogger.objects import MeshObject, MeshObjectConfig
from frogger.robots.robots import (
    AlgrModel, AlgrModelConfig, FR3AlgrModel, FR3AlgrModelConfig
)
from frogger.sampling import HeuristicAlgrICSampler, HeuristicFR3AlgrICSampler
from frogger.solvers import Frogger, FroggerConfig


# loading object
mesh = trimesh.load(ROOT + "/data/001_chips_can/001_chips_can_clean.obj")
bounds = mesh.bounds
lb_O = bounds[0, :]
ub_O = bounds[1, :]
X_WO = RigidTransform(
    RotationMatrix(),
    np.array([0.7, 0.0, -lb_O[-1]]),
)
obj_cfg = MeshObjectConfig(X_WO=X_WO, mesh=mesh, name="001_chips_can", clean=False)
obj = MeshObject(obj_cfg)

# loading model
model_cfg = AlgrModelConfig(
    obj=obj,
    ns=4,
    mu=0.7,
    d_min=0.001,
    d_pen=0.005,
    l_bar_cutoff=0.3,
    hand="rh",
)
model = AlgrModel(model_cfg)
model.warm_start()

frogger_cfg = FroggerConfig(
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
)
frogger = Frogger(frogger_cfg)
q_star = frogger.generate_grasp()
model.viz_config(q_star)
