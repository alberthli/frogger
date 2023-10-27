import numpy as np
import trimesh
from pydrake.math import RigidTransform, RotationMatrix

from frogger import ROOT
from frogger.objects import MeshObject, MeshObjectConfig
from frogger.robots.robots import (
    AlgrModelConfig,
    BH280ModelConfig,
    FR3AlgrModelConfig,
)
from frogger.sampling import (
    HeuristicAlgrICSampler, HeuristicBHICSampler, HeuristicFR3AlgrICSampler
)
from frogger.solvers import Frogger, FroggerConfig


# obj_names = [
#     "001_chips_can",
#     "002_master_chef_can",
#     "003_cracker_box",
#     "004_sugar_box",
#     "005_tomato_soup_can",
#     "006_mustard_bottle",
#     "007_tuna_fish_can",
#     "008_pudding_box",
#     "009_gelatin_box",
#     "010_potted_meat_can",
#     "011_banana",
#     "012_strawberry",
#     "013_apple",
#     "014_lemon",
#     "015_peach",
#     "016_pear",
#     "017_orange",
#     "018_plum",
#     "021_bleach_cleanser",
#     "036_wood_block",
#     "043_phillips_screwdriver",
#     "044_flat_screwdriver",
#     "048_hammer",
#     "051_large_clamp",
#     "052_extra_large_clamp",
#     "054_softball",
#     "055_baseball",
#     "056_tennis_ball",
#     "057_racquetball",
#     "058_golf_ball",
#     "061_foam_brick",
#     "065-a_cups",
#     "065-b_cups",
#     "065-c_cups",
#     "065-d_cups",
#     "065-e_cups",
#     "065-f_cups",
#     "065-g_cups",
#     "065-h_cups",
#     "065-i_cups",
#     "065-j_cups",
#     "077_rubiks_cube",
#     "sns_cup",
# ]

# loading object
mesh = trimesh.load(ROOT + "/data/001_chips_can/001_chips_can_clean.obj")
bounds = mesh.bounds
lb_O = bounds[0, :]
ub_O = bounds[1, :]
X_WO = RigidTransform(
    RotationMatrix(),
    np.array([0.7, 0.0, -lb_O[-1]]),
)
obj = MeshObjectConfig(X_WO=X_WO, mesh=mesh, name="001_chips_can", clean=False).create()

# loading model
# model = AlgrModelConfig(
#     obj=obj,
#     ns=4,
#     mu=0.7,
#     d_min=0.001,
#     d_pen=0.005,
#     l_bar_cutoff=0.3,
#     hand="rh",
# ).create()
model = BH280ModelConfig(
    obj=obj,
    ns=4,
    mu=0.7,
    d_min=0.001,
    d_pen=0.005,
    l_bar_cutoff=0.3,
).create()
frogger = FroggerConfig(
    model=model,
    sampler=HeuristicBHICSampler(model),
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

# ########## #
# DEBUG CODE #
# ########## #

# model.viz_config(model.plant.GetPositions(model.plant_context)[:model.n])  # [DEBUG]

# # [DEBUG]
# from pydrake.visualization import ModelVisualizer
# visualizer = ModelVisualizer(
#     visualize_frames=True,
#     triad_length=0.05,
#     triad_radius=0.0025,
#     browser_new=True,
# )
# visualizer.parser().package_map().Add("frogger", ROOT)
# visualizer.parser().AddModels(
#     ROOT + f"/models/barrett_hand/bh280.urdf"
# )
# visualizer.Run()
# breakpoint()