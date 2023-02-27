import pickle
from pathlib import Path

import numpy as np
import trimesh_fork as trimesh
from pydrake.math import RigidTransform, RotationMatrix

from core.objects import MeshObject
from core.robots.robots import FR3AlgrModel

"""
Specific grasp for figure 1. I manually screenshotted an angle for the fig.
"""

current_dir = Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1] == "manipulation"][0]
ROOT = str(project_dir)

obj_name = "002_master_chef_can"

# loading objects
mesh = trimesh.load(ROOT + f"/data/{obj_name}/{obj_name}_clean.obj")
bounds = mesh.bounds
lb_O = bounds[0, :]
ub_O = bounds[1, :]
X_WO = RigidTransform(
    RotationMatrix(),
    np.array([0.7, 0.0, -lb_O[-1]]),
)
obj_settings = {
    "X_WO": X_WO,
    "mesh": mesh,
    "name": obj_name,
}
obj = MeshObject(obj_settings, reclean=False)
model_settings = {
    "nc": 3,  # 3-finger grasp
    "ns": 4,  # sides of pyramidal approximation
    "mu": 0.5,  # internal coefficient of friction
    "l_cutoff": 1e-6,  # cutoff value for normalized l*.
    "hand": "rh",  # right hand
    "simplified": True,  # simplified collision geometry
    "th_t": np.pi / 3,  # contact point tilt
    "d_min": 0.003,  # minimum safety margin for collision
    "d_pen": 0.003,  # allowed penetration of fingertip into object
    "pregrasp_type": "closed",  # hand pregrasp type
    "baseline": False,  # NOT the baseline optimizer
    "viz": True,
}
model = FR3AlgrModel(obj, model_settings)

exp_name = f"exp_{obj_name}_{model.settings['name']}_1"
path_str = ROOT + f"/data/{obj_name}/{exp_name}.pkl"
with open(path_str, "rb") as handle:
    results = pickle.load(handle)

q0s = results["q0s"]
q0_seeds = results["q0_seeds"]
q_stars = results["q_stars"]
q_star_seeds = results["q_star_seeds"]

for i, (q_star, seed_star) in enumerate(zip(q_stars, q_star_seeds)):
    for q0, seed0 in zip(q0s, q0_seeds):
        if seed0 == 780 and seed_star == 780:  # seed used for the figure
            # if seed0 == seed_star:  # to browse seeds
            print(seed0)
            print(results["total_times"][i])
            print(results["pick_success"][i])
            model.viz_config(q0)
            model.viz_config(q_star)
            continue
