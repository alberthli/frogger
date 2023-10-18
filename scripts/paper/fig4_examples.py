import pickle
from pathlib import Path

import numpy as np
import trimesh
from pydrake.math import RigidTransform, RotationMatrix

from frogger import ROOT
from frogger.objects import MeshObject
from frogger.robots.robots import FR3AlgrModel

"""
Specific grasps for figure 4. I manually screenshotted angles for the fig.

Left: a "typical" grasp. Will pick something and then hide the object geom.

Right: 8 grasps on 8 different objects.
"""


def _view_grasp(obj_name: str, num: int | list[int]) -> None:
    """View a grasp."""
    mesh = trimesh.load(ROOT + f"/data/{obj_name}/{obj_name}_clean.obj")
    bounds = mesh.bounds
    lb_O = bounds[0, :]
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
        "nc": 4,  # 4-finger grasp
        "ns": 4,  # sides of pyramidal approximation
        "mu": 0.5,  # internal coefficient of friction
        "l_cutoff": 0.3,  # cutoff value for normalized l*.
        "hand": "rh",  # right hand
        "simplified": True,  # simplified collision geometry
        "th_t": np.pi / 3,  # contact point tilt
        "d_min": 0.001,  # minimum safety margin for collision
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

    q_stars = results["q_stars"]
    success = results["pick_success"]
    if isinstance(num, int):
        q_star = q_stars[num]
        model.viz_config(q_star)
    elif isinstance(num, list):
        for i in num:
            if success[i]:
                print(f"  {i}")
                q_star = q_stars[i]
                model.viz_config(q_star)
    else:
        raise ValueError


# ################## #
# FIG LEFT HAND SIDE #
# ################## #

obj_name = "002_master_chef_can"
_view_grasp(obj_name, 9)

# ################### #
# FIG RIGHT HAND SIDE #
# ################### #

obj_num_pairs = [
    ("004_sugar_box", 0),  # side
    ("005_tomato_soup_can", 1),  # overhand
    ("006_mustard_bottle", 1),  # edge grab
    ("008_pudding_box", 13),  # flat
    ("011_banana", 0),  # thin + curved
    ("012_strawberry", 16),  # cramped
    ("048_hammer", 10),  # slanted grab of thin handle
    ("051_large_clamp", 11),  # weird geometry
]

for obj_name, num in obj_num_pairs:
    _view_grasp(obj_name, num)
