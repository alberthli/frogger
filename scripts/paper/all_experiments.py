import warnings
from pathlib import Path

import nlopt
import numpy as np
import trimesh
from pydrake.math import RigidTransform, RotationMatrix

from frogger.objects import MeshObject
from frogger.paper_experiments import compute_results, run_exp, summarize_all_results
from frogger.robots.robots import FR3AlgrModel

"""
This is the master script that runs all experiments for the paper.
"""

# ignores qpth DeprecationWarning from LUSolve - shouldn't be a problem if the Docker
# container is used correctly
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1] == "manipulation"][0]
ROOT = str(project_dir)

# information of objects in the experimental suite
obj_names = [
    "001_chips_can",
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "012_strawberry",
    "013_apple",
    "014_lemon",
    "015_peach",
    "016_pear",
    "017_orange",
    "018_plum",
    "021_bleach_cleanser",
    "036_wood_block",
    "043_phillips_screwdriver",
    "044_flat_screwdriver",
    "048_hammer",
    "051_large_clamp",
    "052_extra_large_clamp",
    "054_softball",
    "055_baseball",
    "056_tennis_ball",
    "057_racquetball",
    "058_golf_ball",
    "061_foam_brick",
    "065-a_cups",
    "065-b_cups",
    "065-c_cups",
    "065-d_cups",
    "065-e_cups",
    "065-f_cups",
    "065-g_cups",
    "065-h_cups",
    "065-i_cups",
    "065-j_cups",
    "077_rubiks_cube",
    "sns_cup",
]

# optimizer settings
opt_settings = {
    "tol_joint": 1e-2,  # tolerance on joint constraints
    "tol_surf": 5e-4,  # tolerance on surface constraints
    "tol_col": 1e-3,  # tolerance on collision constraints
    "alg": nlopt.LD_SLSQP,  # nlopt algorithm
    "maxtime": 10.0,  # maximum allowed runtime in seconds for EACH opt call
}

# performing experiments
for obj_name in obj_names:
    print(f"object: {obj_name}")
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

    # run each configuration for each object
    for i in range(2):
        obj.set_X_WO(X_WO)  # reset the object between experimental configs
        print(f"  configuration {i + 1}")
        if i == 0:
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
                "viz": False,  # don't visualize to save time
            }
            model = FR3AlgrModel(obj, model_settings)

        elif i == 1:
            model_settings = {
                "nc": 4,  # 3-finger grasp
                "ns": 4,  # sides of pyramidal approximation
                "mu": 0.5,  # internal coefficient of friction
                "hand": "rh",  # right hand
                "simplified": True,  # simplified collision geometry
                "th_t": np.pi / 3,  # contact point tilt
                "d_min": 0.001,  # minimum safety margin for collision
                "d_pen": 0.003,  # allowed penetration of fingertip into object
                "pregrasp_type": "closed",  # hand pregrasp type
                "baseline": True,  # the baseline optimizer
                "viz": False,  # don't visualize to save time
            }
            model = FR3AlgrModel(obj, model_settings)
        else:
            raise NotImplementedError

        # recomputes the experimental results
        # if you just want to view the results, comment this out
        run_exp(
            obj,
            model,
            opt_settings,
            sampler="heuristic",
            ik_type="partial",
            num_feas_samples=20,
            suffix=f"{i + 1}",
        )

        # computes results for each specific object and exp configuration
        compute_results(model, suffix=f"{i + 1}")

# computes aggregated results reported in the paper
summarize_all_results()
