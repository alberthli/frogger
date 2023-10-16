import pickle
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import nlopt
import numpy as np
import trimesh
from pydrake.math import RigidTransform, RotationMatrix

from frogger.objects import MeshObject
from frogger.paper_experiments import run_exp
from frogger.robots.robots import FR3AlgrModel

"""
This script computes the correlation between the min-weight and ferrari-canny metrics.

We use slightly different settings than the ones in the main experiments, but no major
differences are worth reporting.
"""

current_dir = Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1] == "manipulation"][0]
ROOT = str(project_dir)

# information of small subset of objects to get the idea of the trend
obj_names = [
    "001_chips_can",
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
]
print(f"Number of objects: {len(obj_names)}")

# optimizer settings
opt_settings = {
    "tol_joint": 1e-2,  # tolerance on joint constraints
    "tol_surf": 5e-4,  # tolerance on surface constraints
    "tol_col": 1e-3,  # tolerance on collision constraints
    "tol_baseline": 1e-6,  # tolerance on baseline bilevel constraint
    "alg": nlopt.LD_SLSQP,  # nlopt algorithm
    "maxtime": 10.0,  # maximum allowed runtime in seconds
}

fc_vals = []
l_vals = []

# m = 16
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
    model_settings = {
        "nc": 4,  # 4-finger grasp
        "ns": 4,  # sides of pyramidal approximation
        "mu": 0.5,  # internal coefficient of friction
        "l_cutoff": 1e-6,  # cutoff value for normalized l*
        "hand": "rh",  # right hand
        "simplified": True,  # simplified collision geometry
        "th_t": np.pi / 3,  # contact point tilt
        "d_min": 0.003,  # minimum safety margin for collision
        "d_pen": 0.003,  # allowed penetration of fingertip into object
        "pregrasp_type": "closed",  # hand pregrasp type
        "regularize": False,  # whether to apply curvature regularization
        "baseline": False,  # NOT the baseline optimizer
        "viz": False,  # don't visualize to save time
    }
    model = FR3AlgrModel(obj, model_settings)
    sampler = "heuristic"
    ik_type = "partial"
    run_exp(
        obj,
        model,
        opt_settings,
        sampler=sampler,
        ik_type=ik_type,
        num_feas_samples=100,
        suffix="fccorr16",
        pick=False,
    )
    exp_name = f"exp_{obj_name}_{model.settings['name']}_fccorr16"
    path_str = ROOT + f"/data/{obj_name}/{exp_name}.pkl"
    with open(path_str, "rb") as handle:
        results = pickle.load(handle)

    fc_vals.append(results["fc_stars"])
    l_vals.append(results["l_stars"])

fc_vals = np.stack(fc_vals).reshape(-1)
l_vals = np.stack(l_vals).reshape(-1)
correlation = np.corrcoef(fc_vals, l_vals)[0, 1]
print(f"correlation: {correlation}")

matplotlib.rcParams.update({"font.size": 16})
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.figure(figsize=(8, 4))
plt.scatter((model.ns * model.nc) * l_vals, fc_vals * 1e3)
plt.title(r"$\bar{\ell}^*$ vs. $\epsilon$ Metric, $m=16$")
plt.xlabel(r"$\bar{\ell}^*$")
plt.ylabel(r"$\epsilon$ ($\times$ 1e3)")
plt.xlim([0.0, 0.8])
plt.ylim([0.0, np.max(fc_vals * 1e3) * 1.1])
plt.tight_layout()
plt.rc("axes", labelsize=12)
plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)
plt.savefig("correlation_fig_16.pdf")
