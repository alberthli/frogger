import time

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

# all example robot models
model_sampler_pairs = [
    ("Allegro", AlgrModelConfig, HeuristicAlgrICSampler),
    ("BH280", BH280ModelConfig, HeuristicBH280ICSampler),
    ("FR3-Allegro", FR3AlgrModelConfig, HeuristicFR3AlgrICSampler),
]

# all objects from paper
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

tot_setup_time = 0.0
tot_gen_time = 0.0
NUM_SAMPLES = 1  # [EDIT THIS] number of grasps to sample per object

# looping over the example models
for pair in model_sampler_pairs:
    model_name, ModelConfig, Sampler = pair
    translation = 0.7 if model_name == "FR3-Allegro" else 0.0
    print(f"model: {model_name}")

    # looping over all objects
    for obj_name in obj_names:
        if model_name == "FR3-Allegro" and obj_name in [
            "048_hammer",
            "051_large_clamp",
            "052_extra_large_clamp",
        ]:
            # our heuristic is very bad for flat objects when they lie on a table
            continue

        print(f"  {obj_name}")
        start = time.time()

        # loading object
        mesh = trimesh.load(ROOT + f"/data/{obj_name}/{obj_name}_clean.obj")
        bounds = mesh.bounds
        lb_O = bounds[0, :]
        ub_O = bounds[1, :]
        X_WO = RigidTransform(
            RotationMatrix(),
            np.array([translation, 0.0, -lb_O[-1]]),
        )
        obj = MeshObjectConfig(
            X_WO=X_WO,
            mesh=mesh,
            name=obj_name,
            clean=False,
        ).create()

        # loading model and sampler
        model = ModelConfig(
            obj=obj,
            ns=4,
            mu=0.7,
            d_min=0.001,
            d_pen=0.005,
            l_bar_cutoff=0.3,
            viz=False,
        ).create()
        sampler = Sampler(model)

        # loading grasp generator
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
        end = time.time()
        print(f"    setup time: {end - start}")
        tot_setup_time += end - start

        # timing test
        start = time.time()
        for _ in range(NUM_SAMPLES):
            _ = frogger.generate_grasp()
        end = time.time()
        print(f"    grasp generation time: {end - start}")
        tot_gen_time += end - start

    # computing total times
    avg_setup_time = tot_setup_time / (len(obj_names) * NUM_SAMPLES)
    avg_synthesis_time = tot_gen_time / (len(obj_names) * NUM_SAMPLES)
    print("  Finished!")
    print(f"  Average setup time: {avg_setup_time}")
    print(f"  Average synthesis time: {avg_synthesis_time}")
    print("---------------------------------------------------------------------------")