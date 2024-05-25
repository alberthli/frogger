import time
import warnings
from concurrent.futures import TimeoutError

import numpy as np
import trimesh
from pydrake.math import RigidTransform, RotationMatrix

from frogger import ROOT
from frogger.baselines import WuBaselineConfig
from frogger.metrics import ferrari_canny_L1, min_weight_metric
from frogger.objects import MeshObject, MeshObjectConfig
from frogger.robots.robot_core import RobotModel
from frogger.robots.robots import (
    AlgrModelConfig,
    BH280ModelConfig,
    FR3AlgrModelConfig,
    FR3AlgrZed2iModelConfig,
)
from frogger.sampling import (
    HeuristicAlgrICSampler,
    HeuristicBH280ICSampler,
    HeuristicFR3AlgrICSampler,
)
from frogger.solvers import Frogger, FroggerConfig
from frogger.utils import timeout

# [Feb. 22, 2024] suppress annoying torch warning about LUSolve from qpth
warnings.filterwarnings("ignore", category=UserWarning)

# all example robot models
model_sampler_pairs = [
    (
        "Allegro_WuBaseline",
        WuBaselineConfig.from_cfg(AlgrModelConfig),
        HeuristicAlgrICSampler,
    ),
    ("Allegro", AlgrModelConfig, HeuristicAlgrICSampler),
    ("BH280", BH280ModelConfig, HeuristicBH280ICSampler),
    ("FR3-Allegro", FR3AlgrModelConfig, HeuristicFR3AlgrICSampler),
    ("FR3-Allegro-Zed2i", FR3AlgrZed2iModelConfig, HeuristicFR3AlgrICSampler),
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
TIMEOUT_PERIOD_SEC = 60.0  # [EDIT THIS] the number of seconds to wait before timing out
NUM_SAMPLES = 1  # [EDIT THIS] number of grasps to sample per object
EVAL = True  # [EDIT THIS] whether to eval the grasps on min-weight/Ferrari-Canny
VIZ = False  # [EDIT THIS] whether to visualize the results every grasp

# looping over the example models
for pair in model_sampler_pairs:
    model_name, ModelConfig, Sampler = pair
    translation = 0.7 if model_name in ["FR3-Allegro", "FR3-Allegro-Zed2i"] else 0.0
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

        # example showing how to use the custom collision callback
        if "Allegro" in model_name:

            def _custom_coll_callback(
                model: RobotModel, name_A: str, name_B: str
            ) -> float:
                """A custom collision callback.

                Given two collision geom names, indicates what the lower bound on separation
                should be between them in meters.

                WARNING: for now, if you overwrite this, you MUST ensure manually that the fingertips
                are allowed some penetration with the object!
                """
                # organizing names
                has_palm = "palm" in name_A or "palm" in name_B
                has_ds = (
                    "ds_collision" in name_A or "ds_collision" in name_B
                )  # non-tip distal geoms
                has_md = "md" in name_A or "md" in name_B  # medial geoms
                has_px = "px" in name_A or "px" in name_B  # proximal geoms
                has_bs = "bs" in name_A or "bs" in name_B  # base geoms
                has_mp = (
                    "mp" in name_A or "mp" in name_B
                )  # metacarpal geoms, thumb only
                has_obj = "obj" in name_A or "obj" in name_B

                # MUST MANUALLY DO THIS!
                has_tip = "FROGGERCOL" in name_A or "FROGGERCOL" in name_B

                # provide custom bounds on different geom pairs
                if has_palm and has_obj:
                    return 0.01  # ensure at least 1cm separation
                elif has_ds and has_obj:
                    return 0.002  # ensure at least 2mm separation
                elif has_md and has_obj:
                    return 0.005  # ensure at least 5mm separation
                elif has_px and has_obj:  # noqa: SIM114
                    return 0.01  # ensure at least 1cm separation
                elif has_bs and has_obj:  # noqa: SIM114
                    return 0.01  # ensure at least 1cm separation
                elif has_mp and has_obj:  # noqa: SIM114
                    return 0.01  # ensure at least 1cm separation
                elif has_tip and has_obj:
                    return (
                        -model.d_pen
                    )  # allow tips to penetrate object - MUST MANUALLY DO THIS!
                else:
                    return model.d_min  # default case: use d_min

            custom_coll_callback = _custom_coll_callback
        else:
            custom_coll_callback = None

        # loading model and sampler
        model = ModelConfig(
            obj=obj,
            ns=4,
            mu=0.7,
            d_min=0.001,
            d_pen=0.005,
            l_bar_cutoff=0.3,
            viz=VIZ,
            custom_coll_callback=custom_coll_callback,
        ).create()
        sampler = Sampler(
            model,
            z_axis_fwd=model_name in ["FR3-Allegro", "FR3-Allegro-Zed2i"],
        )

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
        ).create()
        end = time.time()
        print(f"    setup time: {end - start}")
        tot_setup_time += end - start

        # timing test
        sub_time = 0.0
        for _ in range(NUM_SAMPLES):
            start = time.time()
            try:
                q_star = timeout(TIMEOUT_PERIOD_SEC)(
                    frogger.generate_grasp
                )()  # only time generation
            except TimeoutError:
                print(
                    f"        Grasp generation timed out after {TIMEOUT_PERIOD_SEC} seconds!"
                )
                continue
            end = time.time()
            sub_time += end - start

            # evaluate the grasps if requested
            if EVAL:
                print(f"        min-weight: {min_weight_metric(model, q_star)}")
                print(f"        Ferrari-Canny: {ferrari_canny_L1(model, q_star)}")

            # visualize the grasp if requested
            if VIZ:
                model.viz_config(q_star)

        print(f"    grasp generation time: {end - start}")
        tot_gen_time += sub_time

    # computing total times
    avg_setup_time = tot_setup_time / (len(obj_names) * NUM_SAMPLES)
    avg_synthesis_time = tot_gen_time / (len(obj_names) * NUM_SAMPLES)
    print("  Finished!")
    print(f"  Average setup time: {avg_setup_time}")
    print(f"  Average synthesis time: {avg_synthesis_time}")
    print("---------------------------------------------------------------------------")
