import time
from pathlib import Path
from typing import Callable

import dill as pickle
import nlopt
import numpy as np
from pydrake.math import RigidTransform

from frogger import ROOT
from frogger.grasping import ferrari_canny_L1
from frogger.objects import ObjectDescription
from frogger.pickup import PickupSystem
from frogger.robots.robot_core import RobotModel
from frogger.sampling import HeuristicFR3AlgrICSampler


"""
This file contains functions that run the IROS experiments.
"""


def _make_fgh(model: RobotModel) -> tuple[Callable, Callable, Callable]:
    """Returns f, g, and h suitable for use in NLOPT.

    We do not use the Drake NLOPT wrapper because of the overhead required to specify
    gradients that are not autodifferentiated by Drake, which we must cast into
    AutoDiffXd objects. We measured the conversion time and concluded it was
    significant enough to just use the Python implementation of NLOPT directly.

    Parameters
    ----------
    model : RobotModel
        A model of the robot that performs cached computation.

    Returns
    -------
    f : Callable[[np.ndarray, np.ndarray], float]
        Cost function.
    g : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        Vector inequality constraints.
    h : Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
        Vector equality constraints.
    """

    def f(q, grad):
        if grad.size > 0:
            grad[:] = model.compute_Df(q)
        return model.compute_f(q)

    def g(result, q, grad):
        if grad.size > 0:
            grad[:] = model.compute_Dg(q)
        result[:] = model.compute_g(q)

    def h(result, q, grad):
        if grad.size > 0:
            grad[:] = model.compute_Dh(q)
        result[:] = model.compute_h(q)

    return f, g, h


def run_exp(
    obj: ObjectDescription,
    model: RobotModel,
    opt_settings: dict,
    num_feas_samples: int = 20,
    suffix: str = "",
    pick: bool = True,
    timeout: float = 60.0,
) -> None:
    """Runs an experimental configuration.

    It is assumed that the object and its pose are fixed over these trials.

    Parameters
    ----------
    obj : ObjectDescription
        The object.
    model : RobotModel
        A model of the robot that performs cached computation.
    opt_settings : dict
        A dictionary of optimizer settings. See make_optimizer().
    num_feas_samples : int, default=20
        The number of feasible samples to compute for this object.
    suffix : str, default=""
        Additional suffix to add onto an experiment file name.
    pick : bool, default=True
        Whether to simulate pickup.
    timeout : float, default=60.0
        Amount of time before a sample is labeled a failure in seconds.
    """
    # making constraint functions
    f, g, h = _make_fgh(model)

    # relevant info from the model
    n = model.n
    nc = model.nc  # number of contacts
    mu = model.mu
    ns = model.ns

    # number of each constraint type
    n_joint = 2 * n
    n_col = len(model.query_object.inspector().GetCollisionCandidates())
    n_surf = nc

    # tolerances for each constraint type
    n_ineq = n_joint + n_col + 1
    n_eq = n_surf

    tol_surf = opt_settings.get("tol_surf", 1e-3)
    tol_joint = opt_settings.get("tol_joint", 1e-2)
    tol_col = opt_settings.get("tol_col", 1e-3)
    tol_fclosure = opt_settings.get("tol_fclosure", 1e-5)

    tol_eq = tol_surf * np.ones(n_eq)  # surface constraint tolerances
    tol_ineq = tol_col * np.ones(n_ineq)  # collision constraint tolerances
    tol_ineq[:n_joint] = tol_joint  # joint limit constraint tolerances
    tol_ineq[n_joint + n_col] = tol_fclosure  # fclosure constraint tolerance

    # making the optimizer
    alg = opt_settings.get("alg", nlopt.LD_SLSQP)
    opt = nlopt.opt(alg, n)
    opt.set_xtol_rel(1e-6)
    opt.set_xtol_abs(1e-6)
    opt.set_maxeval(1000)
    opt.set_min_objective(f)
    opt.add_inequality_mconstraint(g, tol_ineq)
    opt.add_equality_mconstraint(h, tol_eq)
    if "maxtime" in opt_settings:
        opt.set_maxtime(opt_settings["maxtime"])

    # initializing the PickupSystem for simulating picks
    # if pick:
    #     pickup_system = PickupSystem(
    #         model,
    #         t_lift=0.2,  # start the pick at 0.2 seconds
    #         hold_duration=1.5,  # hold for 1.5 seconds post-pick
    #         lift_duration=1.0,  # lift for 1.0 seconds
    #         lift_height=0.1,  # lift to 10cm
    #         visualize=False,  # [NOTE] turn this on to view the pick
    #     )

    # values computed during the benchmark
    sample_times = []  # sampling time
    solve_times = []  # solve time
    total_times = []  # total times (all sampling + solves)

    q0s = []  # initial configurations
    f_stars = []
    q_stars = []
    fc_stars = []
    l_stars = []  # only for our model
    q0_seeds = []
    q_star_seeds = []

    surf_cons_vio = []  # constraint violations
    # couple_cons_vio = []
    joint_cons_vio = []
    col_cons_vio = []
    fclosure_cons_vio = []

    num_solves = []  # number of solves before finding accepted sample
    num_iks = []  # number of IK calls required for initial condition generation

    pick_success = []
    X_WO_orig = RigidTransform(obj.X_WO)

    # defining the initial condition sampler
    sampler = HeuristicFR3AlgrICSampler(model)

    # running experiment
    b = 0  # the number of samples collected out of the desired batch
    seed = 0
    print("Beginning experiment...")
    while b < num_feas_samples:
        b = b + 1
        print(f"    Sample {b}/{num_feas_samples}")
        model.reset()  # resetting any internal model variables
        model.set_X_WO(X_WO_orig)  # reset the sim back to original pose post-sim

        solve_counter = 0  # counting how many solves are required before acceptance
        ik_counter = 0  # counting how many times IK is called for initial conditions
        feas = False

        tot_start = time.time()
        timed_out = False
        while not feas:
            seed = seed + 1
            solve_counter = solve_counter + 1

            # ############################## #
            # sampling initial configuration #
            # ############################## #

            # if a desired pose is provided and is not kinematically feasible, skip it
            t_start = time.time()
            q0, _ik_iters = sampler.sample_configuration(seed=seed)
            ik_counter = ik_counter + _ik_iters

            t_end = time.time()
            sample_times.append(t_end - t_start)
            q0s.append(q0)
            q0_seeds.append(seed)
            print(f"      obj: {obj.name} | sample: {b} | seed: {seed}")  # [DEBUG]

            # ################# #
            # running optimizer #
            # ################# #

            t_start = time.time()
            try:
                q_star = opt.optimize(q0)
            except (RuntimeError, ValueError, nlopt.RoundoffLimited):
                # [NOTE] RuntimeError catches two extremely rare errors:
                #        "RuntimeError: bug: more than iter SQP iterations"
                #          -this is an NLOPT error
                #        "RuntimeError: Error with configuration"
                #          -see: github.com/RobotLocomotion/drake/issues/18704
                # [NOTE] ValueError catches a rare error involving nans appearing in
                #        MeshObject gradient computation
                q_star = np.nan * np.ones(n)

            # checking feasibility and caching relevant values
            t_end = time.time()
            solve_times.append(t_end - t_start)

            if np.any(np.isnan(q_star)):
                continue  # nan means an error - resample
            else:
                # computing f, g, h at q_star
                f_val = f(q_star, np.zeros(0))
                g_val = np.zeros(n_ineq)
                g(g_val, q_star, np.zeros(0))
                h_val = np.zeros(n_eq)
                h(h_val, q_star, np.zeros(0))

                surf_vio = np.max(np.abs(h_val[:n_surf]))
                # if n_couple > 0:
                #     couple_vio = np.max(np.abs(h_val[n_surf:]))
                # else:
                #     couple_vio = 0.0
                joint_vio = max(np.max(g_val[:n_joint]), 0.0)
                col_vio = max(np.max(g_val[n_joint:-1]), 0.0)
                fclosure_vio = max(g_val[-1], 0.0)

                # setting the feasibility flag
                feas = (
                    surf_vio <= tol_surf
                    # and couple_vio <= tol_couple
                    and joint_vio <= tol_joint
                    and col_vio <= tol_col
                    and fclosure_vio <= tol_fclosure
                )

            # checking for timeout
            curr_time = time.time()
            if curr_time - tot_start >= timeout:
                timed_out = True
                print("    FAILURE - TIMED OUT")
                break

        tot_end = time.time()
        if timed_out:
            total_times.append(np.nan)
        else:
            total_times.append(tot_end - tot_start)

        # caching number of solves/IK calls required
        num_solves.append(solve_counter)
        num_iks.append(ik_counter)

        # cache optimal value/cost + constraint violations
        if timed_out:
            f_stars.append(np.nan)
            q_stars.append(np.nan * np.ones(model.n))
            q_star_seeds.append(np.nan)
            l_stars.append(np.nan)
            surf_cons_vio.append(np.nan)
            # couple_cons_vio.append(np.nan)
            joint_cons_vio.append(np.nan)
            col_cons_vio.append(np.nan)
            fclosure_cons_vio.append(np.nan)
            fc_stars.append(np.nan)
        else:
            f_stars.append(f_val)
            q_stars.append(q_star)
            q_star_seeds.append(seed)
            l_stars.append(model.compute_l(q_star))

            surf_cons_vio.append(surf_vio)
            # couple_cons_vio.append(couple_vio)
            joint_cons_vio.append(joint_vio)
            col_cons_vio.append(col_vio)
            fclosure_cons_vio.append(fclosure_vio)

            # compute Ferrari-Canny
            # [NOTE] for numerical reasons, we may sometimes get a solution that is
            # just out of force closure, but passes feasibility checks. In this case,
            # we set the value of the ferrari-canny metric to 0.0.
            G = model.compute_G(q_star)
            fcl1 = max(ferrari_canny_L1(G, mu, c=0.0, ns=ns, nc=nc), 0.0)
            fc_stars.append(fcl1)

        # simulate feasible (past quality threshold) picks
        # if pick and not timed_out:
        #     try:
        #         print("      Simulating...")  # [DEBUG]
        #         success = pickup_system.simulate(q_star, X_WO_0=X_WO_orig)[-1]
        #     except RuntimeError:
        #         # catches "MultibodyPlant's discrete update solver failed to converge"
        #         success = False
        # else:
        #     success = np.nan
        # pick_success.append(success)

    # saving the benchmark results
    model.obj.X_WO = X_WO_orig  # resetting pickled pose
    results = {
        "solve_times": np.stack(solve_times),
        "sample_times": np.stack(sample_times),
        "total_times": np.stack(total_times),
        "num_solves": np.stack(num_solves),
        "num_iks": np.stack(num_iks),
        "q0s": np.stack(q0s),
        "f_stars": np.stack(f_stars),
        "q_stars": np.stack(q_stars),
        "fc_stars": np.stack(fc_stars),
        "q0_seeds": np.stack(q0_seeds),
        "q_star_seeds": np.stack(q_star_seeds),
        "l_stars": np.stack(l_stars),
        "surf_cons_vio": np.stack(surf_cons_vio),
        # "couple_cons_vio": np.stack(couple_cons_vio),
        "joint_cons_vio": np.stack(joint_cons_vio),
        "col_cons_vio": np.stack(col_cons_vio),
        "fclosure_cons_vio": np.stack(fclosure_cons_vio),
        # "pick_success": np.stack(pick_success),
        "model_settings": model.cfg,  # also pickle settings
        "obj_settings": obj.cfg,
        "opt_settings": opt_settings,
    }
    obj_name = model.obj.name
    model_name = model.name

    if len(suffix) > 0:
        exp_name = f"exp_{obj_name}_{model_name}_{suffix}"
    else:
        exp_name = f"exp_{obj_name}_{model_name}"
    path_str = ROOT + f"/data/{obj_name}/{exp_name}.pkl"
    with open(path_str, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Finished experiment!")


def compute_results(model: RobotModel, suffix: str = "") -> None:
    """Computes the results to be reported in the paper.

    Parameters
    ----------
    model : RobotModel
        A model of the robot that performs cached computation.
    suffix : str, default=""
        Additional suffix to add onto an experiment file name. Here, the suffix will
        denote which experimental configuration was run.
    """
    # loading experiment data
    obj_name = model.obj.name
    model_name = model.name
    if len(suffix) > 0:
        exp_name = f"exp_{obj_name}_{model_name}_{suffix}"
    else:
        exp_name = f"exp_{obj_name}_{model_name}"
    path_str = ROOT + f"/data/{obj_name}/{exp_name}.pkl"
    try:
        with open(path_str, "rb") as handle:
            results = pickle.load(handle)
    except FileNotFoundError:
        raise FileNotFoundError("File not found! Check the benchmark has been run.")

    # unpacking data
    solve_times = results["solve_times"]
    sample_times = results["sample_times"]
    total_times = results["total_times"]

    num_solves = results["num_solves"]
    num_iks = results["num_iks"]

    # q0s = results["q0s"]

    f_stars = results["f_stars"]
    # q_stars = results["q_stars"]
    fc_stars = results["fc_stars"]
    l_stars = results["l_stars"]

    # q0_seeds = results["q0_seeds"]
    # q_star_seeds = results["q_star_seeds"]

    # surf_cons_vio = results["surf_cons_vio"]
    # couple_cons_vio = results["couple_cons_vio"]
    # joint_cons_vio = results["joint_cons_vio"]
    # col_cons_vio = results["col_cons_vio"]
    # fclosure_cons_vio = results["fclosure_cons_vio"]

    # pick_success = results["pick_success"]

    # obj_settings = results["obj_settings"]
    # model_settings = results["model_settings"]
    # opt_settings = results["opt_settings"]

    # tol_joint = opt_settings.get("tol_joint", 1e-2)
    # tol_couple = opt_settings.get("tol_couple", 1e-4)
    # tol_surf = opt_settings.get("tol_surf", 1e-3)
    # tol_col = opt_settings.get("tol_col", 1e-4)

    # checking for timeouts, excluding from stat computation
    num_timeouts = np.sum(np.isnan(pick_success))
    total_times = total_times[~np.isnan(total_times)]
    f_stars = f_stars[~np.isnan(f_stars)]
    fc_stars = fc_stars[~np.isnan(fc_stars)]
    l_stars = l_stars[~np.isnan(l_stars)]
    # pick_success = pick_success[~np.isnan(pick_success)]

    # printing relevant results
    # results will be formatted as "MEDIAN (QUANTILE25, QUANTILE75)".
    B = len(f_stars)  # number of samples
    print(f"OBJECT: {obj_name} | CONFIG: {suffix}")

    # number of solves and IK calls
    med = np.median(num_solves)
    q25 = np.quantile(num_solves, 0.25)
    q75 = np.quantile(num_solves, 0.75)
    print(f"  NUMBER OF SOLVES: {med} ({q25}, {q75})")

    med = np.median(num_iks)
    q25 = np.quantile(num_iks, 0.25)
    q75 = np.quantile(num_iks, 0.75)
    print(f"  NUMBER OF IK CALLS: {med} ({q25}, {q75})")

    # runtime per section
    med = np.median(sample_times)
    q25 = np.quantile(sample_times, 0.25)
    q75 = np.quantile(sample_times, 0.75)
    print(f"  SAMPLING TIMES: {med} ({q25}, {q75})")

    med = np.median(solve_times)
    q25 = np.quantile(solve_times, 0.25)
    q75 = np.quantile(solve_times, 0.75)
    print(f"  OPTIMIZER SOLVE TIMES: {med} ({q25}, {q75})")

    if B > 0:
        med = np.median(total_times)
        q25 = np.quantile(total_times, 0.25)
        q75 = np.quantile(total_times, 0.75)
        print(f"  TOTAL SOLVE TIMES: {med} ({q25}, {q75})")

        # ferrari-canny value (for our model, we also report l^*)
        med = np.median(fc_stars)
        q25 = np.quantile(fc_stars, 0.25)
        q75 = np.quantile(fc_stars, 0.75)
        print(f"  EPSILON METRIC: {med} ({q25}, {q75})")

        m = model.nc * model.ns
        med = np.median(l_stars * m)
        q25 = np.quantile(l_stars * m, 0.25)
        q75 = np.quantile(l_stars * m, 0.75)
        print(f"  NORMALIZED MIN-WEIGHT METRIC: {med} ({q25}, {q75})")

    # pick successes
    # print(f"  PICK SUCCESSES: {np.sum(pick_success.astype(int))}/{B}")
    print(f"  NUM TIMEOUTS: {num_timeouts}")


def summarize_all_results() -> None:
    """Summarizes all results for every object category.

    This prints the results presented in the IROS paper table. There are many hardcoded
    variables here that assume the exact experimental setup we used.
    """
    # explicit object categories
    sph_obj_names = [
        "012_strawberry",
        "013_apple",
        "014_lemon",
        "015_peach",
        "016_pear",
        "017_orange",
        "018_plum",
        "054_softball",
        "055_baseball",
        "056_tennis_ball",
        "057_racquetball",
        "058_golf_ball",
    ]
    boxcyl_obj_names = [
        "001_chips_can",
        "002_master_chef_can",
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "010_potted_meat_can",  # basically a box with rounded edges
        "021_bleach_cleanser",
        "036_wood_block",
        "061_foam_brick",
        "065-f_cups",
        "065-g_cups",
        "065-h_cups",
        "065-i_cups",
        "065-j_cups",
        "077_rubiks_cube",
    ]
    adv_obj_names = [
        "007_tuna_fish_can",
        "008_pudding_box",
        "009_gelatin_box",
        "011_banana",
        "043_phillips_screwdriver",
        "044_flat_screwdriver",
        "048_hammer",
        "051_large_clamp",
        "052_extra_large_clamp",
        "065-a_cups",
        "065-b_cups",
        "065-c_cups",
        "065-d_cups",
        "065-e_cups",
        "sns_cup",
    ]

    # list of all obj name lists
    obj_names_list = [
        sph_obj_names,
        boxcyl_obj_names,
        adv_obj_names,
    ]

    for i in range(2):
        if i == 0:
            print("RESULTS: OUR METHOD")
        else:
            print("RESULTS: BASELINE")

        # pick_successes_all = []
        fc_vals_all = []
        normalized_ls_all = []
        solve_times_all = []
        num_solves_all = []
        total_times_all = []
        num_timeouts_all = 0

        # looping through all object categories and aggregating statistics
        for j, obj_name_list in enumerate(obj_names_list):

            # pick_successes_cat = []
            fc_vals_cat = []
            normalized_ls_cat = []
            solve_times_cat = []
            num_solves_cat = []
            total_times_cat = []
            num_timeouts_cat = 0

            for obj_name in obj_name_list:
                # loading experiment data
                model_name = "fr3_arm_algr_rh"
                exp_name = f"exp_{obj_name}_{model_name}_{i + 1}"
                path_str = ROOT + f"/data/{obj_name}/{exp_name}.pkl"
                try:
                    with open(path_str, "rb") as handle:
                        results = pickle.load(handle)
                except FileNotFoundError:
                    raise FileNotFoundError(
                        "File not found! Check the benchmark has been run."
                    )

                # recovering relevant data
                solve_times = results["solve_times"]
                total_times = results["total_times"]
                num_solves = results["num_solves"]
                fc_stars = results["fc_stars"]
                l_stars = results["l_stars"]
                # pick_success = results["pick_success"]

                # checking for timeouts, excluding from stat computation
                # num_timeouts_cat += np.sum(np.isnan(pick_success))
                total_times = total_times[~np.isnan(total_times)]
                fc_stars = fc_stars[~np.isnan(fc_stars)]
                l_stars = l_stars[~np.isnan(l_stars)]
                # pick_success = pick_success[~np.isnan(pick_success)]

                # aggregating results
                # pick_successes_cat.append(pick_success.astype(int))
                fc_vals_cat.append(fc_stars)
                m = 16
                normalized_ls_cat.append(l_stars * m)
                solve_times_cat.append(solve_times)
                num_solves_cat.append(num_solves)
                total_times_cat.append(total_times)

            # computing aggregated statistics per-category
            # pick_successes_cat = np.concatenate(pick_successes_cat)
            fc_vals_cat = np.concatenate(fc_vals_cat)
            normalized_ls_cat = np.concatenate(normalized_ls_cat)
            solve_times_cat = np.concatenate(solve_times_cat)
            num_solves_cat = np.concatenate(num_solves_cat)
            total_times_cat = np.concatenate(total_times_cat)

            if j == 0:
                print("  SPHEROIDS")
            elif j == 1:
                print("  BOXES/CYLINDERS/BOTTLES")
            elif j == 2:
                print("  ADVERSARIAL OBJECTS")
            # print(
            #     f"    PICK SUCCESS: {np.sum(pick_successes_cat)}/{len(pick_successes_cat)}"
            # )
            print(
                f"    FERRARI-CANNY (median + IQR): {np.median(fc_vals_cat)} ({np.quantile(fc_vals_cat, 0.25)}, {np.quantile(fc_vals_cat, 0.75)})"
            )
            print(
                f"    FERRARI-CANNY (avg + 1std): {np.mean(fc_vals_cat)} +/- {np.std(fc_vals_cat)}"
            )
            print(
                f"    NORMALIZED l* (median + IQR): {np.median(normalized_ls_cat)} ({np.quantile(normalized_ls_cat, 0.25)}, {np.quantile(normalized_ls_cat, 0.75)}"
            )
            print(
                f"    NORMALIZED l* (avg + 1std): {np.mean(normalized_ls_cat)} +/- {np.std(normalized_ls_cat)}"
            )
            print(
                f"    PER-SOLVE TIME (median + IQR): {np.median(solve_times_cat)} ({np.quantile(solve_times_cat, 0.25)}, {np.quantile(solve_times_cat, 0.75)})"
            )
            print(
                f"    PER-SOLVE TIME (avg + 1std): {np.mean(solve_times_cat)} +/- {np.std(solve_times_cat)}"
            )
            print(
                f"    NUM SOLVES (median + IQR): {np.median(num_solves_cat)} ({np.quantile(num_solves_cat, 0.25)}, {np.quantile(num_solves_cat, 0.75)})"
            )
            print(
                f"    NUM SOLVES (avg + 1std): {np.mean(num_solves_cat)} +/- {np.std(num_solves_cat)}"
            )
            print(
                f"    TOTAL TIME (median + IQR): {np.median(total_times_cat)} ({np.quantile(total_times_cat, 0.25)}, {np.quantile(total_times_cat, 0.75)})"
            )
            print(
                f"    TOTAL TIME (avg + 1std): {np.mean(total_times_cat)} +/- {np.std(total_times_cat)}"
            )
            print(f"    NUM TIMEOUTS: {num_timeouts_cat}")

            # pick_successes_all.append(pick_successes_cat)
            fc_vals_all.append(fc_vals_cat)
            normalized_ls_all.append(normalized_ls_cat)
            solve_times_all.append(solve_times_cat)
            num_solves_all.append(num_solves_cat)
            total_times_all.append(total_times_cat)
            num_timeouts_all += num_timeouts_cat

        # computing aggregated total results
        # pick_successes_all = np.concatenate(pick_successes_all)
        fc_vals_all = np.concatenate(fc_vals_all)
        normalized_ls_all = np.concatenate(normalized_ls_all)
        solve_times_all = np.concatenate(solve_times_all)
        num_solves_all = np.concatenate(num_solves_all)
        total_times_all = np.concatenate(total_times_all)
        print("  TOTAL")
        # print(
        #     f"    PICK SUCCESS: {np.sum(pick_successes_all)}/{len(pick_successes_all)}"
        # )
        print(
            f"    FERRARI-CANNY (median + IQR): {np.median(fc_vals_all)} ({np.quantile(fc_vals_all, 0.25)}, {np.quantile(fc_vals_all, 0.75)})"
        )
        print(
            f"    FERRARI-CANNY (avg + 1std): {np.mean(fc_vals_all)} +/- {np.std(fc_vals_all)}"
        )
        print(
            f"    NORMALIZED l* (median + IQR): {np.median(normalized_ls_all)} ({np.quantile(normalized_ls_all, 0.25)}, {np.quantile(normalized_ls_all, 0.75)})"
        )
        print(
            f"    NORMALIZED l* (avg + 1std): {np.mean(normalized_ls_all)} +/- {np.std(normalized_ls_all)}"
        )
        print(
            f"    PER-SOLVE TIME (median + IQR): {np.median(solve_times_all)} ({np.quantile(solve_times_all, 0.25)}, {np.quantile(solve_times_all, 0.75)})"
        )
        print(
            f"    PER-SOLVE TIME (avg + 1std): {np.mean(solve_times_all)} +/- {np.std(solve_times_all)}"
        )
        print(
            f"    NUM SOLVES (median + IQR): {np.median(num_solves_all)} ({np.quantile(num_solves_all, 0.25)}, {np.quantile(num_solves_all, 0.75)})"
        )
        print(
            f"    NUM SOLVES (avg + 1std): {np.mean(num_solves_all)} +/- {np.std(num_solves_all)}"
        )
        print(
            f"    TOTAL TIME (median + IQR): {np.median(total_times_all)} ({np.quantile(total_times_all, 0.25)}, {np.quantile(total_times_all, 0.75)})"
        )
        print(
            f"    TOTAL TIME (avg + 1std): {np.mean(total_times_all)} +/- {np.std(total_times_all)}"
        )
        print(f"    NUM TIMEOUTS: {num_timeouts_all}")
        print()
