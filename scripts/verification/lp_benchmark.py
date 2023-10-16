# flake8: noqa
import time

import cvxpy as cp
import jax
import numpy as np
from cvxpylayers.jax import CvxpyLayer
from numba import jit
from pydrake.math import RigidTransform, RotationMatrix
from quantecon.optimize.linprog_simplex import linprog_simplex as linprog

from frogger.objects import Sphere
from frogger.robots.robots import FR3AlgrModel

"""
This script tests the LP and its gradient solve times.
"""

# numba-compiled methods
@jit(nopython=True, fastmath=True, cache=True)
def fast_lp_solve(W):
    """Fast LP solve by calling numba-compiled LP solver."""
    m = W.shape[-1]
    Im = np.eye(m)
    ones_m = np.ones((m, 1))
    w_bar = np.zeros(6)  # wrench whose convex hull membership is checked

    c = np.concatenate((np.zeros(2 * m), np.array([1, -1])))
    top = np.concatenate((W, -W, np.zeros((6, 2))), axis=1)
    bot = np.concatenate((ones_m.T, -ones_m.T, np.zeros((1, 2))), axis=1)
    Aeq = np.concatenate((top, bot), axis=0)
    beq = np.concatenate((w_bar, np.array([1.0])))
    Ain = np.concatenate((-Im, Im, ones_m, -ones_m), axis=1)
    res = linprog(c, A_ub=Ain, b_ub=np.zeros(m), A_eq=Aeq, b_eq=beq)
    x_ = res.x
    x_p = np.concatenate((x_[:m], np.array([x_[-2]])))
    x_m = np.concatenate((x_[m : 2 * m], np.array([x_[-1]])))

    x_opt = x_p - x_m
    lamb_opt = res.lambd[:m]
    nu_opt = res.lambd[m:]

    return x_opt, lamb_opt, nu_opt


def time_cvxpy():
    cvxpy_times = []
    ns = 4
    nc = 4
    m = ns * nc
    n_samples = 100
    Ws = [np.random.rand(6, m) for _ in range(n_samples)]  # random wrench matrices

    # defining cvx problem
    alpha = cp.Variable(m)  # convex weights
    l = cp.Variable(1)  # minimum weight lower bound
    W = cp.Parameter((6, m))  # wrench matrix
    constraints = [
        W @ alpha == np.zeros(6),
        cp.sum(alpha) == 1.0,
        alpha >= l,
    ]
    objective = cp.Minimize(-l)  # maximizing the minimum convex weight
    cvxpy_problem = cp.Problem(objective, constraints)
    assert cvxpy_problem.is_dpp()
    cvxpylayer = CvxpyLayer(cvxpy_problem, parameters=[W], variables=[alpha, l])
    Dl_W_cvxpy = jax.grad(
        lambda X: cvxpylayer(X, solver_args={"solve_method": "ECOS"})[-1][0]
    )
    Dl_W_cvxpy(Ws[0])  # first-time canonicalization

    # timing
    t1 = time.time()
    for i in range(n_samples):
        _ = Dl_W_cvxpy(Ws[i])
    t2 = time.time()
    print(f"    cvxpylayers avg time: {(t2 - t1) / n_samples}")


def time_exploit():
    exploit_times = []
    ns = 4
    r = 0.03
    obj_settings = {
        "r": r,
        "X_WO": RigidTransform(
            RotationMatrix(),
            np.array([0.7, 0.0, r]),
        ),
    }
    obj = Sphere(obj_settings)
    model_settings = {
        "ns": int(ns),  # sides of pyramidal approximation
        "mu": 0.5,  # coeff of friction
        "hand": "rh",  # right hand
        "simplified": True,  # simplified collision geometry
        "th_t": np.pi / 3,  # contact point tilt
        "d_min": 0.001,  # minimum safety margin for collision
        "d_pen": 0.003,  # allowed penetration of fingertip into object
        "pregrasp_type": "closed",  # only used for sample generation
    }
    model = FR3AlgrModel(obj, model_settings, verbose=True)

    # redefining problem parameters
    nc = model.settings["nc"]
    ns = model.settings["ns"]
    m = ns * nc  # total number of wrenches
    n_samples = 100
    Ws = [np.random.rand(6, m) for _ in range(n_samples)]  # random wrench matrices

    t1 = time.time()
    for i in range(n_samples):
        model.W = Ws[i]
        _ = model._compute_l()  # computes l and its gradient
    t2 = time.time()
    print(f"    our method (with exploit) avg time: {(t2 - t1) / n_samples}")


if __name__ == "__main__":
    time_cvxpy()
    time_exploit()
