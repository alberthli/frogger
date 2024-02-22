from typing import TYPE_CHECKING

import numpy as np
from numba import jit
from quantecon.optimize.linprog_simplex import linprog_simplex as linprog
from scipy.spatial import ConvexHull

from frogger.grasping import compute_primitive_forces

if TYPE_CHECKING:
    from frogger.robots.robot_core import RobotModel

# ############# #
# FERRARI-CANNY #
# ############# #


def _ferrari_canny_L1(
    G: np.ndarray,
    mu: float,
    c: np.ndarray | None = None,
    ns: int = 4,
    nc: int = 4,
    lamb: float = 1.0,
) -> float:
    """Computes the L1 Ferrari-Canny metric.

    TODO(ahl): convert this to a numba function.

    Parameters
    ----------
    G : np.ndarray, shape=(6, nc*nb)
        Grasp matrix. nb=3 for hard contact, nb=4 for soft contact.
    mu : float
        Coefficient of friction.
    c : np.ndarray | None, default=None
        The center about which to check the ball radius. If None, defaults to 0.
    ns : int, default=4
        Number of sides of pyramidal friction cone approximation.
    nc : int, default=4
        Number of contact points.
    lamb : float, default=1.0
        The weighting parameter defining the wrench space metric. lamb=1.0 -> L2
        metric.
        For discussion of alternatives see paragraph before Sec. 3.3.2 of "Grasp
        quality measures: review and performance" by Roa et al.
        norm(w) = sqrt(|f|^2 + lamb*|tau|^2)

    Returns
    -------
    Q : float
        Grasp quality. Geometrically, the radius of the largest sphere that is
        contained in the grasp wrench space centered around c. Returns -1.0 if c is
        not contained in the convex hull.
    """
    assert mu > 0.0
    assert lamb > 0.0

    if c is None:
        c = np.zeros(6)

    # columns of W are the primitive wrenches.
    F = compute_primitive_forces(ns, mu, model="hard")  # (3, ns)
    W = G @ np.kron(np.eye(nc), F)  # generating primitive wrenches from forces
    W[3:, :] *= np.sqrt(lamb)  # scales the torques by lamb

    # hull.equations returns [A, b] for normals and offset of hyperplanes of facets.
    # The hull is defined as the polyhedron Ax + b <= 0, where rows of A are normals.
    # If min over all -b is negative, then the origin is not contained.
    # Otherwise, the quality is given by the min over all -b.
    # In general, if x is a point in the convex hull, Ax + b <= 0.
    W = ConvexHull(W.T - c)  # subtracts off the center from all points in W
    _Q = min(-W.equations[:, -1])
    Q = _Q if _Q >= 0.0 else -1.0
    return Q


# ################# #
# MIN-WEIGHT METRIC #
# ################# #


@jit(nopython=True, fastmath=True, cache=True)
def min_weight_lp(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given basis wrenches, solves the min-weight LP.

    Parameters
    ----------
    W : np.ndarray, shape=(6, m)
        The matrix of basis wrenches.

    Returns
    -------
    x_opt : np.ndarray, shape=(m + 1,)
        The optimal primal solution, x = (alpha, l), where the alphas are the
        convex weights and l is the min-weight.
    lamb_opt : np.ndarray, shape=(m,)
        The optimal dual solution for the inequality constraints.
    nu_opt : np.ndarray, shape=(7,)
        The optimal dual solution for the equality constraints.
    """
    m = W.shape[-1]

    Im = np.eye(m)
    ones_m = np.ones((m, 1))
    w_bar = np.zeros(6)  # wrench whose convex hull membership is checked

    # numba-compiled quantecon simplex method
    # [NOTE] we must convert the LP to standard form to use quantecon's linprog.
    # Thus, we add slack variables for the non-negativity constraints on the
    # standard form decision variables
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


@jit(nopython=True, fastmath=True, cache=True)
def min_weight_gradient(
    x_opt: np.ndarray,
    lamb_opt: np.ndarray,
    nu_opt: np.ndarray,
    W: np.ndarray,
    DW: np.ndarray,
) -> np.ndarray:
    """Computes the gradient of the min-weight metric with respect to q.

    Parameters
    ----------
    x_opt : np.ndarray, shape=(m + 1,)
        The optimal primal solution, x = (alpha, l), where the alphas are the
        convex weights and l is the min-weight.
    lamb_opt : np.ndarray, shape=(m,)
        The optimal dual solution for the inequality constraints.
    nu_opt : np.ndarray, shape=(7,)
        The optimal dual solution for the equality constraints.
    W : np.ndarray, shape=(6, m)
        The matrix of basis wrenches.
    DW : np.ndarray, shape=(6, m, n)
        The Jacobian of the basis wrenches with respect to q.

    Returns
    -------
    Dl : np.ndarray, shape=(n,)
        The gradient of the min-weight metric with respect to q.
    """
    m = lamb_opt.shape[0]
    Im = np.eye(m)
    ones_m = np.ones((m, 1))

    # [NOTE] these are the inequality and equality constraint matrices in the
    # original formulation. We use these to compute the gradient, not to set up
    # the LP, which must be put in standard form for quantecon's linprog method.
    Ain = np.concatenate((-Im, ones_m), axis=1)
    Aeq = np.zeros((7, m + 1))
    Aeq[:-1, :-1] = W
    Aeq[-1, :-1] = 1.0

    # jacobian of KKT wrt W
    _DH_W1_T = np.kron(Im, nu_opt[:6]).reshape((m, m, 6))
    _DH_W1 = np.swapaxes(_DH_W1_T, -1, -2)
    _DH_W2 = np.kron(np.eye(6), x_opt[:m]).reshape((6, 6, m))
    DH_W = np.concatenate(
        (_DH_W1, np.zeros((m + 1, 6, m)), _DH_W2, np.zeros((1, 6, m)))
    )
    C = np.vstack((Ain * np.expand_dims(lamb_opt, -1), Aeq))
    _DH_W = DH_W.reshape((2 * m + 8, 6 * m))  # 3d -> 2d for lstsq

    # computing Dl explicitly using chain rule + sparsity exploit
    RHS2 = _DH_W[m + 1 :, :]  # bottom block of _DH_W
    Dl_W = -np.linalg.lstsq(C, RHS2)[0][-1, :].reshape((6, m))
    Dl = Dl_W.reshape(-1) @ DW.reshape((-1, DW.shape[-1]))
    return Dl


# ########### #
# CONVENIENCE #
# ########### #


def min_weight_metric(robot: "RobotModel") -> float:
    """Convenience function for computing the min-weight metric.

    Assumes that the basis wrenches have already been computed and cached.

    Parameters
    ----------
    robot : RobotModel
        The robot model.
    """
    x_opt, lamb_opt, nu_opt = min_weight_lp(robot.W)
    return x_opt[-1]


def ferrari_canny_L1(robot: "RobotModel") -> float:
    """Convenience function for computing the Ferrari-Canny L1 metric.

    Assumes that the grasp matrix has already been computed and cached.

    Parameters
    ----------
    robot : RobotModel
        The robot model.
    """
    return _ferrari_canny_L1(robot.G, robot.mu, ns=robot.ns, nc=robot.nc, lamb=1.0)
