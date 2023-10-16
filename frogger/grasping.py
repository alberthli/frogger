import numpy as np
from numba import jit
from scipy.spatial import ConvexHull


@jit(nopython=True, fastmath=True, cache=True)
def wedge(a: np.ndarray) -> np.ndarray:
    """Skew-symmetric wedge operator. Can compute in batches.

    Parameters
    ----------
    a : np.ndarray, shape=(..., 3)
        Input vector.

    Returns
    -------
    a_wedge : np.ndarray, shape=(..., 3, 3)
        Skew-symmetric form of a.
    """
    assert a.shape[-1] == 3
    a1 = a[..., 0]
    a2 = a[..., 1]
    a3 = a[..., 2]
    a_wedge = np.zeros(a.shape[:-1] + (3, 3))
    a_wedge[..., 0, 1] = -a3
    a_wedge[..., 0, 2] = a2
    a_wedge[..., 1, 0] = a3
    a_wedge[..., 1, 2] = -a1
    a_wedge[..., 2, 0] = -a2
    a_wedge[..., 2, 1] = a1
    return a_wedge


@jit(nopython=True, fastmath=True, cache=True)
def vee(A: np.ndarray) -> np.ndarray:
    """Inverse of the wedge operator. Can compute in batches.

    Parameters
    ----------
    A : np.ndarray, shape=(..., 3, 3)
        A skew-symmetric matrix.

    Returns
    -------
    A_vee : np.ndarray, shape=(..., 3)
        The vee'd representation of A.
    """
    assert A.shape[-2:] == (3, 3)
    A_vee = np.zeros(A.shape[:-1])
    A_vee[..., 0] = A[..., 2, 1]
    A_vee[..., 1] = A[..., 0, 2]
    A_vee[..., 2] = A[..., 1, 0]
    return A_vee


def compute_g_inv(g: np.ndarray) -> np.ndarray:
    """Inverse of homogeneous transform. Can compute in batches.

    Parameters
    ----------
    g : np.ndarray, shape=(..., 4, 4)
        Input homogeneous transform.

    Returns
    -------
    g_inv : np.ndarray, shape=(..., 4, 4)
        Inverse of g.
    """
    assert g.shape[-2:] == (4, 4)
    R = g[..., :3, :3]  # (..., 3, 3)
    p = g[..., :3, -1:]  # (..., 3, 1)

    Rt = np.swapaxes(R, -2, -1)

    top = np.concatenate((Rt, -Rt @ p), axis=-1)
    bottom = np.zeros(Rt.shape[:-2] + (1, 4))
    bottom[..., -1] = 1
    g_inv = np.concatenate((top, bottom), axis=-2)
    return g_inv


def compute_adjoint(g: np.ndarray) -> np.ndarray:
    """Computes the adjoint of a homogeneous transform. Can compute in batches.

    Parameters
    ----------
    g : np.ndarray, shape=(..., 4, 4)
        Input homogeneous transform.

    Returns
    -------
    Ad_g : np.ndarray, shape=(..., 6, 6)
        Adjoint of g.
    """
    assert g.shape[-2:] == (4, 4)
    R = g[..., :3, :3]  # (..., 3, 3)
    p = g[..., :3, -1]  # (..., 3, 1)

    Ad_g = np.zeros(R.shape[:-2] + (6, 6))
    Ad_g[..., :3, :3] = R
    Ad_g[..., 3:, 3:] = R
    Ad_g[..., :3, 3:] = wedge(p) @ R
    return Ad_g


@jit(nopython=True, fastmath=True, cache=True)
def compute_gOCs(ps: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Computes the transformation from contact to object frame.

    Parameters
    ----------
    ps : np.ndarray, shape=(3, nc)
        Location of contacts in the object frame. ps=P_OCs.
    normals : np.ndarray, shape=(3, nc)
        Normal vectors at each point of contact in the object frame. These point INTO
        the object as a convention. Keywords: inward, outward, normal.

    Returns
    -------
    g_OCs : np.ndarray, shape=(nc, 4, 4)
        Rigid body transforms from the contact frames to the world.
    """
    nc = ps.shape[1]

    # generating tangent vector using projection and cross product
    zeta = np.array([1.2, 2.3, 3.4])  # arbitrary choices for creating tangent vectors
    zeta_prime = np.array([3.4, 2.3, 1.2])

    txs = np.repeat(zeta, nc).reshape((3, nc))
    flag = np.sqrt(np.sum((normals - np.expand_dims(zeta, -1)) ** 2, axis=0)) <= 1e-6
    txs[:, flag] = np.expand_dims(zeta_prime, -1)  # if normal = zeta, use zeta_prime

    n_dot_ts = np.diag(normals.T @ txs)  # <n, t1>
    txs -= normals * n_dot_ts  # subtracting the projected component
    txs /= np.sqrt(np.sum(txs**2, axis=0))  # normalizing
    tys = np.cross(normals.T, txs.T).T

    # creating contact frame rotations
    # double swapaxes same as `np.moveaxis(_Rs, (0, 1, 2), (1, 2, 0))`
    # written this way for numba-compliance since moveaxis isn't supported yet
    _Rs = np.stack((txs, tys, normals), axis=-2)
    Rs = np.swapaxes(np.swapaxes(_Rs, 0, 1), 0, 2)

    # creating transforms
    top = np.concatenate((Rs, np.expand_dims(ps.T, -1)), axis=-1)
    bottom = np.zeros(Rs.shape[:-2] + (1, 4))
    bottom[..., -1] = 1
    g_OCs = np.concatenate((top, bottom), axis=-2)

    return g_OCs


def compute_grasp_matrix(g_OCs: np.ndarray, model: str = "hard") -> np.ndarray:
    """Computes the grasp matrix.

    Parameters
    ----------
    g_OCs : np.ndarray, shape=(nc, 4, 4)
        An array of all the rigid body transforms from contact frame i to the world.
        Each element is given by the 4x4 homogeneous transform.
    model : str, default="hard"
        Contact model. Can be "hard" or "soft".

    Returns
    -------
    G : np.ndarray, shape=(6, nc*nb)
        Grasp matrix. nc=number of contacts, nb=number of wrench basis vectors.
        nb is the number of basis vectors for contact wrench. With a hard contact
        model, nb=3. With soft, nb=4.
    """
    assert model in ["hard", "soft"]

    # constructing wrench basis
    if model == "hard":
        Bc = np.zeros((6, 3))
        Bc[0, 0] = 1
        Bc[1, 1] = 1
        Bc[2, 2] = 1
    elif model == "soft":
        Bc = np.zeros((6, 4))
        Bc[0, 0] = 1
        Bc[1, 1] = 1
        Bc[2, 2] = 1
        Bc[-1, -1] = 1
    else:
        raise NotImplementedError

    # computing grasp matrix
    # (nc, 6, 6)
    Ad_ginv_T = np.swapaxes(compute_adjoint(compute_g_inv(g_OCs)), -2, -1)
    G_stacked = Ad_ginv_T @ Bc  # (nc, 6, nb)
    G = np.hstack(G_stacked)  # (6, nc*nb)
    return G


def compute_primitive_forces(ns: int, mu: float, model: str = "hard"):
    """Computes primitive contact forces in a contact frame for a pyramidal approx.

    These are the forces along the edges of a friction pyramid with unit norm.

    Parameters
    ----------
    ns : int
        Number of sides of pyramid.
    mu : float
        Coefficient of friction.
    model : str, default="hard"
        Contact model. Can be "hard" or "soft".

    Returns
    -------
    fs : np.ndarray, shape=(nb, ns)
        Array of primitive forces. nb=3 for hard contact, nb=4 for soft contact.
    """
    assert ns >= 3
    assert mu > 0.0
    assert model in ["hard", "soft"]
    if model == "hard":
        fs = np.zeros((3, ns))
        nums = np.arange(ns)
        fs[0, :] = mu * np.sqrt(1 / (1 + mu**2)) * np.cos(2 * np.pi * nums / ns)
        fs[1, :] = mu * np.sqrt(1 / (1 + mu**2)) * np.sin(2 * np.pi * nums / ns)
        fs[2, :] = np.sqrt(1 / (1 + mu**2))
    elif model == "soft":
        raise NotImplementedError
    else:
        raise ValueError("model must be 'hard' or 'soft'!")
    return fs


def ferrari_canny_L1(
    G: np.ndarray,
    mu: float,
    c: np.ndarray | None = None,
    ns: int = 4,
    nc: int = 4,
    lamb: float = 1.0,
) -> float:
    """Computes the L1 Ferrari-Canny metric.

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
