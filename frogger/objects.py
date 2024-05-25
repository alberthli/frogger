from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import open3d as o3d
import trimesh
from jax import hessian, jacobian, jit, vmap
from numba import jit as numba_jit
from pydrake.geometry import Convex, Mesh
from pydrake.math import RigidTransform
from skimage.measure import marching_cubes

from frogger import ROOT
from frogger.sdfs import poisson_reconstruction

jax.config.update("jax_platform_name", "cpu")  # force everything to run on CPU
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)  # errors out when encountering nans


@dataclass(kw_only=True)
class ObjectDescriptionConfig:
    """A configuration for an object description.

    Attributes
    ----------
    X_WO : RigidTransform
        The pose of the object with respect to the world frame.
    lb_O : np.ndarray | None, shape=(3,), default=None
        Lower bounds of the object in the object frame. Defaults to None.
    ub_O : np.ndarray | None, shape=(3,), default=None
        Upper bounds of the object in the object frame. Defaults to None.
    mass : float | None, default=None
        A known mass of the object.
    enforce_watertight : bool, default=False
        Whether to enforce checks on whether the mesh is watertight.
    name : str, default="abstract"
        The name of the object.
    """

    X_WO: RigidTransform
    lb_O: np.ndarray | None = None
    ub_O: np.ndarray | None = None
    mass: float | None = None
    enforce_watertight: bool = False

    # fields with overridden defaults in child classes
    name: str | None = None

    def __post_init__(self):
        """Used to handle overridden defaults."""
        if self.name is None:
            self.name = "abstract"

    def create(self) -> "ObjectDescription":
        """Creates the object description."""
        obj = ObjectDescription(self)
        return obj


class ObjectDescription(ABC):
    """Abstract class for describing an object to manipulate."""

    def __init__(self, cfg: ObjectDescriptionConfig) -> None:
        """Initialize the object description."""
        # unpacking config
        self.cfg = cfg
        self.X_WO = cfg.X_WO
        self.lb_O = cfg.lb_O
        self.ub_O = cfg.ub_O
        self.mass = cfg.mass
        self.name = cfg.name
        self.enforce_watertight = cfg.enforce_watertight

        # mesh objects compute the SDF and its gradients differently
        if not isinstance(self, MeshObject):
            # object description functions
            self._s_O = None  # s, its Jacobian, and its Hessian in the object frame
            self._Ds_O = None
            self._D2s_O = None
            self._s_W = None  # s, its Jacobian, and its Hessian in the world frame
            self._Ds_W = None
            self._D2s_W = None

            # pre-compile all Jax functions
            self._compute_functions_O()

        # bounds of the object
        if self.lb_O is not None and self.ub_O is not None:
            assert self.lb_O.shape == (3,) and self.ub_O.shape == (3,)
            self.lb_W = self.lb_O + self.X_WO.translation()
            self.ub_W = self.ub_O + self.X_WO.translation()
            self.compute_mesh()
        else:
            self.lb_W = None
            self.ub_W = None
            self.shape_visual = None
            self.shape_collision_list = None

    @abstractmethod
    def _s_O_jax(self, p: jnp.ndarray) -> jnp.ndarray:
        """The differentiable description of the object written in Jax.

        Parameters
        ----------
        p : jnp.ndarray, shape=(3,)
            A point in the object frame.

        Returns
        -------
        signed_dist : jnp.ndarray, shape=()
            The signed distance of p from the object.
        """

    def s_O(
        self, p: np.ndarray, jax_out: bool = False, batched: bool = False
    ) -> np.ndarray:
        """Computes the value of s_O(p)."""
        if batched:
            assert len(p.shape) == 2
            assert p.shape[-1] == 3
        else:
            assert p.shape == (3,)

        if self._s_O is None:
            self._compute_functions_O()
        val = self._s_O(p, jax_out=jax_out, batched=batched)

        if batched:
            assert len(val.shape) == 1
        else:
            assert val.shape == ()

        return val

    def Ds_O(
        self, p: np.ndarray, jax_out: bool = False, batched: bool = False
    ) -> np.ndarray:
        """Computes the value of Ds_O(p)."""
        if batched:
            assert len(p.shape) == 2
            assert p.shape[-1] == 3
        else:
            assert p.shape == (3,)

        if self._Ds_O is None:
            self._compute_functions_O()
        val = self._Ds_O(p, jax_out=jax_out, batched=batched)

        if batched:
            assert len(val.shape) == 2
            assert val.shape[-1] == 3
        else:
            assert val.shape == (3,)

        return val

    def D2s_O(
        self, p: np.ndarray, jax_out: bool = False, batched: bool = False
    ) -> np.ndarray:
        """Computes the value of D2s_O(p)."""
        if batched:
            assert len(p.shape) == 2
            assert p.shape[-1] == 3
        else:
            assert p.shape == (3,)

        if self._D2s_O is None:
            self._compute_functions_O()
        val = self._D2s_O(p, jax_out=jax_out, batched=batched)

        if batched:
            assert len(val.shape) == 3
            assert val.shape[-2:] == (3, 3)
        else:
            assert val.shape == (3, 3)

        return val

    def s_W(
        self, p: np.ndarray, jax_out: bool = False, batched: bool = False
    ) -> np.ndarray:
        """Computes the value of s_W(p)."""
        if batched:
            assert len(p.shape) == 2
            assert p.shape[-1] == 3
        else:
            assert p.shape == (3,)

        if self._s_W is None:
            self._compute_functions_W()
        val = self._s_W(p, jax_out=jax_out, batched=batched)

        if batched:
            assert len(val.shape) == 1
        else:
            assert val.shape == ()
        return val

    def Ds_W(
        self, p: np.ndarray, jax_out: bool = False, batched: bool = False
    ) -> np.ndarray:
        """Computes the value of Ds_W(p)."""
        if batched:
            assert len(p.shape) == 2
            assert p.shape[-1] == 3
        else:
            assert p.shape == (3,)

        if self._Ds_W is None:
            self._compute_functions_W()
        val = self._Ds_W(p, jax_out=jax_out, batched=batched)

        if batched:
            assert len(val.shape) == 2
            assert val.shape[-1] == 3
        else:
            assert val.shape == (3,)
        return val

    def D2s_W(
        self, p: np.ndarray, jax_out: bool = False, batched: bool = False
    ) -> np.ndarray:
        """Computes the value of D2s_W(p)."""
        if batched:
            assert len(p.shape) == 2
            assert p.shape[-1] == 3
        else:
            assert p.shape == (3,)

        if self._D2s_W is None:
            self._compute_functions_W()
        val = self._D2s_W(p, jax_out=jax_out, batched=batched)

        if batched:
            assert len(val.shape) == 3
            assert val.shape[-2:] == (3, 3)
        else:
            assert val.shape == (3, 3)
        return val

    def _compute_functions_O(self) -> None:
        """Computes functions, Jacobians, and Hessians and compiles them in O frame."""
        # jitting Ds, and D2s
        s_O_jit = jit(self._s_O_jax)

        @jit
        def Ds_jit(p):
            return jacobian(self._s_O_jax)(p)

        @jit
        def D2s_jit(p):
            return hessian(self._s_O_jax)(p)

        @jit
        def s_jit_batched(p):
            return vmap(self._s_O_jax)(p)

        @jit
        def Ds_jit_batched(p):
            return vmap(jacobian(self._s_O_jax))(p)

        @jit
        def D2s_jit_batched(p):
            return vmap(hessian(self._s_O_jax))(p)

        # ensure functions can return numpy arrays
        def _s_O(p, jax_out=False, batched=False):
            if batched:
                if jax_out:
                    return s_jit_batched(p)
                else:
                    return np.array(s_jit_batched(p))
            else:
                if jax_out:
                    return s_O_jit(p)
                else:
                    return np.array(s_O_jit(p))

        def _Ds_O(p, jax_out=False, batched=False):
            if batched:
                if jax_out:
                    return Ds_jit_batched(p)
                else:
                    return np.array(Ds_jit_batched(p))
            else:
                if jax_out:
                    return Ds_jit(p)
                else:
                    return np.array(Ds_jit(p))

        def _D2s_O(p, jax_out=False, batched=False):
            if batched:
                if jax_out:
                    return D2s_jit_batched(p)
                else:
                    return np.array(D2s_jit_batched(p))
            else:
                if jax_out:
                    return D2s_jit(p)
                else:
                    return np.array(D2s_jit(p))

        self._s_O = _s_O
        self._Ds_O = _Ds_O
        self._D2s_O = _D2s_O

    def _compute_functions_W(self) -> None:
        """Computes functions, Jacobians, and Hessians and compiles them in W frame."""
        if isinstance(self, MeshObject):
            return None

        if None in [self._s_O, self._Ds_O, self._D2s_O]:
            self._compute_functions_O()

        # rotation and translation from world frame to object frame
        R = self.X_WO.inverse().rotation().matrix()
        t = self.X_WO.inverse().translation()

        # computing s, Ds, D2s
        def _s_W(p, jax_out=False, batched=False):
            if batched:
                arg = (R @ p[..., None]).squeeze(-1) + t
            else:
                arg = R @ p + t
            return self._s_O(arg, jax_out=jax_out, batched=batched)

        def _Ds_W(p, jax_out=False, batched=False):
            if batched:
                arg = (R @ p[..., None]).squeeze(-1) + t
            else:
                arg = R @ p + t
            return self._Ds_O(arg, jax_out=jax_out, batched=batched) @ R

        def _D2s_W(p, jax_out=False, batched=False):
            if batched:
                arg = (R @ p[..., None]).squeeze(-1) + t
            else:
                arg = R @ p + t
            return R.T @ self._D2s_O(arg, jax_out=jax_out, batched=batched) @ R

        self._s_W = _s_W
        self._Ds_W = _Ds_W
        self._D2s_W = _D2s_W

    def set_X_WO(self, X_WO: RigidTransform) -> None:
        """Updates the pose of the object."""
        self.X_WO = RigidTransform(X_WO)
        self._compute_functions_W()

    def compute_mesh(self) -> None:
        """Computes a coarse mesh with the object's functional description.

        `shape_visual` will be a Mesh object representing the visual geometry in Drake.
        `shape_collision_list` will be a list of Convex objects representing a convex
        decomposition of the object used for fast collision detection.
        All geometries will be represented in the world frame.
        """
        lb = self.lb_O  # upper and lower bounds of the object
        ub = self.ub_O

        # if MeshObject, we already have a visual geometry
        if isinstance(self, MeshObject):
            _mesh_viz = self.mesh

        # use marching cubes to compute a mesh from the 0-level set of s approximately
        else:
            # computing grid
            npts = 31  # hardcoded gridding resolution
            X, Y, Z = np.mgrid[
                lb[0] : ub[0] : npts * 1j,
                lb[1] : ub[1] : npts * 1j,
                lb[2] : ub[2] : npts * 1j,
            ]

            pts_plot = np.stack((X, Y, Z), axis=-1)  # (npts, npts, npts, 3)
            pts_plot_flat = pts_plot.reshape((-1, 3))  # (B, 3)
            _vol = []
            for i in range(pts_plot_flat.shape[0]):
                p = pts_plot_flat[i, :]  # (3,)
                _vol.append(self.s_O(p))
            vol = np.stack(_vol).reshape(pts_plot.shape[:-1])  # (npts, npts, npts)
            _verts, faces, normals, _ = marching_cubes(vol, 0.0, allow_degenerate=False)
            verts = (ub - lb) * _verts / (np.array(X.shape) - 1) + lb
            _mesh_viz = trimesh.Trimesh(
                vertices=verts, faces=faces, vertex_normals=normals, process=False
            )

        # estimating the "scale" of the mesh triangles
        # use the mean of the unique edge lengths in the mesh
        self.mesh_scale = np.mean(_mesh_viz.edges_unique_length)

        # if the meshes aren't watertight, then mesh pre-processing should be fixed
        if self.enforce_watertight:
            assert _mesh_viz.is_watertight
        pth = "/tmp/mesh_viz.obj"  # export the visual mesh
        _mesh_viz.export(pth)
        mesh_viz = Mesh(pth)
        self.shape_visual = mesh_viz

        # we may already have true collision geometry. if not, generate it.
        self.shape_collision_list = []
        true_collision_geom_path = Path(ROOT + f"/data/{self.name}/collisions")
        if true_collision_geom_path.exists():
            mesh_col = _mesh_viz

            n_files = len(list(true_collision_geom_path.iterdir()))
            for i in range(n_files):
                pth = ROOT + f"/data/{self.name}/collisions/{self.name}_col_{i}.obj"
                mesh_col_i = Convex(pth)
                self.shape_collision_list.append(mesh_col_i)
        else:
            # use Poisson surface reconstruction to clean up mesh for collision geoms
            # [NOTE] this may not be watertight! But, the true mesh is watertight.
            mesh_col = poisson_reconstruction(_mesh_viz)

            # perform convex decomposition of visual meshes using VHACD
            # for flags, see https://github.com/mikedh/trimesh/blob/e62c526306766c4c597512ed7acc60c788e19e6f/trimesh/decomposition.py
            _cd_meshes = trimesh.Trimesh.convex_decomposition(
                mesh_col,
                maxConvexHulls=12,
                resolution=10000,
                minimumVolumePercentErrorAllowed=1.0,
                maxNumVerticesPerCH=16,
                maxRecursionDepth=5,
            )
            if not isinstance(_cd_meshes, list):  # rarely, vhacd returns only 1 body
                cd_meshes = [_cd_meshes]
            else:
                cd_meshes = _cd_meshes

            # create Mesh drake object for the visual geometry
            # mesh files are generated in /tmp
            pth = "/tmp/mesh_viz.obj"
            _mesh_viz.export(pth)
            mesh_viz = Mesh(pth)

            # create Convex drake objects for each of the collision geometries
            for i, m in enumerate(cd_meshes):
                pth = f"/tmp/mesh_col_{i}.obj"
                m.export(pth)
                mesh_col_i = Convex(pth)
                self.shape_collision_list.append(mesh_col_i)

        # use the clean mesh to compute the mass + inertia tensor of the object
        # density set to 150 kg/m^3, https://arxiv.org/pdf/2011.09584.pdf
        if self.mass is None:
            mesh_col.density = 150.0
            self.mass = mesh_col.mass
            self.inertia = mesh_col.moment_inertia
        else:
            vol = mesh_col.volume
            mesh_col.density = self.mass / vol
            self.inertia = mesh_col.mass_properties["inertia"]
        self.center_mass = _mesh_viz.center_mass

        # use the visual mesh to compute an oriented bounding box
        T, extents = trimesh.bounds.oriented_bounds(_mesh_viz)
        self.X_OBB = RigidTransform(T).inverse()
        self.lb_oriented = -extents / 2.0
        self.ub_oriented = extents / 2.0


@dataclass(kw_only=True)
class MeshObjectConfig(ObjectDescriptionConfig):
    """A configuration class for MeshObjects.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh object. Must be watertight.
    name : str | None, default="mesh"
        The name of the object.
    clean : bool, default=False
        Whether to clean the input mesh.
    """

    mesh: trimesh.Trimesh
    clean: bool = False

    def __post_init__(self):
        """Post initialization ops: overridden defaults + mesh bounds."""
        # overriding defaults
        if self.name is None:
            self.name = "mesh"

        # cleaning mesh
        if self.clean:
            # repair the mesh with poisson reconstruction
            print("Cleaning input mesh...")
            self.mesh = poisson_reconstruction(self.mesh)

        # compute object bounds from the mesh
        bounds = self.mesh.bounds  # (2, 3) bounds
        self.lb_O = bounds[0, :]
        self.ub_O = bounds[1, :]

    def create(self) -> "MeshObject":
        """Creates the MeshObject."""
        obj = MeshObject(self)
        return obj


class MeshObject(ObjectDescription):
    """An object represented entirely as a mesh.

    The signed distance function and its gradient is computed entirely with Open3D
    instead of Jax.
    """

    def __init__(self, cfg: MeshObjectConfig) -> None:
        """Initialize a mesh object. See ObjectDescription for parameter description."""
        self.mesh = cfg.mesh

        # MeshObject-specific cached values for efficient computation
        self.scene = o3d.t.geometry.RaycastingScene()  # scene for sdf computation
        self.scene.add_triangles(
            mesh=o3d.t.geometry.TriangleMesh.from_legacy(self.mesh.as_open3d)
        )

        self.p = None  # cached value of query point
        self.s_p = None  # cached values of SDF eval'd at p and its gradient
        self.Ds_p = None  # [NOTE] very coarse numerical estimate

        super().__init__(cfg)

    def _s_O_jax(self, p: jnp.ndarray) -> None:
        """Overwritten for API compliance."""
        return None

    def compute_all(self, p: np.ndarray, batched: bool) -> None:
        """Computes and caches the signed distance, its gradient, and Hessian.

        Parameters
        ----------
        p : np.ndarray, shape=(..., 3)
            Points in Cartesian space at which to query the signed distance.
        batched : bool
            Whether the points are batched or not. This is explicit just so the user is
            conscious about how they're calling the functions here.
        """
        # converting to open3d expected format
        if not batched:
            _p = p[None, :].astype(np.float32)  # (1, 3)
        else:
            _p = p.astype(np.float32)  # (B, 3)

        # (i) construct a tensor of perturbed points in ambient space
        #     P should have shape (B, n_perturbed, 3), where n_perturbed=3
        #     _p has shape (B, 3)
        B = _p.shape[0]
        p1s = _p
        _d2s = np.random.randn(B, 3)
        _d3s = np.random.randn(B, 3)
        d2s = _d2s / np.linalg.norm(_d2s, axis=-1, keepdims=True)  # (B, 3)
        d3s = _d3s / np.linalg.norm(_d3s, axis=-1, keepdims=True)
        eps = 10.0 * self.mesh_scale
        p2s = p1s + eps * d2s
        p3s = p1s + eps * d3s

        P = np.stack((p1s, p2s, p3s), axis=-2).astype(np.float32)  # (B, n_pert=3, 3)

        # (ii) compute s and Ds
        # compute_occupancy returns 0 if a point is outside and 1 if inside.
        # we convert that value to -1 if inside and 1 if outside.
        scene_query = self.scene.compute_closest_points(P)
        o_closest = scene_query["points"].numpy()
        occupancy = 1.0 - 2.0 * self.scene.compute_occupancy(P).numpy()

        # computing the signed distances
        t_op = P - o_closest  # vector from o_closest to p
        norm_t_op = np.linalg.norm(t_op, axis=-1)
        s_p_all = (occupancy * norm_t_op).squeeze().astype(np.float64)
        self.s_p = s_p_all[..., 0]  # only want signed distances of unperturbed points

        # computing gradients
        # when s_p = 0 (or is sufficiently close), use the triangle normals instead
        tri_normals = scene_query["primitive_normals"].numpy()  # (..., 3)
        ind_on_surface = np.abs(norm_t_op) <= 1e-6
        n_op = t_op / norm_t_op[..., None]  # normal from o_closest to p
        n_op[ind_on_surface] = tri_normals[ind_on_surface]
        occupancy[ind_on_surface] = 1.0  # if on surface, always point normal out
        Ds_p_all = (occupancy[..., None] * n_op).astype(np.float64)
        Ds_p = Ds_p_all[..., 0, :]  # only want gradients of unperturbed points
        self.Ds_p = Ds_p.squeeze()  # in the single sample case, squeeze the dim

        # (iii) estimating D2s numerically from perturbations
        N = Ds_p_all  # (B, n_pert, 3), normal vectors at perturbations
        Y = (N - N[..., 0:1, :]) / eps  # directional derivative estimates of normals
        D = np.stack((Ds_p, d2s, d3s), axis=-2)  # (B, n_pert, 3)

        # symmetrizing the result
        D2s_p = np.empty((B, 3, 3))
        for b in range(B):
            D2s_p[b] = self.compiled_full_solve(D[b], Y[b])
        self.D2s_p = D2s_p.squeeze()

    @staticmethod
    @numba_jit(nopython=True, fastmath=True, cache=True)
    def compiled_full_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Solves the 3x3 inverse problem and symmetrizes the result."""
        # solving, symmetrizing, and ensuring positive definiteness
        _X = np.linalg.solve(A, B)
        X = (_X + _X.T) / 2
        return X

    def s_O(
        self, p: np.ndarray, jax_out: bool = False, batched: bool = False
    ) -> np.ndarray:
        """Computes the value of s_O(p)."""
        assert not jax_out  # no Jax for MeshObject
        if batched:
            assert len(p.shape) == 2
            assert p.shape[-1] == 3
        else:
            assert p.shape == (3,)

        if self.s_p is None or np.any(p != self.p):
            self.compute_all(p, batched=batched)
            self.p = np.copy(p)
        return self.s_p

    def Ds_O(
        self, p: np.ndarray, jax_out: bool = False, batched: bool = False
    ) -> np.ndarray:
        """Computes the value of Ds_O(p)."""
        assert not jax_out  # no Jax for MeshObject
        if batched:
            assert len(p.shape) == 2
            assert p.shape[-1] == 3
        else:
            assert p.shape == (3,)

        if self.Ds_p is None or np.any(p != self.p):
            self.compute_all(p, batched=batched)
            self.p = np.copy(p)
        return self.Ds_p

    def D2s_O(
        self, p: np.ndarray, jax_out: bool = False, batched: bool = False
    ) -> np.ndarray:
        """Computes the value of D2s_O(p)."""
        assert not jax_out  # no Jax for MeshObject
        if batched:
            assert len(p.shape) == 2
            assert p.shape[-1] == 3
        else:
            assert p.shape == (3,)

        if self.D2s_p is None or np.any(p != self.p):
            self.compute_all(p, batched=batched)
            self.p = np.copy(p)
        return self.D2s_p

    def s_W(
        self, p: np.ndarray, jax_out: bool = False, batched: bool = False
    ) -> np.ndarray:
        """Computes the value of s_W(p)."""
        if batched:
            assert len(p.shape) == 2
            assert p.shape[-1] == 3
        else:
            assert p.shape == (3,)

        # rotation and translation from world frame to object frame
        R = self.X_WO.inverse().rotation().matrix()
        t = self.X_WO.inverse().translation()
        if batched:
            p_new = (R @ p[..., None]).squeeze(-1) + t
        else:
            p_new = R @ p + t

        if self.s_p is None or np.any(p != self.p):
            self.compute_all(p_new, batched=batched)
            self.p = np.copy(p_new)
        return self.s_p

    def Ds_W(
        self, p: np.ndarray, jax_out: bool = False, batched: bool = False
    ) -> np.ndarray:
        """Computes the value of Ds_W(p)."""
        if batched:
            assert len(p.shape) == 2
            assert p.shape[-1] == 3
        else:
            assert p.shape == (3,)

        # rotation and translation from world frame to object frame
        R = self.X_WO.inverse().rotation().matrix()
        t = self.X_WO.inverse().translation()
        if batched:
            p_new = (R @ p[..., None]).squeeze(-1) + t
        else:
            p_new = R @ p + t

        if self.Ds_p is None or np.any(p != self.p):
            self.compute_all(p_new, batched=batched)
            self.p = np.copy(p_new)
            self.Ds_p = self.Ds_p @ R
        return self.Ds_p

    def D2s_W(
        self, p: np.ndarray, jax_out: bool = False, batched: bool = False
    ) -> np.ndarray:
        """Computes the value of D2s_W(p)."""
        assert not jax_out  # no Jax for MeshObject
        if batched:
            assert len(p.shape) == 2
            assert p.shape[-1] == 3
        else:
            assert p.shape == (3,)

        # rotation and translation from world frame to object frame
        R = self.X_WO.inverse().rotation().matrix()
        t = self.X_WO.inverse().translation()
        if batched:
            p_new = (R @ p[..., None]).squeeze(-1) + t
        else:
            p_new = R @ p + t

        if self.D2s_p is None or np.any(p != self.p):
            self.compute_all(p_new, batched=batched)
            self.p = np.copy(p_new)
            self.D2s_p = R.T @ self.D2s_p @ R
        return self.D2s_p


@dataclass
class CustomObjectConfig(ObjectDescriptionConfig):
    """A configuration class for CustomObjects.

    Attributes
    ----------
    s_O_jax : Callable[[jnp.ndarray], jnp.ndarray]
        The SDF of the object in jax.
    """

    s_O_jax: Callable[[jnp.ndarray], jnp.ndarray]

    def __post_init__(self):
        """Post initialization ops: overridden defaults + mesh bounds."""
        # overriding defaults
        if self.name is None:
            self.name = "custom"

    def create(self) -> "CustomObject":
        """Creates the CustomObject."""
        obj = CustomObject(self)
        return obj


class CustomObject(ObjectDescription):
    """A custom object defined by a user-provided Jax function."""

    def __init__(self, cfg: CustomObjectConfig) -> None:
        """Initialize a custom object. See ObjectDescription."""
        self.s_O_jax = cfg.s_O_jax
        super().__init__(cfg)

    @partial(jit, static_argnums=(0,))
    def _s_O_jax(self, p: jnp.ndarray) -> jnp.ndarray:
        """The differentiable description of the custom object."""
        return self.s_O_jax(p)
