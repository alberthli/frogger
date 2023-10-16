import numpy as np
import open3d as o3d
from trimesh import Trimesh


def poisson_reconstruction(mesh: Trimesh) -> Trimesh:
    """Performs Poisson reconstruction of a Trimesh object using open3d.

    Parameters
    ----------
    mesh : Trimesh
        A trimesh mesh.

    Returns
    -------
    poisson_mesh : Trimesh
        The mesh after Poisson reconstruction is executed on it.
    """
    # upsampling points on the original mesh for high quality when generating mesh
    # [NOTE] for some reason, the sampling function does not take a random seed, so
    # results won't be completely reproducible, but they should be close
    _mesh = mesh.as_open3d
    _mesh.compute_vertex_normals()
    _pcd = _mesh.sample_points_uniformly(number_of_points=100000)
    pcd = _pcd.voxel_down_sample(voxel_size=0.001)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(10)
    o3d_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, n_threads=1
    )

    # converting open3d mesh to trimesh mesh
    vertices = np.array(o3d_mesh.vertices)
    faces = np.array(o3d_mesh.triangles)
    normals = np.array(o3d_mesh.triangle_normals)
    poisson_mesh = Trimesh(vertices, faces, normals)
    return poisson_mesh
