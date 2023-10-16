import numpy as np
import open3d as o3d
import trimesh


def to_trimesh(o3d_mesh: o3d.TriangleMesh) -> trimesh.Trimesh:
    """Converts an open3d mesh to a trimesh.Trimesh mesh."""
    vertices = np.array(o3d_mesh.vertices)
    faces = np.array(o3d_mesh.triangles)
    normals = np.array(o3d_mesh.triangle_normals)
    mesh = trimesh.Trimesh(vertices, faces, normals)
    return mesh
