from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh_fork as trimesh
from trimesh_fork.interfaces.vhacd import convex_decomposition

from frogger.sdfs.utils import to_trimesh

"""
Mesh processing script.

Out of the chosen objects, if the google_16k mesh is available, we use that, and
otherwise we use the Poisson reconstruction. Finally, if even that is unavailable, we
resort to the tsdf file (e.g., wine_glass). This is because sometimes the tsdf file
has significant errors (for example, on 001_chips_can, the entire bottom is missing).

Some objects are flat by nature (e.g. small silverware or tools or a box placed on its
main face). Instead of excluding these objects from the dataset due to the robots's
inability to pick them from a tabletop, we instead initialize their height at 5cm and
let the fingers go under without colliding with the tabletop.

Besides these objects, a small number of objects from the dataset are simply too small
for the robot to feasibly pick using all its fingers without a more specialized hand
e.g. nut and bolt). These items are excluded from the data.

Similarly, some objects are removed for being too large/heavy.

Some objects have distortion so severe that the mesh could not be accurately formed (e.g. windex bottle). These objects were excluded. Some objects have noticeable distortion, but are still included because the object is still recognizable (e.g. spoon, knife).

The objects "025_mug", "024_bowl", and "029_plate" have degenerate geometry (thin walls) that make it difficult to process them. The mug was replaced by a cup from the ShapeNetSem dataset.

---------------------------------------------------------------------------------------

We also generate "true" collision geometry by computing a high-quality convex decomposition of the mesh using VHACD. This is saved in the data directory so it doesn't need to be recomputed every time the object is initialized.

LIST OF EXCLUDED OBJECTS
------------------------
019_pitcher_base: big
022_windex_bottle: distorted object model
023_wine_glass: unable to generate model due to transparency
024_bowl: unable to generate geometry due to thin walls
025_mug: unable to generate geometry due to thin bottom
026_sponge: deformable, flat
028_skillet_lid: unable to generate model due to transparency
029_plate: unable to generate geometry due to thin walls
030_fork: too flat
031_spoon: too flat
032_knife: too flat
033_spatula: big
035_power_drill: big
037_scissors: too flat
038_padlock: distorted object model, small
039_key: no file, small
040_large_marker: too flat
041_small_marker: distorted object model
042_adjustable_wrench: too flat
046_plastic_bolt: no file, small
047_plastic_nut: no file, small
049_small_clamp: distorted object model
050_medium_clamp: too flat
053_mini_soccer_ball: big
059_chain: multibody
076_timer: features lost, remaining object not interesting
"""

recompute = True  # set this to true if you want to recompute from scratch

current_dir = Path(__file__)
project_dir = [p for p in current_dir.parents if p.parts[-1] == "manipulation"][0]
ROOT = str(project_dir)


def rmtree(f: Path) -> None:
    """Recursive function for deleting a directory.

    stackoverflow.com/questions/50186904
    """
    if f.is_file():
        f.unlink()
    else:
        for child in f.iterdir():
            rmtree(child)
        f.rmdir()


def remove_clusters(
    mesh: o3d.geometry.TriangleMesh,
    top_cluster: bool = False,
) -> o3d.geometry.TriangleMesh:
    """Removes spurious clusters of triangles from a mesh."""
    # computing clusters
    (
        triangle_clusters,
        cluster_n_triangles,
        cluster_area,
    ) = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    # removing small clusters of triangles/taking the top cluster
    mesh_modified = deepcopy(mesh)
    if top_cluster:
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh_modified.remove_triangles_by_mask(triangles_to_remove)
    else:
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
        mesh_modified.remove_triangles_by_mask(triangles_to_remove)
        _, _, cluster_area = mesh_modified.cluster_connected_triangles()
        assert np.asarray(cluster_area).shape[0] == 1
    return mesh_modified


# objects that are easy to generate watertight meshes for
easy_obj_names = [
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
    "048_hammer",
    "054_softball",
    "055_baseball",
    "056_tennis_ball",
    "057_racquetball",
    "058_golf_ball",
    "077_rubiks_cube",
]

# objects that are harder to generate watertight meshes for
hard_obj_names = [
    "043_phillips_screwdriver",
    "044_flat_screwdriver",
    "051_large_clamp",
    "052_extra_large_clamp",
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
    "sns_cup",  # 23fb2a2231263e261a9ac99425d3b306 / 0.00038748778493825193
]

# processing "easy" objects using a standardized pipeline
for obj_name in easy_obj_names:
    print(f"Processing {obj_name}...")
    obj_dir = ROOT + f"/data/{obj_name}"

    # checking for the mesh
    if not Path(obj_dir + f"/{obj_name}_clean.obj").exists() or recompute:
        print("  Generating mesh...")
        while True:
            # generating a point cloud from the dirty input mesh
            dirty_mesh = trimesh.load(obj_dir + f"/{obj_name}.obj")
            _mesh = dirty_mesh.as_open3d
            _mesh.compute_vertex_normals()
            pcd_master = _mesh.sample_points_uniformly(number_of_points=1000000)

            # method: from the master point cloud, take 12 different views of the
            # object to get the normal directions correct. Then, merge the point clouds
            # together and perform Poisson reconstruction on the merged cloud.
            # [NOTE] for some reason the racquetball has trouble here
            if obj_name == "057_racquetball":
                coeff = 2.0
            else:
                coeff = 1.0
            diameter = coeff * np.linalg.norm(
                np.asarray(pcd_master.get_max_bound())
                - np.asarray(pcd_master.get_min_bound())
            )
            lb = _mesh.get_axis_aligned_bounding_box().min_bound
            ub = _mesh.get_axis_aligned_bounding_box().max_bound
            max_val = max(np.max(ub), np.max(lb))
            min_val = min(np.min(ub), np.min(lb))
            r = max(np.abs(max_val), np.abs(min_val))
            center = _mesh.get_center()

            cam_locs = [
                center + np.array([-diameter, 0.0, 0.0]),
                center + np.array([diameter, 0.0, 0.0]),
                center + np.array([0.0, -diameter, 0.0]),
                center + np.array([0.0, diameter, 0.0]),
                center + np.array([0.0, 0.0, -diameter]),
                center + np.array([0.0, 0.0, diameter]),
                center + np.array([-diameter, -diameter, 0.0]),
                center + np.array([diameter, -diameter, 0.0]),
                center + np.array([0.0, -diameter, -diameter]),
                center + np.array([0.0, diameter, -diameter]),
                center + np.array([-diameter, 0.0, -diameter]),
                center + np.array([-diameter, 0.0, diameter]),
                center + np.array([-diameter, diameter, 0.0]),
                center + np.array([diameter, diameter, 0.0]),
                center + np.array([0.0, -diameter, diameter]),
                center + np.array([0.0, diameter, diameter]),
                center + np.array([diameter, 0.0, -diameter]),
                center + np.array([diameter, 0.0, diameter]),
            ]
            pcd_new = None
            for c_loc in cam_locs:
                scale = 10.0
                pt_map = pcd_master.hidden_point_removal(c_loc, diameter * scale)[1]
                pcd_view = pcd_master.select_by_index(pt_map)
                pcd_view = pcd_view.remove_statistical_outlier(20, 2.0)[0]
                pcd_view.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.01, max_nn=30
                    ),
                    fast_normal_computation=False,
                )
                pcd_view.orient_normals_towards_camera_location(camera_location=c_loc)

                if pcd_new is None:
                    pcd_new = pcd_view
                else:
                    pcd_new = pcd_new + pcd_view

            pcd = pcd_new.voxel_down_sample(voxel_size=0.001)
            _o3d_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, n_threads=1
            )

            # removing spurious clusters on each mesh + confirming there is only 1
            # connected component in the final geometry
            try:
                o3d_mesh = remove_clusters(_o3d_mesh)
            except AssertionError:
                print(
                    "    WARNING: More than 1 connected component! Regenerating mesh..."
                )
                continue

            # converting open3d mesh to trimesh mesh and checking watertightness
            poisson_mesh = to_trimesh(o3d_mesh)
            if poisson_mesh.is_watertight:
                poisson_mesh.export(obj_dir + f"/{obj_name}_clean.obj")
                break
            else:
                print("    WARNING: mesh was not watertight! Regenerating...")
    else:
        print("  Mesh already exists...")
        poisson_mesh = trimesh.load(obj_dir + f"/{obj_name}_clean.obj")

    # checking for collision geometry
    if not Path(obj_dir + f"/collisions/{obj_name}_col_0.obj").exists() or recompute:
        rmtree(Path(obj_dir + "/collisions"))
        Path(obj_dir + "/collisions").mkdir(parents=True, exist_ok=True)
        print("  Generating collision geometry...")

        # computing accurate convex collision geometry using VHACD
        cd_meshes = convex_decomposition(poisson_mesh, h=16, v=64)
        for i, m in enumerate(cd_meshes):
            pth = obj_dir + f"/collisions/{obj_name}_col_{i}.obj"
            m.export(pth)
    else:
        print("  Collision geometry already exists...")

# processing "hard" objects using custom camera locations
for obj_name in hard_obj_names:
    print(f"Processing {obj_name}...")
    obj_dir = ROOT + f"/data/{obj_name}"

    # checking for the mesh
    if not Path(obj_dir + f"/{obj_name}_clean.obj").exists() or recompute:
        print("  Generating mesh...")
        while True:
            dirty_mesh = trimesh.load(obj_dir + f"/{obj_name}.obj")

            # checking whether the object is from ShapeNetSem, applying scale
            # the mug is already watertight, no need to process it
            if obj_name == "sns_cup":
                scale = 0.00038748778493825193
                dirty_mesh = dirty_mesh.apply_scale(scale)

            _mesh = dirty_mesh.as_open3d
            _mesh.compute_vertex_normals()
            pcd_master = _mesh.sample_points_uniformly(number_of_points=1000000)
            diameter = np.linalg.norm(
                np.asarray(pcd_master.get_max_bound())
                - np.asarray(pcd_master.get_min_bound())
            )
            lb = _mesh.get_axis_aligned_bounding_box().min_bound
            ub = _mesh.get_axis_aligned_bounding_box().max_bound
            max_val = max(np.max(ub), np.max(lb))
            min_val = min(np.min(ub), np.min(lb))
            r = max(np.abs(max_val), np.abs(min_val))
            center = _mesh.get_center()

            # using custom settings for different objects
            if any(
                subs in obj_name
                for subs in ["065-a", "065-b", "065-c", "065-d", "065-e"]
            ):
                cam_locs = [
                    center,
                    center + np.array([-diameter, 0.0, 0.0]),
                    center + np.array([diameter, 0.0, 0.0]),
                    center + np.array([0.0, -diameter, 0.0]),
                    center + np.array([0.0, diameter, 0.0]),
                    center + np.array([0.0, 0.0, -diameter]),
                    center + np.array([0.0, 0.0, diameter]),
                    center + np.array([-diameter, -diameter, 0.0]),
                    center + np.array([diameter, -diameter, 0.0]),
                    center + np.array([0.0, -diameter, -diameter]),
                    center + np.array([0.0, diameter, -diameter]),
                    center + np.array([-diameter, 0.0, -diameter]),
                    center + np.array([-diameter, 0.0, diameter]),
                    center + np.array([-diameter, diameter, 0.0]),
                    center + np.array([diameter, diameter, 0.0]),
                    center + np.array([0.0, -diameter, diameter]),
                    center + np.array([0.0, diameter, diameter]),
                    center + np.array([diameter, 0.0, -diameter]),
                    center + np.array([diameter, 0.0, diameter]),
                ]
                scale = 10.0
                pcd_new = None
                for c_loc in cam_locs:
                    pt_map = pcd_master.hidden_point_removal(c_loc, diameter * scale)[1]
                    pcd_view = pcd_master.select_by_index(pt_map)
                    pcd_view.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.01, max_nn=30
                        ),
                        fast_normal_computation=False,
                    )
                    pcd_view.orient_normals_towards_camera_location(
                        camera_location=c_loc
                    )
                    if pcd_new is None:
                        pcd_new = pcd_view
                    else:
                        pcd_new = pcd_new + pcd_view
                top_cluster = False
            elif any(
                subs in obj_name
                for subs in ["065-f", "065-g", "065-h", "065-i", "065-j"]
            ):
                cam_locs = [
                    center,
                    center + np.array([-diameter, 0.0, 0.0]),
                    center + np.array([diameter, 0.0, 0.0]),
                    center + np.array([0.0, -diameter, 0.0]),
                    center + np.array([0.0, diameter, 0.0]),
                    center + np.array([0.0, 0.0, -diameter]),
                    center + np.array([0.0, 0.0, diameter]),
                    center + np.array([-diameter, -diameter, 0.0]),
                    center + np.array([diameter, -diameter, 0.0]),
                    center + np.array([0.0, -diameter, -diameter]),
                    center + np.array([0.0, diameter, -diameter]),
                    center + np.array([-diameter, 0.0, -diameter]),
                    center + np.array([-diameter, 0.0, diameter]),
                    center + np.array([-diameter, diameter, 0.0]),
                    center + np.array([diameter, diameter, 0.0]),
                    center + np.array([0.0, -diameter, diameter]),
                    center + np.array([0.0, diameter, diameter]),
                    center + np.array([diameter, 0.0, -diameter]),
                    center + np.array([diameter, 0.0, diameter]),
                ]
                scale = 100.0
                pcd_new = None
                for c_loc in cam_locs:
                    pt_map = pcd_master.hidden_point_removal(c_loc, diameter * scale)[1]
                    pcd_view = pcd_master.select_by_index(pt_map)
                    pcd_view = pcd_view.remove_radius_outlier(10, 0.001)[0]
                    pcd_view.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.01, max_nn=30
                        ),
                        fast_normal_computation=False,
                    )
                    pcd_view.orient_normals_towards_camera_location(
                        camera_location=c_loc
                    )
                    if pcd_new is None:
                        pcd_new = pcd_view
                    else:
                        pcd_new = pcd_new + pcd_view
                top_cluster = False
            elif any(
                subs in obj_name
                for subs in [
                    "030_fork",
                    "043_phillips_screwdriver",
                    "044_flat_screwdriver",
                ]
            ):
                cam_locs = [
                    center + np.array([-diameter, 0.0, 0.0]),
                    center + np.array([diameter, 0.0, 0.0]),
                    center + np.array([0.0, -diameter, 0.0]),
                    center + np.array([0.0, diameter, 0.0]),
                    center + np.array([0.0, 0.0, -diameter]),
                    center + np.array([0.0, 0.0, diameter]),
                    center + np.array([-diameter, -diameter, 0.0]),
                    center + np.array([diameter, -diameter, 0.0]),
                    center + np.array([0.0, -diameter, -diameter]),
                    center + np.array([0.0, diameter, -diameter]),
                    center + np.array([-diameter, 0.0, -diameter]),
                    center + np.array([-diameter, 0.0, diameter]),
                    center + np.array([-diameter, diameter, 0.0]),
                    center + np.array([diameter, diameter, 0.0]),
                    center + np.array([0.0, -diameter, diameter]),
                    center + np.array([0.0, diameter, diameter]),
                    center + np.array([diameter, 0.0, -diameter]),
                    center + np.array([diameter, 0.0, diameter]),
                ]
                scale = 100.0
                pcd_new = None
                for c_loc in cam_locs:
                    pt_map = pcd_master.hidden_point_removal(c_loc, diameter * scale)[1]
                    pcd_view = pcd_master.select_by_index(pt_map)
                    pcd_view = pcd_view.remove_radius_outlier(10, 0.001)[0]
                    pcd_view.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.01, max_nn=30
                        ),
                        fast_normal_computation=False,
                    )
                    if pcd_new is None:
                        pcd_new = pcd_view
                    else:
                        pcd_new = pcd_new + pcd_view
                top_cluster = False
            elif any(subs in obj_name for subs in ["031_spoon"]):
                cam_locs = [
                    center + np.array([-diameter, 0.0, 0.0]),
                    center + np.array([diameter, 0.0, 0.0]),
                    center + np.array([0.0, -diameter, 0.0]),
                    center + np.array([0.0, diameter, 0.0]),
                    center + np.array([0.0, 0.0, -diameter]),
                    center + np.array([0.0, 0.0, diameter]),
                    center + np.array([-diameter, -diameter, 0.0]),
                    center + np.array([diameter, -diameter, 0.0]),
                    center + np.array([-diameter, diameter, 0.0]),
                    center + np.array([diameter, diameter, 0.0]),
                    center + np.array([0.0, -diameter, -diameter]),
                    center + np.array([0.0, diameter, -diameter]),
                    center + np.array([-diameter, 0.0, diameter]),
                    center + np.array([0.0, -diameter, diameter]),
                    center + np.array([0.0, diameter, diameter]),
                    center + np.array([diameter, 0.0, -diameter]),
                    center + np.array([diameter, 0.0, diameter]),
                ]
                scale = 10.0
                pcd_new = None
                for c_loc in cam_locs:
                    pt_map = pcd_master.hidden_point_removal(c_loc, diameter * scale)[1]
                    pcd_view = pcd_master.select_by_index(pt_map)
                    pcd_view = pcd_view.remove_statistical_outlier(20, 2.0)[0]
                    pcd_view = pcd_view.remove_radius_outlier(20, 0.001)[0]
                    pcd_view.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.001, max_nn=30
                        ),
                        fast_normal_computation=False,
                    )
                    pcd_view.orient_normals_towards_camera_location(
                        camera_location=c_loc
                    )
                    if pcd_new is None:
                        pcd_new = pcd_view
                    else:
                        pcd_new = pcd_new + pcd_view
                top_cluster = False
            elif any(subs in obj_name for subs in ["clamp"]):
                if "050" in obj_name:
                    first_point = center + np.array([-0.1 * diameter, 0.0, 0.0])
                    top_cluster = False
                elif "051" in obj_name:
                    first_point = center + np.array([0.0, -0.1 * diameter, 0.0])
                    top_cluster = False
                elif "052" in obj_name:
                    first_point = center + np.array([-0.1 * diameter, 0.0, 0.0])
                    top_cluster = True

                cam_locs = [
                    first_point,
                    center + np.array([-diameter, 0.0, 0.0]),
                    center + np.array([diameter, 0.0, 0.0]),
                    center + np.array([0.0, -diameter, 0.0]),
                    center + np.array([0.0, diameter, 0.0]),
                    center + np.array([0.0, 0.0, -diameter]),
                    center + np.array([0.0, 0.0, diameter]),
                    center + np.array([-diameter, -diameter, 0.0]),
                    center + np.array([diameter, -diameter, 0.0]),
                    center + np.array([0.0, -diameter, -diameter]),
                    center + np.array([0.0, diameter, -diameter]),
                    center + np.array([-diameter, 0.0, -diameter]),
                    center + np.array([-diameter, 0.0, diameter]),
                    center + np.array([-diameter, diameter, 0.0]),
                    center + np.array([diameter, diameter, 0.0]),
                    center + np.array([0.0, -diameter, diameter]),
                    center + np.array([0.0, diameter, diameter]),
                    center + np.array([diameter, 0.0, -diameter]),
                    center + np.array([diameter, 0.0, diameter]),
                ]
                scale = 100.0
                pcd_new = None
                for c_loc in cam_locs:
                    pt_map = pcd_master.hidden_point_removal(c_loc, diameter * scale)[1]
                    pcd_view = pcd_master.select_by_index(pt_map)
                    pcd_view = pcd_view.remove_radius_outlier(10, 0.001)[0]
                    pcd_view.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.01, max_nn=30
                        ),
                        fast_normal_computation=False,
                    )
                    pcd_view.orient_normals_towards_camera_location(
                        camera_location=c_loc
                    )
                    if pcd_new is None:
                        pcd_new = pcd_view
                    else:
                        pcd_new = pcd_new + pcd_view
            elif obj_name == "061_foam_brick":
                cam_locs = [
                    center + np.array([0.0, 0.0, 0.3 * diameter]),
                    center + np.array([0.0, 0.2 * diameter, 0.3 * diameter]),
                    center + np.array([0.0, -0.2 * diameter, 0.3 * diameter]),
                    center + np.array([-diameter, 0.0, 0.0]),
                    center + np.array([diameter, 0.0, 0.0]),
                    center + np.array([0.0, -diameter, 0.0]),
                    center + np.array([0.0, diameter, 0.0]),
                    center + np.array([0.0, 0.0, -diameter]),
                    center + np.array([0.0, 0.0, diameter]),
                    center + np.array([-diameter, -diameter, 0.0]),
                    center + np.array([diameter, -diameter, 0.0]),
                    center + np.array([0.0, -diameter, -diameter]),
                    center + np.array([0.0, diameter, -diameter]),
                    center + np.array([-diameter, 0.0, -diameter]),
                    center + np.array([-diameter, 0.0, diameter]),
                    center + np.array([-diameter, diameter, 0.0]),
                    center + np.array([diameter, diameter, 0.0]),
                    center + np.array([0.0, -diameter, diameter]),
                    center + np.array([0.0, diameter, diameter]),
                    center + np.array([diameter, 0.0, -diameter]),
                    center + np.array([diameter, 0.0, diameter]),
                ]
                scale = 100.0
                pcd_new = None
                for c_loc in cam_locs:
                    pt_map = pcd_master.hidden_point_removal(c_loc, diameter * scale)[1]
                    pcd_view = pcd_master.select_by_index(pt_map)
                    pcd_view = pcd_view.remove_radius_outlier(10, 0.001)[0]
                    pcd_view.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.01, max_nn=30
                        ),
                        fast_normal_computation=False,
                    )
                    if pcd_new is None:
                        pcd_new = pcd_view
                    else:
                        pcd_new = pcd_new + pcd_view
                top_cluster = False
            elif any(subs in obj_name for subs in ["sns_cup"]):
                cam_locs = [
                    center + np.array([0.0, -0.235 * diameter, 0.0]),
                    center,
                    center + np.array([-0.5 * diameter, 0.0, 0.0]),
                    center + np.array([diameter, 0.0, 0.0]),
                    center + np.array([0.0, -diameter, 0.0]),
                    center + np.array([0.0, diameter, 0.0]),
                    center + np.array([0.0, 0.0, -diameter]),
                    center + np.array([0.0, 0.0, diameter]),
                ]
                first_time = True
                pcd_new = None
                for c_loc in cam_locs:
                    if first_time:
                        scale = 10.0
                        first_time = False
                    else:
                        scale = 100.0
                    pt_map = pcd_master.hidden_point_removal(c_loc, diameter * scale)[1]
                    pcd_view = pcd_master.select_by_index(pt_map)
                    pcd_view = pcd_view.remove_statistical_outlier(20, 1.0)[0]
                    pcd_view = pcd_view.remove_radius_outlier(10, 0.001)[0]
                    pcd_view.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.01, max_nn=30
                        ),
                        fast_normal_computation=False,
                    )
                    pcd_view.orient_normals_towards_camera_location(
                        camera_location=c_loc
                    )
                    if pcd_new is None:
                        pcd_new = pcd_view
                    else:
                        pcd_new = pcd_new + pcd_view
                top_cluster = False

            # downsampling and performing Poisson reconstruction
            pcd = pcd_new.voxel_down_sample(voxel_size=0.001)
            _o3d_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, n_threads=1
            )
            try:
                o3d_mesh = remove_clusters(_o3d_mesh, top_cluster=top_cluster)
            except AssertionError:
                print(
                    "    WARNING: More than 1 connected component! Regenerating mesh..."
                )
                continue

            # converting open3d mesh to trimesh mesh
            poisson_mesh = to_trimesh(o3d_mesh)
            if poisson_mesh.is_watertight:
                poisson_mesh.export(obj_dir + f"/{obj_name}_clean.obj")
                break
            else:
                print("    WARNING: mesh was not watertight! Regenerating...")
    else:
        print("  Mesh already exists...")
        poisson_mesh = trimesh.load(obj_dir + f"/{obj_name}_clean.obj")

    # checking for collision geometry
    if not Path(obj_dir + f"/collisions/{obj_name}_col_0.obj").exists() or recompute:
        rmtree(Path(obj_dir + "/collisions"))
        Path(obj_dir + "/collisions").mkdir(parents=True, exist_ok=True)
        print("  Generating collision geometry...")

        # computing accurate convex collision geometry using VHACD
        cd_meshes = convex_decomposition(poisson_mesh, h=16, v=64)
        for i, m in enumerate(cd_meshes):
            pth = obj_dir + f"/collisions/{obj_name}_col_{i}.obj"
            m.export(pth)
    else:
        print("  Collision geometry already exists...")

print("All objects processed!")
