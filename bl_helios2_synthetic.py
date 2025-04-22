# Standard lib
import gc
import math
import os
from math import radians, tan
from pathlib import Path
import yaml

# Third-party
import bpy
import numpy as np
import open3d as o3d
from mathutils import Vector
from tqdm import tqdm

# Local imports
from camera_params import get_3x4_P_matrix_from_blender


def get_depthmap_from_helios2(
    camera_position,
    lookat_coords,
    scan_number,
    display=False,
    camera_name="Camera",
    output_dir="/",
    file_prefix="depth_",
):
    """
    Generate a depth map from a camera position and lookat coordinates.
    Similate a Helios2 ToF camera in Blender.

    Args:
        camera_position (tuple): Camera position in world coordinates.
        lookat_coords (tuple): Coordinates to look at.
        scan_number (int): Scan number for naming the depth map file.
        display (bool): Whether to display camera properties (for debug).
        camera_name (str): Name of the camera object in Blender (for data access).
        output_dir (str): Directory to save the depth map.
        file_prefix (str): Prefix for the depth map filename. ex : "depth_"

    Returns:
        tuple (ndarray, bpy.types.Object): The depth map as a NumPy array and the camera object.
    """
    #################################################
    # Camera and snene setup
    #################################################
    scene = bpy.context.scene

    # Render resolution
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100

    # Create a new camera
    bpy.ops.object.camera_add(location=camera_position)
    cam = bpy.context.object
    cam.name = camera_name
    cam.data.type = "PERSP"

    # Define camera properties
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = 6.4  # mm
    cam.data.sensor_height = 4.8  # mm

    # Define camera focal length
    fov_h = radians(69)
    focal_mm = (cam.data.sensor_width / 2) / tan(fov_h / 2)
    cam.data.lens = focal_mm

    # clipping in meters, can be ajusted if needed
    cam.data.clip_start = 0.1
    cam.data.clip_end = 3

    # Set camera rotation (look at 0,0,0)
    cam = bpy.data.objects[camera_name]
    target = Vector(lookat_coords)
    direction = (target - cam.location).normalized()
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam.rotation_euler = rot_quat.to_euler()

    # select the camera
    scene.camera = bpy.data.objects.get(camera_name)

    # print camera properties
    if display:
        print("========================================================")
        print("Camera properties (Helios2 ToF) : ")
        print(f"Camera position: {cam.location}")
        print(f"Camera rotation: {cam.rotation_euler}")
        print(f"Camera focal length: {cam.data.lens} mm")
        print(f"Camera sensor width: {cam.data.sensor_width} mm")
        print(f"Camera sensor height: {cam.data.sensor_height} mm")
        print(f"Camera clip start: {cam.data.clip_start} m")
        print(f"Camera clip end: {cam.data.clip_end} m")
        print(f"Camera FOV: {fov_h} rad")
        print(f"Focal length set to {focal_mm:.2f} mm")
        print("========================================================")

    #################################################
    # Render the depth map using the compositor.
    #################################################

    # Enable the Z (depth) pass on all view layers
    for vl in scene.view_layers:
        vl.use_pass_z = True

    # Set up the compositor nodes
    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    # Clear existing nodes
    nodes.clear()

    # Render Layers node
    rl = nodes.new(type="CompositorNodeRLayers")
    rl.location = (0, 0)

    # File Output node
    depth_out = nodes.new(type="CompositorNodeOutputFile")
    depth_out.location = (300, 0)
    depth_out.base_path = output_dir
    depth_out.file_slots[0].path = f"all_depth_maps/{scan_number}_{file_prefix}_{camera_name}"
    depth_out.format.file_format = "OPEN_EXR"
    depth_out.format.color_depth = "32"  # full precision

    # Link Depth → FileOutput
    links.new(rl.outputs["Depth"], depth_out.inputs[0])

    # Render and write the depth map
    scene.render.engine = "CYCLES"  # or 'BLENDER_EEVEE'
    bpy.ops.render.render(write_still=True)

    # Load the depth map from an EXR file
    exr_path = f"{output_dir}/all_depth_maps/{scan_number}_{file_prefix}_{camera_name}0001.exr"
    raw_exr = bpy.data.images.load(exr_path)

    # Convert pixels to a NumPy array and reshape to (height, width, 4)
    depth_map = np.array(raw_exr.pixels[:], dtype=np.float32).reshape(
        (scene.render.resolution_y, scene.render.resolution_x, 4)
    )
    # Extract the depth channel (first channel)
    depth_map = depth_map[:, :, 0]

    return (depth_map, cam)


def add_stl(stl_path, location=(0, 0, 0), scale=(0, 0, 0)):
    """
    Import an STL model into the scene.
    """
    # Import the model
    bpy.ops.wm.stl_import(filepath=stl_path)
    obj = bpy.context.object
    # Set the origin at the center of mass
    bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS", center="MEDIAN")

    # Set, name scale, and location
    obj.name = "stl_model"
    obj.scale = scale
    obj.location = location

    return obj


def clear_scene():
    """
    Clear the scene by removing all objects.
    """
    # Select and delete all objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Remove all leftover mesh, camera, light datablocks
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh, do_unlink=True)
    for cam in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam, do_unlink=True)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light, do_unlink=True)

    # 3. Force garbage collection
    gc.collect()


def fibonacci_sphere(samples=10, radius=1.0):
    """
    Generate points on a sphere using the Fibonacci sphere algorithm.
    """
    points = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(samples):
        # Y is distributed linearly between -1 and 1
        y = 1.0 - (2.0 * i) / float(samples - 1)
        r = math.sqrt(1.0 - y * y)  # circle radius at y

        theta = golden_angle * i

        x = math.cos(theta) * r
        z = math.sin(theta) * r

        # Scale the points to the desired radius
        points.append((radius * x, radius * y, radius * z))

    return points


def select_best_depth_map(depth_maps):
    """
    Filter the depth maps, we want to keep the best views, so we keep the one with the
    most non_null values. Also return the index of the best depth map.
    """
    max_non_null = 0
    max_dm = None
    best_cam = None

    for candidate_depth_map, cam in depth_maps:
        non_null = np.count_nonzero(candidate_depth_map)
        if non_null > max_non_null:
            max_non_null = non_null
            max_dm = candidate_depth_map
            best_cam = cam

    return max_dm, best_cam


def processed_depth_map(depth_maps):
    """
    Take the near 0 pixel and set them to 0.
    """
    for depth_map, _ in depth_maps:
        max_value = np.max(depth_map)
        depth_map[depth_map == max_value] = 0
    return depth_maps


def add_noise_to_dms(depth_maps, noise_level=0.001):
    """
    Add additive Gaussian noise to each non‐zero pixel in each depth map.
    """
    for depth_map, _ in depth_maps:
        mask = depth_map > 0
        # Generate noise only for the valid pixels
        noise = np.random.normal(loc=0.0, scale=noise_level, size=mask.sum())
        depth_map[mask] += noise

    return depth_maps


def depth_to_pointcloud(depth_map, camera):
    """
    Back‐project a depth map (in meters) into a world‐space point cloud.

    Args:
        depth_map (np.ndarray): H×W array of depths (meters).
        camera (bpy.types.Object): Blender camera object.

    Returns:
        open3d.geometry.PointCloud: the reconstructed point cloud.
    """
    # 1) Get intrinsics & extrinsics from Blender
    if camera is None:
        # if none, return empty point cloud
        print("No camera provided. Returning empty point cloud.")
        return o3d.geometry.PointCloud()

    _, K_bl, _ = get_3x4_P_matrix_from_blender(camera)
    K = np.array(K_bl)  # (3×3)

    # 2) Flatten pixel grid + depth
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    us = u.ravel()
    vs = v.ravel()
    zs = depth_map.ravel()

    # 3) Discard invalid (zero) depths
    mask = zs > 0
    us, vs, zs = us[mask], vs[mask], zs[mask]

    # 4) Build homogeneous pixel coords
    pix_h = np.vstack((us, vs, np.ones_like(us)))  # (3,N)

    # 5) Back‐project to camera space (CV convention)
    K_inv = np.linalg.inv(K)
    cam_pts = (K_inv @ pix_h) * zs  # (3,N)

    # 6) Convert from CV frame (X right, Y down, Z forward)
    #    to Blender camera frame (X right, Y up, Z backward)
    M_cv2bl = np.diag([1, -1, -1])  # flip Y and Z
    cam_pts_bl = M_cv2bl @ cam_pts  # (3,N)
    print(cam_pts_bl.T[:1])

    # 7) Camera → World via Blender's matrix_world
    R_wc = np.array(camera.matrix_world.to_3x3())  # (3×3)
    t_wc = np.array(camera.matrix_world.to_translation()).reshape(3, 1)  # (3×1)
    world_pts = (R_wc @ cam_pts_bl + t_wc).T  # (N×3)
    print(world_pts[:1])
    point_after_rotation = o3d.geometry.PointCloud()
    point_after_rotation.points = o3d.utility.Vector3dVector((R_wc @ cam_pts_bl).T)

    # 8) Build the Open3D point cloud and return it
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_pts)
    return pcd


def generate_synthetic_data(output_dir, stl_path, number_of_cameras=10, radius=1, noise=0.001):
    """
    Generate synthetic data as PLY format from STL files.

    Args:
        output_dir (str): the output directory for the generated data.
        stl_path (str): the path to the STL file to be processed.
    """
    number_of_cameras = CONFIG["number_of_cameras"]
    radius = CONFIG["radius"]
    noise = CONFIG["noise"]
    scan_num = stl_path.split("/")[-1].split(".")[0]

    # 1 CLEAR THE SCENE
    clear_scene()

    # 2 ADD THE STL (for the example, we use a single STL file)
    add_stl(
        stl_path,
        location=(0, 0, 0),  # Position the model at the origin.
        scale=(0.001, 0.001, 0.001),  # Scale the model appropriately.
    )

    # Generate camera positions
    # Increased samples will slow down the process
    # radius is the distance of the camera from the center of the sphere (aka the object)
    camera_locations = fibonacci_sphere(samples=number_of_cameras, radius=radius)

    camera_names = [f"Camera{i}_" for i in range(len(camera_locations))]
    dms = []  # list to store depth maps

    for pos, cam_name in zip(camera_locations, camera_names):
        # 3 ADD THE CAMERA, RENDER AND SAVE THE DEPTH MAP
        dm, cam_bpy_obj = get_depthmap_from_helios2(
            pos,
            lookat_coords=(0, 0, 0),
            scan_number=scan_num,
            camera_name=cam_name,
            output_dir=output_dir,
            file_prefix="depth",
        )
        dms.append((dm, cam_bpy_obj))

    # 4 PROCESS THE DEPTH MAPS
    processed_dms = processed_depth_map(dms)
    noised_dms = add_noise_to_dms(processed_dms, noise_level=noise)
    best_depth_map, best_camera = select_best_depth_map(noised_dms)

    # 5 BACK PROJECT THE DEPTH MAP TO A POINT CLOUD FROM BEST VIEW
    world_pts = depth_to_pointcloud(best_depth_map, best_camera)

    # save the blender scene
    # bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT_DIR, f"scene_save_{SCAN_NUM}.blend"))

    o3d.io.write_point_cloud(f"{output_dir}/{scan_num}.ply", world_pts)


def load_config(config_path):
    """
    Load the configuration file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def main():
    """
    Main function to generate synthetic data from STL files.
    """
    stl_folder = CONFIG["stl_directory"]
    out_dir = CONFIG["output_directory"]

    # Gather all STL files
    stl_files = [f for f in os.listdir(stl_folder) if f.lower().endswith(".stl")]

    # Ensure output directory exists
    output_folder = Path(out_dir).expanduser()
    output_folder.mkdir(parents=True, exist_ok=True)

    # Process with a progress bar
    for stl_file in tqdm(stl_files, desc="Generating synthetic data"):
        stl_path = os.path.join(stl_folder, stl_file)

        generate_synthetic_data(str(output_folder), str(Path(stl_path).expanduser()))

        # Clear everything for the next iteration
        bpy.ops.wm.quit_blender()  # should reset memory


if __name__ == "__main__":
    CONFIG = load_config("config1.yaml")
    main()
