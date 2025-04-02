import os
import gc
from math import tan, radians, cos, sin, sqrt, pi
import sys

# Add the Conda environment (GIF-7005) to Blender's Python path
conda_python_path = "/Users/charlesbeaulieu/anaconda3/envs/GIF-7005/lib/python3.11/site-packages"
if conda_python_path not in sys.path:
    sys.path.insert(0, conda_python_path)

import bpy
import numpy as np
import open3d
from scipy.spatial.transform import Rotation as R
from mathutils import Vector


# -------------------------------------------------------------------
# Function: add_camera
# Description: Adds a new camera to the Blender scene.
# -------------------------------------------------------------------
def add_camera(
    camera_name,
    location=(0, 0, 0),
    rotation=(0, 0, 0),
    start_clip=0.1,
    end_clip=10,
    to_rad=True,
    display=False,
):
    """
    Add a new camera to the scene.
    """
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.object
    cam.name = camera_name
    # Convert rotation to radians if needed
    cam.rotation_euler = [np.radians(i) for i in rotation] if to_rad else rotation
    cam.data.clip_start = start_clip
    cam.data.clip_end = end_clip

    if display:
        print(f"Camera '{camera_name}' created at location {location} with rotation {rotation}")
        print(f"Clip start: {cam.data.clip_start}")
        print(f"Clip end: {cam.data.clip_end}")

    return cam


# -------------------------------------------------------------------
# Function: clear_scene
# Description: Deletes all objects in the current Blender scene and frees memory.
# -------------------------------------------------------------------
def clear_scene():
    """
    Delete all objects in the current scene and remove unused data blocks.
    """
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Remove all meshes and cameras to free memory.
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block, do_unlink=True)
    for block in bpy.data.cameras:
        bpy.data.cameras.remove(block, do_unlink=True)

    # Force garbage collection.
    gc.collect()


# -------------------------------------------------------------------
# Function: add_stl
# Description: Imports an STL model into the scene at a given location and scale.
# -------------------------------------------------------------------
def add_stl(stl_path, location=(0, 0, 0), scale=(0, 0, 0), display=False):
    """
    Import an STL model into the scene.
    """
    bpy.ops.wm.stl_import(filepath=stl_path)
    obj = bpy.context.object
    obj.name = "model"
    obj.scale = scale
    obj.location = location
    if display:
        print(
            f"STL model '{stl_path}' added to the scene at location {location} with scale {scale}"
        )
    return obj


# -------------------------------------------------------------------
# Function: get_raw_depth_from_camera
# Description: Renders the scene from a camera and returns the raw depth map as a numpy array.
#              The depth map is saved as an EXR file in a temporary folder.
# -------------------------------------------------------------------
def get_raw_depth_from_camera(cam, file_path):
    """
    Render the scene from the camera's perspective and return the raw depth map as a numpy array.
    The depth map is saved as an EXR file in a temporary folder.
    """
    os.makedirs("tmp", exist_ok=True)
    scene = bpy.context.scene
    scene.camera = cam

    # Enable the depth pass in the current view layer
    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True
    view_layer.update()  # Force scene update

    # Set up compositor nodes for rendering depth
    scene.use_nodes = True
    tree = scene.node_tree
    # Remove any existing nodes
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Create the Render Layers node
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.location = (0, 0)

    # Create the File Output node to save depth as an EXR file
    depth_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_output.location = (200, 0)
    depth_output.base_path = "tmp"
    depth_output.file_slots[0].path = file_path
    depth_output.format.file_format = "OPEN_EXR"
    depth_output.format.color_depth = "32"

    # Link the depth output from the Render Layers node to the File Output node
    tree.links.new(rl.outputs["Depth"], depth_output.inputs[0])

    # Render the scene and write the depth file
    bpy.ops.render.render(write_still=True)

    # Load the saved depth map EXR image
    depth_map = bpy.data.images.load(filepath=f"tmp/{file_path}0001.exr")
    depth_pixels = np.array(depth_map.pixels[:])
    # Reshape the flat pixel array into a 1080x1920 image with 4 channels, extract the depth channel
    depth_pixels = depth_pixels.reshape((1080, 1920, 4))[:, :, 0]

    return depth_pixels


# -------------------------------------------------------------------
# Function: get_intrinsic
# Description: Computes the camera's intrinsic matrix from the scene's render settings.
# -------------------------------------------------------------------
def get_intrinsic(camera, scene):
    """
    Compute the intrinsic matrix of the camera based on the scene's render settings.
    """
    resolution_x = scene.render.resolution_x
    resolution_y = scene.render.resolution_y
    resolution_percentage = scene.render.resolution_percentage / 100.0
    width = resolution_x * resolution_percentage
    height = resolution_y * resolution_percentage

    focal_length = camera.data.lens  # in millimeters
    sensor_width = camera.data.sensor_width  # in millimeters
    sensor_height = camera.data.sensor_height  # in millimeters

    # Convert focal length to pixel units
    fx = (focal_length / sensor_width) * width
    fy = (focal_length / sensor_height) * height
    cx = width / 2.0
    cy = height / 2.0

    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return intrinsic_matrix


# -------------------------------------------------------------------
# Function: get_extrinsic
# Description: Computes the extrinsic matrix of a Blender camera.
# -------------------------------------------------------------------
def get_extrinsic(camera):
    """
    Compute the extrinsic matrix of the Blender camera.
    """
    rotation = camera.rotation_euler
    # Compute the rotation matrix from Euler angles
    rotation_matrix = R.from_euler("xyz", rotation).as_matrix()

    # Build a 4x4 extrinsic matrix
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = camera.location

    # Flip the Z-axis to match Blender's coordinate system
    T_z = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    extrinsic = T_z @ extrinsic
    return extrinsic


# -------------------------------------------------------------------
# Function: depth_to_3d_world
# Description: Converts a depth image into 3D world coordinates using intrinsic and extrinsic matrices.
# -------------------------------------------------------------------
def depth_to_3d_world(depth_img, intrinsic, extrinsic):
    """
    Convert a depth image into 3D world coordinates.
    """
    h, w = depth_img.shape
    # Create a grid of pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.flatten()
    v = v.flatten()

    intrinsic_inv = np.linalg.inv(intrinsic)
    pixel_homogeneous = np.stack([u, v, np.ones_like(u)])
    # Convert pixel coordinates to camera coordinates, multiplied by the depth
    camera_coordinates = intrinsic_inv @ pixel_homogeneous
    camera_coordinates *= depth_img.flatten()

    # Convert from camera coordinates to world coordinates using the extrinsic matrix
    pts_camera = np.vstack((camera_coordinates, np.ones((1, camera_coordinates.shape[1]))))
    pts_world = np.linalg.inv(extrinsic) @ pts_camera
    world_coordinates = pts_world[:3, :].T

    return np.asarray(world_coordinates)


# -------------------------------------------------------------------
# Function: process_depth_maps
# Description: Processes a list of depth maps by zeroing out background pixels and adding noise.
# -------------------------------------------------------------------
def process_depth_maps(depth_maps, noise_lvl=0.01):
    """
    Process a list of depth maps:
      1. Set background (infinity depth) to 0.
      2. Add a small amount of noise to valid depth pixels before normalization.
    """
    processed_depth_maps = []

    for dm in depth_maps:
        dm_processed = np.copy(dm)
        # Identify background pixels (assumed to be the maximum depth) and set them to 0
        background_mask = dm_processed == np.max(dm_processed)
        dm_processed[background_mask] = 0

        # Add noise to non-background pixels
        noise = np.random.normal(
            0, noise_lvl * np.mean(dm_processed[dm_processed > 0]), dm_processed.shape
        )
        dm_processed[dm_processed > 0] += noise[dm_processed > 0]

        processed_depth_maps.append(dm_processed)

    return processed_depth_maps


# -------------------------------------------------------------------
# Function: get_camera_positions_on_sphere
# Description: Generates evenly distributed camera positions on a sphere using the Fibonacci sphere algorithm.
#              The positions are based on the object's bounding box and converted to meters.
# -------------------------------------------------------------------
def get_camera_positions_on_sphere(obj, num_cameras, distance):
    """
    Generates a list of camera positions evenly distributed on a sphere
    of radius `distance` around the object's bounding box center using the
    Fibonacci sphere algorithm.

    The Z-axis is used as the vertical direction (Blender's default), ensuring
    cameras cover the top, bottom, and sides. Final positions are converted to meters.

    Args:
        obj (bpy.types.Object): The target object.
        num_cameras (int): Number of desired camera positions.
        distance (float): The sphere radius (in scene units) from the object's center.

    Returns:
        list of mathutils.Vector: A list of camera locations in meters.
    """
    # Get object's bounding box corners in world space
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [v.x for v in bbox_corners]
    ys = [v.y for v in bbox_corners]
    zs = [v.z for v in bbox_corners]
    center = Vector(
        ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, (min(zs) + max(zs)) / 2.0)
    )

    points = []
    offset = 2.0 / num_cameras  # Uniformly space z-values in [-1,1]
    increment = pi * (3.0 - sqrt(5.0))  # Golden angle in radians

    for i in range(num_cameras):
        # Compute normalized z coordinate
        z = ((i * offset) - 1) + (offset / 2)
        # Compute radius of horizontal circle at this z
        r = sqrt(max(0.0, 1 - z * z))
        phi = i * increment
        x = cos(phi) * r
        y = sin(phi) * r

        # Calculate camera position in unit-sphere coordinates, scale and shift by object's center
        local_pos = Vector((x, y, z))
        scaled_pos = local_pos * distance
        pos = scaled_pos + center
        # Convert scene units (e.g., mm) to meters
        pos_m = pos / 1000.0
        points.append(pos_m)
    return points


# -------------------------------------------------------------------
# Function: generate_point_cloud
# Description: Generates a point cloud from an STL model by rendering from multiple camera views.
#              The best view is selected based on the number of valid depth pixels.
# -------------------------------------------------------------------
def generate_point_cloud(
    stl_path,
    model_location,
    model_scale,
    num_cameras,  # Number of cameras to place on the sphere
    distance_to_camera,
    output_folder,
):
    """
    Generate a point cloud from a given STL model.
    The model is rendered from multiple camera views automatically positioned on a sphere
    around the object. The view with the most valid depth pixels is used to generate the final point cloud.
    """
    start_clip = 0.1  # Near clipping plane (meters)
    end_clip = 10  # Far clipping plane (meters)
    # margin = 5  # Extra multiplier to frame the object
    part_number = stl_path.split("/")[-1].split(".")[0]
    print(f"Processing part {part_number} using STL file: {stl_path}")

    # Clear the current scene
    clear_scene()

    # Import the STL model into the scene.
    obj = add_stl(stl_path, model_location, model_scale)

    # --- Compute a suitable distance (sphere radius) ---
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    xs = [v.x for v in bbox_corners]
    ys = [v.y for v in bbox_corners]
    zs = [v.z for v in bbox_corners]
    # Compute center in scene units.
    center = Vector(
        ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0, (min(zs) + max(zs)) / 2.0)
    )
    # Convert center to meters.
    center_m = center / 1000.0

    dimensions = Vector((max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)))
    max_dimension = max(dimensions)
    if max_dimension == 0:
        print("Warning: Object bounding box is degenerate. Using default size.")
        max_dimension = 1.0

    # Create a temporary camera to get the field of view (FOV)
    bpy.ops.object.camera_add(location=(0, 0, 0))
    temp_cam = bpy.context.object
    temp_cam.data.clip_start = start_clip
    temp_cam.data.clip_end = end_clip
    fov = temp_cam.data.angle
    bpy.data.objects.remove(temp_cam, do_unlink=True)

    # Compute distance required to frame the object.
    # distance = (max_dimension / 2) / tan(fov / 2) * margin
    # print(f"Distance to object center: {distance} meters")
    distance = distance_to_camera  # Set a fixed distance for testing

    # Get camera positions on a sphere (in meters)
    cam_positions = get_camera_positions_on_sphere(obj, num_cameras, distance)
    cams = []
    depth_maps = []

    # For each computed camera position, add a camera and render its depth map.
    for idx, pos in enumerate(cam_positions):
        bpy.ops.object.camera_add(location=(0, 0, 0))
        cam = bpy.context.object
        cam.name = f"SphereCam_{idx}"
        cam.data.clip_start = start_clip
        cam.data.clip_end = end_clip
        # Set camera position (in meters)
        cam.location = pos

        # Calculate direction from camera to object's center (in meters)
        direction = center_m - cam.location
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cams.append(cam)

        # Render depth map from this camera
        dm = get_raw_depth_from_camera(cam, str(idx))
        depth_maps.append(dm)

    # Process the raw depth maps (remove background, add noise)
    processed_depth_maps = process_depth_maps(depth_maps, noise_lvl=0.001)

    # Select the best view based on the highest count of valid (nonzero) depth pixels.
    best_world_coordinates = None
    best_depth_map = None
    dm_stats = []

    # Count the non-zero pixels for each depth map
    for idx, (cam, dm) in enumerate(zip(cams, processed_depth_maps)):
        nonzero_count = np.count_nonzero(dm)
        dm_stats.append((cam, dm, nonzero_count, idx))

    # Sort the list by the number of non-zero pixels (descending order)
    dm_stats.sort(key=lambda x: x[2], reverse=True)

    # Retrieve best and median views
    best_view_info = dm_stats[0]  # View with most non-zero pixels
    median_view_info = dm_stats[len(dm_stats) // 2]  # Median view

    best_cam, best_depth_map, _, best_view_index = best_view_info
    median_cam, median_depth_map, _, median_view_index = median_view_info

    # Compute best view world coordinates
    extrinsic_best = get_extrinsic(best_cam)
    intrinsic_best = get_intrinsic(best_cam, bpy.context.scene)
    best_world_coordinates = depth_to_3d_world(best_depth_map, intrinsic_best, extrinsic_best)

    # Compute median view world coordinates
    extrinsic_median = get_extrinsic(median_cam)
    intrinsic_median = get_intrinsic(median_cam, bpy.context.scene)
    median_world_coordinates = depth_to_3d_world(
        median_depth_map, intrinsic_median, extrinsic_median
    )

    # If no valid depth data is found for the best view, save a debug Blender file and exit
    if best_world_coordinates is None or best_world_coordinates.shape[0] == 0:
        print(
            f"Warning: No valid depth data found for part {part_number}. Skipping point cloud generation."
        )
        os.makedirs(output_folder, exist_ok=True)
        blend_filename = os.path.join(output_folder, f"{part_number}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_filename)
        return

    # Create Open3D point clouds for both best and median views
    pcd_best = open3d.geometry.PointCloud()
    pcd_best.points = open3d.utility.Vector3dVector(best_world_coordinates)
    pcd_best = pcd_best.voxel_down_sample(voxel_size=0.001)  # Downsample
    pcd_best = pcd_best.translate(-pcd_best.get_center())  # Center the point cloud

    pcd_median = open3d.geometry.PointCloud()
    pcd_median.points = open3d.utility.Vector3dVector(median_world_coordinates)
    pcd_median = pcd_median.voxel_down_sample(voxel_size=0.001)  # Downsample
    pcd_median = pcd_median.translate(-pcd_median.get_center())  # Center the point cloud

    print(f"Generated best view point cloud with {len(pcd_best.points)} points")
    print(f"Generated median view point cloud with {len(pcd_median.points)} points")

    # Convert depth maps to 8-bit Open3D images
    dm_image_best = open3d.geometry.Image((best_depth_map * 255).astype(np.uint8))
    dm_image_median = open3d.geometry.Image((median_depth_map * 255).astype(np.uint8))

    # Ensure output directories exist
    pcd_best_folder = os.path.join(output_folder, "pcd_best")
    pcd_median_folder = os.path.join(output_folder, "pcd_median")
    dm_best_folder = os.path.join(output_folder, "depthmap_best")
    dm_median_folder = os.path.join(output_folder, "depthmap_median")
    blend_folder = os.path.join(output_folder, "blend")

    os.makedirs(pcd_best_folder, exist_ok=True)
    os.makedirs(pcd_median_folder, exist_ok=True)
    os.makedirs(dm_best_folder, exist_ok=True)
    os.makedirs(dm_median_folder, exist_ok=True)
    os.makedirs(blend_folder, exist_ok=True)

    # Define filenames
    pcd_filename_best = os.path.join(pcd_best_folder, f"{part_number}_{best_view_index}.ply")
    dm_filename_best = os.path.join(dm_best_folder, f"{part_number}_{best_view_index}.png")
    pcd_filename_median = os.path.join(pcd_median_folder, f"{part_number}_{median_view_index}.ply")
    dm_filename_median = os.path.join(dm_median_folder, f"{part_number}_{median_view_index}.png")
    blend_filename = os.path.join(blend_folder, f"{part_number}.blend")

    # Save best view files
    open3d.io.write_point_cloud(pcd_filename_best, pcd_best)
    open3d.io.write_image(dm_filename_best, dm_image_best)
    print(f"Saved best view point cloud: {pcd_filename_best}")
    print(f"Saved best view depth map: {dm_filename_best}")

    # Save median view files
    open3d.io.write_point_cloud(pcd_filename_median, pcd_median)
    open3d.io.write_image(dm_filename_median, dm_image_median)
    print(f"Saved median view point cloud: {pcd_filename_median}")
    print(f"Saved median view depth map: {dm_filename_median}")

    # Save Blender file
    bpy.ops.wm.save_as_mainfile(filepath=blend_filename)
    print(f"Saved Blender file: {blend_filename}")

    # Clear the scene and run garbage collection
    clear_scene()
    gc.collect()
