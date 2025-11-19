#!/usr/bin/env python3
# --------------------------------------------------------
# Script to pre-process the aria-ase dataset
# Usage:
# 1. Prepare the codebase and environment for the projectaria_tools
# 2. copy this script to the project root directory
# 3. Run the script
# --------------------------------------------------------
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import os
from PIL import Image
from scipy.spatial.transform import Rotation as R
from projectaria_tools.projects import ase
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.image import InterpolationMethod
from projects.AriaSyntheticEnvironment.tutorial.code_snippets.readers import read_trajectory_file
import cv2
from tqdm import tqdm
import os, sys, json
import open3d as o3d
import random


def save_pointcloud(points_3d_array, rgb ,pcd_name):
    # Flatten the instance values array
    rgb_values_flat = rgb

    # Check if the number of points matches the number of instance values
    assert points_3d_array.shape[0] == rgb_values_flat.shape[0], "The number of points must match the number of instance values"

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Assign the 3D points to the point cloud object
    pcd.points = o3d.utility.Vector3dVector(points_3d_array)

    # Assign the colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(rgb_values_flat / 255.0)  # Normalize colors to [0, 1]

    # Define the file path where you want to save the point cloud
    output_file_path = pcd_name+'.pcd'

    # Save the point cloud in PCD format
    o3d.io.write_point_cloud(output_file_path, pcd)

    print(f"Point cloud saved to {output_file_path}")


def unproject(camera_params, undistorted_depth,undistorted_rgb):
    # Get the height and width of the depth image
    height, width = undistorted_depth.shape

    # Generate pixel coordinates
    y, x = np.indices((height, width))
    pixel_coords = np.stack((x, y), axis=-1).reshape(-1, 2)

    # Flatten the depth image to create a 1D array of depth values
    depth_values_flat = undistorted_depth.flatten()
    rgb_values_flat = undistorted_rgb.reshape(-1,3)

    # Initialize an array to store 3D points
    points_3d = []
    valid_rgb = []

    for pixel_coord, depth, rgb in zip(pixel_coords, depth_values_flat, rgb_values_flat):
        # Format the pixel coordinate for unproject (reshape to [2, 1])
        pixel_coord_reshaped = np.array([[pixel_coord[0]], [pixel_coord[1]]], dtype=np.float64)

        # Unproject the pixel to get the direction vector (ray)
        #   direction_vector = device.unproject(pixel_coord_reshaped)
        X = (pixel_coord_reshaped[0] - camera_params[2]) / camera_params[0] # X = (u - cx) / fx
        Y = (pixel_coord_reshaped[1] - camera_params[3]) / camera_params[1] # Y = (v - cy) / fy
        direction_vector = np.array([X[0], Y[0], 1],dtype=np.float32)
        if direction_vector is not None:
            # Replace the z-value of the direction vector with the depth value
            # Assuming the direction vector is normalized
            direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)
            point_3d = direction_vector_normalized * (depth / 1000)

            # Append the computed 3D point and the corresponding instance
            points_3d.append(point_3d.flatten())
            valid_rgb.append(rgb)

    # Convert the list of 3D points to a numpy array
    points_3d_array = np.array(points_3d)
    points_rgb = np.array(valid_rgb)
    return points_3d_array,points_rgb

def distance_to_depth(K, dist, uv=None):
    if uv is None and len(dist.shape) >= 2:
        # create mesh grid according to d
        uv = np.stack(np.meshgrid(np.arange(dist.shape[1]), np.arange(dist.shape[0])), -1)
        uv = uv.reshape(-1, 2)
        dist = dist.reshape(-1)
        if not isinstance(dist, np.ndarray):
            import torch
            uv = torch.from_numpy(uv).to(dist)
    if isinstance(dist, np.ndarray):
        # z * np.sqrt(x_temp**2+y_temp**2+z_temp**2) = dist
        uvh = np.concatenate([uv, np.ones((len(uv), 1))], -1)
        uvh = uvh.T # N, 3
        temp_point = np.linalg.inv(K) @ uvh # 3, N  
        temp_point = temp_point.T # N, 3
        z = dist / np.linalg.norm(temp_point, axis=1)
    else:
        uvh = torch.cat([uv, torch.ones(len(uv), 1).to(uv)], -1)
        temp_point = torch.inverse(K) @ uvh
        z = dist / torch.linalg.norm(temp_point, dim=1)
    return z

def transform_3d_points(transform, points):
    N = len(points)
    points_h = np.concatenate([points, np.ones((N, 1))], axis=1)
    transformed_points_h = (transform @ points_h.T).T
    transformed_points = transformed_points_h[:, :-1]
    return transformed_points


def aria_export_to_scannet(scene_id, seed):
    random.seed(int(seed + scene_id))
    src_folder = Path("ase_raw/"+str(scene_id))
    trgt_folder = Path("processed_ase/"+str(scene_id))
    trgt_folder.mkdir(parents=True, exist_ok=True)
    SCENE_ID = src_folder.stem
    print("SCENE_ID:", SCENE_ID)

    scene_max_depth = 0
    scene_min_depth = np.inf

    Path(trgt_folder, "segmentation").mkdir(exist_ok=True)

    instance_dir = src_folder / "instances"
    # Load camera calibration
    device = ase.get_ase_rgb_calibration()
    # Load the trajectory using read_trajectory_file() 
    trajectory_path = src_folder / "trajectory.csv"
    trajectory = read_trajectory_file(trajectory_path)
    all_points_3d = []
    all_rgb = []
    num_frames = len(list(instance_dir.glob("*.jpg")))
    # Path('./debug').mkdir(exist_ok=True)
    for frame_idx in tqdm(range(num_frames)):   
        frame_id = str(frame_idx).zfill(7)
        ins_path = instance_dir / f"instance{frame_id}.jpg"   
        ins = cv2.imread(str(ins_path), cv2.IMREAD_UNCHANGED)
        T_world_from_device = trajectory["Ts_world_from_device"][frame_idx] # camera-to-world
        assert device.get_image_size()[0] == 704
        # https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/advanced_code_snippets/image_utilities
        focal_length = device.get_focal_lengths()[0]
        pinhole = calibration.get_linear_camera_calibration(
            512,
            512,
            focal_length,
            "camera-rgb",
            device.get_transform_device_camera() # important to get correct transformation matrix in pinhole_cw90
            )
        # distort image
        rectified_ins = calibration.distort_by_calibration(np.array(ins), pinhole, device, InterpolationMethod.BILINEAR)
        
        rotated_ins = np.rot90(rectified_ins, k=3)

        cv2.imwrite(str(Path(trgt_folder, "segmentation", f"{frame_id}.jpg")), rotated_ins)

if __name__ == "__main__": 
    seed = 42   
    for scene_id in tqdm(range(0, 500)):
        aria_export_to_scannet(scene_id=scene_id, seed = seed)