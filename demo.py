"""
IGGT Demo Script
================

This script demonstrates the usage of IGGT model
for 3D scene reconstruction and segmentation from multi-view images.

Features:
- Multi-view image processing
- Depth estimation and 3D point cloud generation
- Feature extraction and clustering
- 3D scene visualization export (GLB format)
"""

import os
import cv2
import torch
import numpy as np
import sys
import torch.nn.functional as F
import glob
import gc
import time
import logging
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add IGGT / VGGT module to path
sys.path.append("iggt/")

# IGGT imports
from visual_util import predictions_to_glb
from iggt.models.vggt import VGGT, IGGT
from iggt.utils.load_fn import load_and_preprocess_images
from iggt.utils.pose_enc import pose_encoding_to_extri_intri
from iggt.utils.geometry import unproject_depth_map_to_point_map, depth_to_world_coords_points, closed_form_inverse_se3
from iggt.utils.misc import (
    apply_pca_colormap,
    cluster_features_to_masks,
    cluster_features_to_masks_mv,
    knn_avg_features_pyg
)
from iggt.utils.arguments import load_opt_from_config_file
from iggt.datasets.utils.misc import threshold_depth_map
from iggt.utils.image import imread_cv2
from iggt.metrics import SceneEvaluator

# Utility imports
from utils.model import align_and_update_state_dicts

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_IMAGE_SIZE = (504, 336)
DEFAULT_CONF_THRESHOLD = 0.3

# Clustering parameters
#####  Small ######
# CLUSTERING_CONFIG = {
#     'eps': 0.005,
#     'min_samples': 50,
#     'min_cluster_size': 500,
#     'knn_k': 20
# }
#####  Medium ######
# CLUSTERING_CONFIG = {
#     'eps': 0.01,
#     'min_samples': 100,
#     'min_cluster_size': 500,
#     'knn_k': 20
# }
#####  Large ######
CLUSTERING_CONFIG = {
    'eps': 0.06,
    'min_samples': 100,
    'min_cluster_size': 500,
    'knn_k': 20
}

class IGGTProcessor:
    """
    IGGT model processor for 3D scene reconstruction and segmentation.
    """

    def __init__(self, model_path: str):
        """
        Initialize the IGGT processor.

        Args:
            model_path: Path to the model checkpoint
        """
        self.device = DEVICE
        self.model = self._load_model(model_path)
        self.evaluator = SceneEvaluator(depth_alignment="median", depth_clip_range=(0.1, 100.0))
        logger.info("IGGT model initialized successfully")

    def _load_model(self, model_path: str) -> IGGT:
        """Load and initialize the IGGT model."""
        logger.info("Loading IGGT model...")

        # Load SAM configuration
        model = IGGT()

        # Load model weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        state_dict = align_and_update_state_dicts(logger, model.state_dict(), state_dict)
        model.load_state_dict(state_dict, strict=False)

        model.eval()
        model = model.to(self.device)
        
        return model

    def process_scene(self, target_dir: str, save_dir: str) -> Dict[str, Any]:
        """
        Process a scene from multi-view images.

        Args:
            target_dir: Directory containing input images
            save_dir: Directory to save results
    
        Returns:
            Dictionary containing model predictions
        """
        logger.info(f"Processing scene from {target_dir}")
    
        # Validate input directory
        images_dir = os.path.join(target_dir, "images")
        if not os.path.exists(images_dir):
            raise ValueError(f"Images directory not found: {images_dir}")
    
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)

        # Load ground truth data if available
        gt_data = self._load_gt_data(target_dir)
        if gt_data is not None:
            self._save_gt_data(gt_data, save_dir)

        # Load and preprocess images
        predictions = self._run_inference(target_dir, save_dir)

        # Evaluate predictions against ground truth if available
        evaluation_results = None
        if gt_data is not None:
            logger.info("Evaluating predictions against ground truth...")
            evaluation_results = self.evaluator.evaluate_scene(gt_data, predictions)

            # Print evaluation summary
            self.evaluator.print_summary(evaluation_results)

            # Save evaluation report
            eval_report_path = os.path.join(save_dir, "evaluation_report.json")
            self.evaluator.save_evaluation_report(evaluation_results, eval_report_path)

        # Save predictions
        self._save_predictions(predictions, save_dir)

        # Combine results
        results = {
            'predictions': predictions,
            'evaluation': evaluation_results,
            'gt_data_available': gt_data is not None
        }

        logger.info("Scene processing completed successfully")
        return results

    def _run_inference(self, target_dir: str, save_dir: str) -> Dict[str, Any]:
        """Run model inference on input images."""
        # Load images
        image_paths = self._get_image_paths(target_dir)
        images = load_and_preprocess_images(
            image_paths,
            mode="resize",
            resize_target_size=DEFAULT_IMAGE_SIZE
        ).to(self.device)

        logger.info(f"Loaded {len(image_paths)} images with shape: {images.shape}")

        # Run inference
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = self.model(images)

        logger.info("Model inference completed")

        # Post-process predictions
        predictions = self._post_process_predictions(predictions, images, save_dir)

        return predictions

    def _get_image_paths(self, target_dir: str) -> List[str]:
        """Get sorted list of image paths."""
        images_dir = os.path.join(target_dir, "images")
        image_paths = glob.glob(os.path.join(images_dir, "*"))
        image_paths = sorted(image_paths)

        if len(image_paths) == 0:
            raise ValueError(f"No images found in {images_dir}")

        return image_paths

    def _load_gt_data(self, target_dir: str) -> Dict[str, Any]:
        """
        Load ground truth camera poses and depth maps from the scene directory.

        Args:
            target_dir: Directory containing the scene data

        Returns:
            Dictionary containing GT poses, intrinsics, and depth maps
        """
        logger.info("Loading ground truth camera poses and depth maps...")

        # Get paths
        # Define directory paths
        images_dir = os.path.join(target_dir, "images")
        depth_dir = os.path.join(target_dir, "depth")
        cam_dir = os.path.join(target_dir, "cam")

        # 检查目录是否存在，不存在则使用备选目录
        if not os.path.exists(depth_dir):
            depth_dir = os.path.join(target_dir, "gt_depth")
            logger.info(f"Using alternative depth directory: {depth_dir}")

        if not os.path.exists(cam_dir):
            cam_dir = os.path.join(target_dir, "gt_cam")
            logger.info(f"Using alternative camera directory: {cam_dir}")

        # Check if GT data exists
        if not os.path.exists(depth_dir) or not os.path.exists(cam_dir):
            logger.warning(f"GT data not found in {target_dir}. Depth dir: {os.path.exists(depth_dir)}, Cam dir: {os.path.exists(cam_dir)}")
            return None

        # Get sorted file lists
        image_paths = sorted(glob.glob(os.path.join(images_dir, "*")))
        depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
        cam_paths = sorted(glob.glob(os.path.join(cam_dir, "*.npz")))

        if len(image_paths) != len(depth_paths) or len(image_paths) != len(cam_paths):
            logger.warning(f"Mismatch in number of files: images={len(image_paths)}, depths={len(depth_paths)}, cams={len(cam_paths)}")
            return None

        # Load camera parameters and depth maps
        gt_extrinsics = []
        gt_intrinsics = []
        gt_depths = []
        gt_world_points = []

        for i, (img_path, depth_path, cam_path) in enumerate(zip(image_paths, depth_paths, cam_paths)):
            # Load camera parameters
            camera_info = np.load(cam_path)
            pose = np.array(camera_info['pose'], dtype=np.float32)  # world-to-camera
            intrinsics = np.array(camera_info['intrinsics'], dtype=np.float32)

            # Validate shapes
            assert pose.shape == (4, 4), f"Pose shape mismatch in {cam_path}: {pose.shape}"
            assert intrinsics.shape == (3, 3), f"Intrinsics shape mismatch in {cam_path}: {intrinsics.shape}"

            # Load and preprocess depth map (following scannet.py preprocessing)
            depthmap = imread_cv2(depth_path, cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000  # Convert to meters
            depthmap[~np.isfinite(depthmap)] = 0  # Set invalid depths to 0

            # Apply depth thresholding (same as scannet.py)
            depthmap = threshold_depth_map(depthmap, max_percentile=99, min_percentile=-1)

            # Convert pose to camera-to-world (same as scannet.py)
            camera_pose = closed_form_inverse_se3(pose[None])[0]

            # Compute world coordinates from depth map
            world_coords_points, cam_coords_points, point_mask = depth_to_world_coords_points(
                depthmap, camera_pose, intrinsics, z_far=100.0
            )

            gt_extrinsics.append(camera_pose[:3])  # Only take 3x4 part
            gt_intrinsics.append(intrinsics)
            gt_depths.append(depthmap)
            gt_world_points.append(world_coords_points)

        # Convert to numpy arrays
        gt_data = {
            'gt_extrinsic': np.stack(gt_extrinsics),  # (N, 3, 4)
            'gt_intrinsic': np.stack(gt_intrinsics),  # (N, 3, 3)
            'gt_depth': np.stack(gt_depths),          # (N, H, W)
            'gt_world_points': np.stack(gt_world_points),  # (N, H, W, 3)
            'image_paths': image_paths,
            'depth_paths': depth_paths,
            'cam_paths': cam_paths
        }

        logger.info(f"Loaded GT data for {len(image_paths)} frames")
        logger.info(f"GT extrinsic shape: {gt_data['gt_extrinsic'].shape}")
        logger.info(f"GT intrinsic shape: {gt_data['gt_intrinsic'].shape}")
        logger.info(f"GT depth shape: {gt_data['gt_depth'].shape}")

        return gt_data

    def _save_gt_data(self, gt_data: Dict[str, Any], save_dir: str):
        """Save ground truth data for comparison."""
        if gt_data is None:
            return

        gt_dir = os.path.join(save_dir, "ground_truth")
        os.makedirs(gt_dir, exist_ok=True)

        # Save GT depth visualizations
        self._save_depth_visualizations(gt_data['gt_depth'], gt_dir)

        # Save GT data arrays
        np.savez(
            os.path.join(gt_dir, "gt_data.npz"),
            gt_extrinsic=gt_data['gt_extrinsic'],
            gt_intrinsic=gt_data['gt_intrinsic'],
            gt_depth=gt_data['gt_depth'],
            gt_world_points=gt_data['gt_world_points']
        )

        logger.info(f"Ground truth data saved to {gt_dir}")

    def _post_process_predictions(self, predictions: Dict, images: torch.Tensor, save_dir: str) -> Dict[str, Any]:
        """Post-process model predictions."""
        # Extract pose encoding
        predictions["pose_enc"] = predictions["pose_enc"][-1]

        # Convert pose encoding to camera matrices
        logger.info("Converting pose encoding to camera matrices...")
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic

        # Convert tensors to numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    
        # Generate 3D world points
        logger.info("Computing 3D world points from depth maps...")
        depth_map = predictions["depth"]
        world_points = unproject_depth_map_to_point_map(
            depth_map, predictions["extrinsic"], predictions["intrinsic"]
        )
        predictions["world_points_from_depth"] = world_points

        # Process features and clustering
        predictions = self._process_features_and_clustering(predictions, world_points, save_dir)

        # Save depth visualizations
        self._save_depth_visualizations(depth_map, save_dir)

        return predictions

    def _process_features_and_clustering(self, predictions: Dict, world_points: np.ndarray, save_dir: str) -> Dict[str, Any]:
        """Process features and perform clustering."""
        # Extract and normalize part features
        part_feature = torch.from_numpy(predictions['part_feat']).permute(0, 2, 3, 1)
        part_feature = F.normalize(part_feature, dim=3)

        # Generate PCA visualization
        pred_spatial_pca_masks = apply_pca_colormap(part_feature)
        self._save_pca_masks(pred_spatial_pca_masks, save_dir, "colored_pca")

        # Apply spatial KNN smoothing
        spatial_knn_part_features = knn_avg_features_pyg(
            world_points, part_feature, k=CLUSTERING_CONFIG['knn_k']
        )
        # spatial_knn_part_features = F.normalize(spatial_knn_part_features, dim=3)
        pred_spatial_pca_masks_3d = apply_pca_colormap(spatial_knn_part_features)
        self._save_pca_masks(pred_spatial_pca_masks_3d, save_dir, "colored_pca_3d")

        # Perform DBSCAN clustering
        logger.info("Performing DBSCAN clustering...")
        dbscan_masks = cluster_features_to_masks_mv(
            spatial_knn_part_features,
            # part_feature,
            method="dbscan",
            eps=CLUSTERING_CONFIG['eps'],
            min_samples=CLUSTERING_CONFIG['min_samples'],
            min_cluster_size=CLUSTERING_CONFIG['min_cluster_size'],
            apply_colormap=True
        )

        self._save_dbscan_masks(dbscan_masks, save_dir)

        # Update predictions
        predictions['features'] = dbscan_masks[1]
        predictions['pca_features'] = pred_spatial_pca_masks.unsqueeze(0)

        return predictions

    def _save_pca_masks(self, pca_masks: torch.Tensor, save_dir: str, subdir: str):
        """Save PCA visualization masks."""
        output_dir = os.path.join(save_dir, subdir)
        os.makedirs(output_dir, exist_ok=True)

        N = pca_masks.shape[0]
        for i in range(N):
            color_img = Image.fromarray(
                (pca_masks[i] * 255).cpu().numpy().astype(np.uint8),
                mode="RGB"
            )
            color_img.save(os.path.join(output_dir, f"mask_colored_{i:03d}.png"))

        logger.info(f"PCA masks saved to {output_dir}")

    def _save_dbscan_masks(self, dbscan_masks: Tuple, save_dir: str):
        """Save DBSCAN clustering results."""
        masks, colored_masks = dbscan_masks
        output_dir = os.path.join(save_dir, "dbscan_masks")
        os.makedirs(output_dir, exist_ok=True)

        N = masks.shape[0]
        for i in range(N):
            # Save colored mask
            color_img = Image.fromarray(colored_masks[i].astype(np.uint8), mode="RGB")
            color_img.save(os.path.join(output_dir, f"mask_colored_{i:03d}.png"))

            # Save binary mask
            np.save(os.path.join(output_dir, f"mask_{i:03d}.npy"), masks[i])

        logger.info(f"DBSCAN masks saved to {output_dir}")

    def _save_depth_visualizations(self, depth_maps: np.ndarray, save_dir: str):
        """
        Save depth map visualizations with proper normalization and multiple visualization modes.

        Args:
            depth_maps: Array of depth maps with shape (N, H, W)
            save_dir: Directory to save visualizations
        """
        output_dir = os.path.join(save_dir, "pred_depths")
        os.makedirs(output_dir, exist_ok=True)

        # Compute depth statistics
        valid_depths = depth_maps[depth_maps > 0]  # Filter out invalid depths
        if len(valid_depths) == 0:
            logger.warning("No valid depth values found!")
            return

        depth_min = np.percentile(valid_depths, 1)   # Use percentiles to handle outliers
        depth_max = np.percentile(valid_depths, 99)
        depth_mean = np.mean(valid_depths)
        depth_std = np.std(valid_depths)

        logger.info(f"Depth statistics - Min: {depth_min:.3f}, Max: {depth_max:.3f}, "
                   f"Mean: {depth_mean:.3f}, Std: {depth_std:.3f}")

        # Save depth statistics
        stats = {
            'min': depth_min,
            'max': depth_max,
            'mean': depth_mean,
            'std': depth_std,
            'percentile_1': depth_min,
            'percentile_99': depth_max,
            'valid_pixel_ratio': len(valid_depths) / depth_maps.size
        }
        np.save(os.path.join(output_dir, 'depth_statistics.npy'), stats)

        # Visualization modes
        vis_modes = {
            'jet': cv2.COLORMAP_JET,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'plasma': cv2.COLORMAP_PLASMA,
            'turbo': cv2.COLORMAP_TURBO
        }

        images_dict = {mode: [] for mode in vis_modes.keys()}

        for i, depth_map in enumerate(depth_maps):
            # Handle invalid depths (set to min depth for visualization)
            depth_vis = depth_map.copy()
            depth_vis[depth_vis <= 0] = depth_min

            # Normalize depth to [0, 1] using robust statistics
            depth_normalized = np.clip((depth_vis - depth_min) / (depth_max - depth_min), 0, 1)

            # Convert to uint8 for visualization
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)

            # Save raw normalized depth
            np.save(os.path.join(output_dir, f'depth_normalized_{i:04d}.npy'), depth_normalized)

            # Generate visualizations with different colormaps
            for mode_name, colormap in vis_modes.items():
                # Apply colormap
                depth_colored = cv2.applyColorMap(depth_uint8, colormap)

                # Create mode-specific directory
                mode_dir = os.path.join(output_dir, mode_name)
                os.makedirs(mode_dir, exist_ok=True)

                # Save individual frame
                img_path = os.path.join(mode_dir, f'frame_{i:04d}.png')
                cv2.imwrite(img_path, depth_colored)
                images_dict[mode_name].append(Image.open(img_path))

                # Save with depth scale bar (for the first mode only)
                if mode_name == 'jet':
                    self._add_depth_scale_bar(depth_colored, depth_min, depth_max, img_path.replace('.png', '_with_scale.png'))

        # Create animated GIFs for each visualization mode
        for mode_name, images in images_dict.items():
            if images:
                gif_path = os.path.join(output_dir, f'depth_maps_{mode_name}.gif')
                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=200,  # Slower animation for better viewing
                    loop=0
                )

        # Create a comparison visualization (side by side different colormaps)
        self._create_comparison_visualization(depth_maps, depth_min, depth_max, output_dir)

        logger.info(f"Depth visualizations saved to {output_dir}")
        logger.info(f"Generated {len(vis_modes)} visualization modes: {list(vis_modes.keys())}")

    def _add_depth_scale_bar(self, depth_image: np.ndarray, depth_min: float, depth_max: float, save_path: str):
        """Add a depth scale bar to the visualization."""


        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Display the depth image
        ax.imshow(cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB))
        ax.axis('off')

        # Add scale bar
        scale_bar_height = 20
        scale_bar_width = 200
        scale_bar_x = depth_image.shape[1] - scale_bar_width - 20
        scale_bar_y = depth_image.shape[0] - scale_bar_height - 40

        # Create gradient for scale bar
        gradient = np.linspace(0, 1, scale_bar_width).reshape(1, -1)
        gradient = np.repeat(gradient, scale_bar_height, axis=0)

        # Apply same colormap as depth image
        scale_colored = cv2.applyColorMap((gradient * 255).astype(np.uint8), cv2.COLORMAP_JET)
        scale_colored = cv2.cvtColor(scale_colored, cv2.COLOR_BGR2RGB)

        # Add scale bar to image
        ax.imshow(scale_colored, extent=[scale_bar_x, scale_bar_x + scale_bar_width,
                                       scale_bar_y + scale_bar_height, scale_bar_y])

        # Add text labels
        ax.text(scale_bar_x, scale_bar_y - 5, f'{depth_min:.2f}m',
               color='white', fontsize=10, ha='left', weight='bold')
        ax.text(scale_bar_x + scale_bar_width, scale_bar_y - 5, f'{depth_max:.2f}m',
               color='white', fontsize=10, ha='right', weight='bold')
        ax.text(scale_bar_x + scale_bar_width//2, scale_bar_y - 5, 'Depth',
               color='white', fontsize=10, ha='center', weight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def _create_comparison_visualization(self, depth_maps: np.ndarray, depth_min: float, depth_max: float, output_dir: str):
        """Create a comparison visualization showing different colormaps side by side."""

        # Select a representative frame (middle frame)
        mid_idx = len(depth_maps) // 2
        depth_map = depth_maps[mid_idx]

        # Normalize depth
        depth_vis = depth_map.copy()
        depth_vis[depth_vis <= 0] = depth_min
        depth_normalized = np.clip((depth_vis - depth_min) / (depth_max - depth_min), 0, 1)

        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        colormaps = ['jet', 'viridis', 'plasma', 'turbo']
        cv_colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_PLASMA, cv2.COLORMAP_TURBO]

        for i, (cmap_name, cv_cmap) in enumerate(zip(colormaps, cv_colormaps)):
            depth_uint8 = (depth_normalized * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_uint8, cv_cmap)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

            axes[i].imshow(depth_colored)
            axes[i].set_title(f'{cmap_name.capitalize()} Colormap', fontsize=12, weight='bold')
            axes[i].axis('off')

        plt.suptitle(f'Depth Visualization Comparison (Frame {mid_idx})\n'
                    f'Depth Range: {depth_min:.2f}m - {depth_max:.2f}m',
                    fontsize=14, weight='bold')
        plt.tight_layout()

        comparison_path = os.path.join(output_dir, 'colormap_comparison.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Comparison visualization saved to {comparison_path}")

    def _save_predictions(self, predictions: Dict, save_dir: str):
        """Save predictions to file."""
        prediction_path = os.path.join(save_dir, "predictions.npz")
        np.savez(prediction_path, **predictions)
        logger.info(f"Predictions saved to {prediction_path}")


def export_glb_visualizations(predictions: Dict[str, Any], target_dir: str, save_dir: str,
                            frame_filter: str = "All", conf_thres: float = DEFAULT_CONF_THRESHOLD):
    """
    Export 3D visualizations in GLB format.

    Args:
        predictions: Model predictions dictionary
        target_dir: Input directory path
        save_dir: Output directory path
        frame_filter: Frame filtering option
        conf_thres: Confidence threshold for visualization
    """
    scene_name = target_dir.split('/')[-1]

    # Export different visualization modes
    vis_modes = [
        ("rgb", "color"),
        ("mask", "mask"),
        ("pca", "pca")
    ]

    for vis_mode, file_prefix in vis_modes:
        logger.info(f"Exporting {vis_mode} visualization...")

        glb_scene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=False,
            mask_white_bg=False,
            show_cam=True,
            mask_sky=False,
            target_dir=target_dir,
            prediction_mode="Pointmap Regression",
            vis_mode=vis_mode,
        )

        output_path = os.path.join(save_dir, f"{file_prefix}_glbscene_{scene_name}.glb")
        glb_scene.export(file_obj=output_path)
        logger.info(f"{vis_mode.upper()} GLB exported to {output_path}")


def main():
    """Main execution function."""
    start_time = time.time()

    # Configuration
    MODEL_PATH = "model path"

    # Input/Output paths
    TARGET_DIR = "./iggt_demo/demo9"
    SAVE_DIR = "./iggt_demo/demo9"
    try:
        # Initialize processor
        processor = IGGTProcessor(MODEL_PATH)

        # Process scene
        results = processor.process_scene(TARGET_DIR, SAVE_DIR)

        # Export GLB visualizations
        export_glb_visualizations(results['predictions'], TARGET_DIR, SAVE_DIR)

        # Cleanup
        del results
        gc.collect()
        torch.cuda.empty_cache()

        end_time = time.time()
        logger.info(f"Total processing time: {end_time - start_time:.2f} seconds")
        logger.info("Scene reconstruction completed successfully!")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
