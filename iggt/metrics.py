
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple, Optional, Any
import logging
from scipy.spatial.transform import Rotation as R
import skimage.transform
import sys
import os

logger = logging.getLogger(__name__)


def calculate_iou(mask1, mask2):
    """计算两个布尔掩码的 IoU。"""
    intersection = np.sum(np.logical_and(mask1, mask2))
    union = np.sum(np.logical_or(mask1, mask2))
    return intersection / union if union > 0 else 0.0

def evaluate_matched_instances(gt_masks, pred_masks, iou_threshold=0.5):
    """
    先用匈牙利算法匹配，然后计算匹配实例的平均 IoU 和平均像素准确率。

    返回:
    - metrics (dict): 包含各项指标的字典。
    - matches (list): 匹配上的 (gt_index, pred_index) 对。
    """
    num_gt = len(gt_masks)
    num_pred = len(pred_masks)
    
    if num_gt == 0 or num_pred == 0:
        return {"matched_miou": 0, "matched_macc": 0, "num_matches": 0}, []

    # 1. 构建 IoU 矩阵
    iou_matrix = np.zeros((num_gt, num_pred))
    for i in range(num_gt):
        for j in range(num_pred):
            iou_matrix[i, j] = calculate_iou(gt_masks[i], pred_masks[j])

    # 2. 使用匈牙利算法找到最优匹配
    cost_matrix = 1 - iou_matrix
    gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

    # 3. 根据 IoU 阈值过滤匹配结果
    matches = []
    matched_ious = []
    matched_accs = []

    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        if iou_matrix[gt_idx, pred_idx] >= iou_threshold:
            matches.append((gt_idx, pred_idx))
            
            # 收集用于计算指标的数据
            matched_ious.append(iou_matrix[gt_idx, pred_idx])
            
            gt_mask = gt_masks[gt_idx]
            pred_mask = pred_masks[pred_idx]
            
            # 计算单个匹配对的像素准确率
            tp_pixels = np.sum(np.logical_and(gt_mask, pred_mask))
            gt_pixels = np.sum(gt_mask)
            acc_pair = tp_pixels / gt_pixels if gt_pixels > 0 else 0
            matched_accs.append(acc_pair)

    # 4. 计算平均指标
    if not matches:
        return {"matched_miou": 0, "matched_macc": 0, "num_matches": 0}, []

    matched_miou = np.mean(matched_ious)
    matched_macc = np.mean(matched_accs)
    
    metrics = {
        "matched_miou": matched_miou,
        "matched_macc": matched_macc,
        "num_matches": len(matches)
    }
    
    return metrics, matches

def valid_mean(arr, mask, axis=None, keepdims=np._NoValue):
    """Compute mean of elements across given dimensions of an array, considering only valid elements.

    Args:
        arr: The array to compute the mean.
        mask: Array with numerical or boolean values for element weights or validity. For bool, False means invalid.
        axis: Dimensions to reduce.
        keepdims: If true, retains reduced dimensions with length 1.

    Returns:
        Mean array/scalar and a valid array/scalar that indicates where the mean could be computed successfully.
    """

    mask = mask.astype(arr.dtype) if mask.dtype == bool else mask
    num_valid = np.sum(mask, axis=axis, keepdims=keepdims)
    masked_arr = arr * mask
    masked_arr_sum = np.sum(masked_arr, axis=axis, keepdims=keepdims)

    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mean = masked_arr_sum / num_valid
        is_valid = np.isfinite(valid_mean)
        valid_mean = np.nan_to_num(valid_mean, copy=False, nan=0, posinf=0, neginf=0)

    return valid_mean, is_valid


def thresh_inliers(gt, pred, thresh, mask=None, output_scaling_factor=1.0):
    """Computes the inlier (=error within a threshold) ratio for a predicted and ground truth depth map.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        thresh: Threshold for the relative difference between the prediction and ground truth.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]).

    Returns:
        Scalar that indicates the inlier ratio. Scalar is np.nan if the result is invalid.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_1 = np.nan_to_num(gt / pred, nan=thresh+1, posinf=thresh+1, neginf=thresh+1)  # pred=0 should be an outlier
        rel_2 = np.nan_to_num(pred / gt, nan=0, posinf=0, neginf=0)  # gt=0 is masked out anyways

    max_rel = np.maximum(rel_1, rel_2)
    inliers = ((0 < max_rel) & (max_rel < thresh)).astype(np.float32)  # 1 for inliers, 0 for outliers

    inlier_ratio, valid = valid_mean(inliers, mask)

    inlier_ratio = inlier_ratio * output_scaling_factor
    inlier_ratio = inlier_ratio if valid else np.nan

    return inlier_ratio


def m_rel_ae(gt, pred, mask=None, output_scaling_factor=1.0):
    """Computes the mean-relative-absolute-error for a predicted and ground truth depth map.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]).


    Returns:
        Scalar that indicates the mean-relative-absolute-error. Scalar is np.nan if the result is invalid.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    e = pred - gt
    ae = np.abs(e)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_ae = np.nan_to_num(ae / gt, nan=0, posinf=0, neginf=0)

    m_rel_ae, valid = valid_mean(rel_ae, mask)

    m_rel_ae = m_rel_ae * output_scaling_factor
    m_rel_ae = m_rel_ae if valid else np.nan

    return m_rel_ae


def pointwise_rel_ae(gt, pred, mask=None, output_scaling_factor=1.0):
    """Computes the pointwise relative-absolute-error for a predicted and ground truth depth map.

    Args:
        gt: Ground truth depth map as numpy array of shape 1HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape 1xHxW.
        mask: Array of shape 1xHxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        output_scaling_factor: Scaling factor that is applied after computing the metrics (e.g. to get [%]).

    Returns:
        Numpy array of shape 1xHxW with pointwise relative-absolute-error values.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    e = pred - gt
    ae = np.abs(e)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_ae = np.nan_to_num(ae / gt, nan=0, posinf=0, neginf=0)  # nan values are masked out anyways
    rel_ae *= mask

    rel_ae = rel_ae * output_scaling_factor

    return rel_ae


def sparsification(gt, pred, uncertainty, mask=None, error_fct=m_rel_ae, show_pbar=False, pbar_desc=None):
    """Computes the sparsification curve for a predicted and ground truth depth map and a given ranking.

    Args:
        gt: Ground truth depth map as numpy array of shape HxW. Negative or 0 values are invalid and ignored.
        pred: Predicted depth map as numpy array of shape HxW.
        uncertainty: Uncertainty measure for the predicted depth map. Numpy array of shape HxW.
        mask: Array of shape HxW with numerical or boolean values for element weights or validity.
            For bool, False means invalid.
        error_fct: Function that computes a metric between ground truth and prediction for the sparsification curve.
        show_pbar: Show progress bar.
        pbar_desc: Prefix for the progress bar.

    Returns:
        Pandas Series with (sparsification_ratio, error_ratio) values of the sparsification curve.
    """
    mask = (gt > 0).astype(np.float32) * mask if mask is not None else (gt > 0).astype(np.float32)

    y, x = np.unravel_index(np.argsort((uncertainty - uncertainty.min() + 1) * mask, axis=None), uncertainty.shape)
    # (masking out values that are anyways not considered for computing the error)
    ranking = np.flip(np.stack((x, y), axis=1), 0).tolist()

    num_valid = np.sum(mask.astype(bool))
    sparsification_steps = [int((num_valid / 100) * i) for i in range(100)]

    base_error = error_fct(gt=gt, pred=pred, mask=mask)
    sparsification_x, sparsification_y = [], []

    num_masked = 0
    pbar = tqdm(total=num_valid, desc=pbar_desc,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit}',
                disable=not show_pbar, unit="removed pixels", ncols=80)
    for x, y in ranking:
        if num_masked >= num_valid:
            break

        if mask[y, x] == 0:
            raise RuntimeError('This should never happen. If it happens, please open a GitHub issue.')

        if num_masked in sparsification_steps:
            cur_error = error_fct(gt=gt, pred=pred, mask=mask)
            sparsification_frac = num_masked / num_valid
            error_frac = cur_error / base_error
            if np.isfinite(cur_error):
                sparsification_x.append(sparsification_frac)
                sparsification_y.append(error_frac)

        mask[y, x] = 0
        num_masked += 1
        pbar.update(1)

    pbar.close()
    x = np.linspace(0, 0.99, 100)

    if len(sparsification_x) > 1:
        sparsification = np.interp(x, sparsification_x, sparsification_y)
    else:
        sparsification = np.array([np.nan] * 100, dtype=np.float64)
    sparsification = pd.Series(sparsification, index=x)

    return sparsification


class DepthEvaluator:
    """Depth estimation evaluation metrics."""

    def __init__(self,
                 alignment: str = "median",
                 clip_pred_depth: Optional[Tuple[float, float]] = (0.1, 100.0),
                 sparse_pred: bool = False):
        """
        Initialize depth evaluator.

        Args:
            alignment: Alignment method ("median", "least_squares", or None)
            clip_pred_depth: Depth clipping range (min, max) or None
            sparse_pred: Whether predictions are sparse
        """
        self.alignment = alignment
        self.clip_pred_depth = clip_pred_depth
        self.sparse_pred = sparse_pred

    def evaluate_depth(self, gt_depth: np.ndarray, pred_depth: np.ndarray) -> Dict[str, float]:
        """
        Evaluate depth prediction against ground truth.

        Args:
            gt_depth: Ground truth depth map (H, W) or (H, W, 1)
            pred_depth: Predicted depth map (H, W) or (H, W, 1)

        Returns:
            Dictionary containing evaluation metrics
        """
        # Ensure both depth maps are 2D
        if gt_depth.ndim == 3 and gt_depth.shape[-1] == 1:
            gt_depth = gt_depth.squeeze(-1)
        if pred_depth.ndim == 3 and pred_depth.shape[-1] == 1:
            pred_depth = pred_depth.squeeze(-1)

        # Resize prediction to match ground truth
        if gt_depth.shape != pred_depth.shape:
            pred_depth = skimage.transform.resize(
                pred_depth, gt_depth.shape, order=0, anti_aliasing=False
            )

        # Create masks
        pred_mask = pred_depth != 0 if self.sparse_pred else np.ones_like(pred_depth, dtype=bool)
        gt_mask = gt_depth > 0
        valid_mask = gt_mask & pred_mask

        if not valid_mask.any():
            logger.warning("No valid pixels for depth evaluation")
            return self._get_empty_depth_metrics()

        # Apply alignment
        aligned_pred_depth, scaling_factor = self._align_depth(gt_depth, pred_depth, valid_mask)

        # Apply clipping
        if self.clip_pred_depth is not None:
            aligned_pred_depth = np.clip(
                aligned_pred_depth,
                self.clip_pred_depth[0],
                self.clip_pred_depth[1]
            ) * pred_mask

        # Compute metrics
        metrics = self._compute_depth_metrics(gt_depth, aligned_pred_depth, valid_mask)
        metrics['scaling_factor'] = scaling_factor
        metrics['valid_pixels'] = np.sum(valid_mask)
        metrics['total_pixels'] = gt_depth.size
        metrics['valid_ratio'] = np.sum(valid_mask) / gt_depth.size

        return metrics

    def _align_depth(self, gt_depth: np.ndarray, pred_depth: np.ndarray,
                     mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply depth alignment."""
        if self.alignment == "median":
            gt_valid = gt_depth[mask]
            pred_valid = pred_depth[mask]

            if len(gt_valid) > 0 and len(pred_valid) > 0:
                ratio = np.median(gt_valid) / np.median(pred_valid)
                if np.isfinite(ratio):
                    return pred_depth * ratio, ratio

            logger.warning("Median alignment failed, using original prediction")
            return pred_depth, 1.0

        elif self.alignment == "least_squares":
            gt_valid = gt_depth[mask].flatten()
            pred_valid = pred_depth[mask].flatten()

            if len(gt_valid) > 0 and len(pred_valid) > 0:
                # Least squares: minimize ||gt - scale * pred||^2
                scale = np.sum(gt_valid * pred_valid) / np.sum(pred_valid ** 2)
                if np.isfinite(scale) and scale > 0:
                    return pred_depth * scale, scale

            logger.warning("Least squares alignment failed, using original prediction")
            return pred_depth, 1.0

        else:
            return pred_depth, 1.0

    def _compute_depth_metrics(self, gt_depth: np.ndarray, pred_depth: np.ndarray,
                              mask: np.ndarray) -> Dict[str, float]:
        """Compute depth evaluation metrics following the reference implementation."""
        if not mask.any():
            return self._get_empty_depth_metrics()

        # Create evaluation mask (same as reference code)
        eval_mask = pred_depth != 0 if self.sparse_pred else np.ones_like(pred_depth, dtype=bool)
        eval_mask = eval_mask & mask  # Combine with valid mask

        # Compute main metrics using vggt.metrics functions (same as reference)
        absrel = m_rel_ae(gt=gt_depth, pred=pred_depth, mask=eval_mask, output_scaling_factor=100.0)
        inliers103 = thresh_inliers(gt=gt_depth, pred=pred_depth, thresh=1.03, mask=eval_mask, output_scaling_factor=100.0)

        # Compute prediction depth density
        pred_depth_density = np.sum(eval_mask) / eval_mask.size * 100

        # Additional standard metrics for completeness
        gt_valid = gt_depth[eval_mask]
        pred_valid = pred_depth[eval_mask]

        if len(gt_valid) > 0:
            # Absolute errors
            abs_error = np.abs(gt_valid - pred_valid)
            mae = np.mean(abs_error)
            rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))

            # Threshold accuracies (delta metrics)
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_1 = np.maximum(gt_valid / pred_valid, pred_valid / gt_valid)
                ratio_1 = ratio_1[np.isfinite(ratio_1)]

            if len(ratio_1) > 0:
                delta_1 = np.mean(ratio_1 < 1.25) * 100  # Convert to percentage
                delta_2 = np.mean(ratio_1 < 1.25 ** 2) * 100
                delta_3 = np.mean(ratio_1 < 1.25 ** 3) * 100
            else:
                delta_1 = delta_2 = delta_3 = np.nan
        else:
            mae = rmse = delta_1 = delta_2 = delta_3 = np.nan

        return {
            'absrel': absrel,
            'inliers103': inliers103,
            'pred_depth_density': pred_depth_density,
            'mae': mae,
            'rmse': rmse,
            'delta_1': delta_1,
            'delta_2': delta_2,
            'delta_3': delta_3
        }

    def _get_empty_depth_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'absrel': np.nan,
            'inliers103': np.nan,
            'pred_depth_density': 0.0,
            'mae': np.nan,
            'rmse': np.nan,
            'delta_1': np.nan,
            'delta_2': np.nan,
            'delta_3': np.nan,
            'scaling_factor': np.nan,
            'valid_pixels': 0,
            'total_pixels': 0,
            'valid_ratio': 0.0
        }


class PoseEvaluator:
    """Camera pose evaluation metrics."""

    def __init__(self):
        """Initialize pose evaluator."""
        pass

    def evaluate_poses(self, gt_poses: np.ndarray, pred_poses: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate predicted poses against ground truth.

        Args:
            gt_poses: Ground truth poses (N, 3, 4) or (N, 4, 4)
            pred_poses: Predicted poses (N, 3, 4) or (N, 4, 4)

        Returns:
            Dictionary containing pose evaluation metrics
        """
        if gt_poses.shape != pred_poses.shape:
            logger.error(f"Pose shape mismatch: GT {gt_poses.shape}, Pred {pred_poses.shape}")
            return self._get_empty_pose_metrics()

        # Convert to 4x4 if necessary
        gt_poses_4x4 = self._to_4x4_poses(gt_poses)
        pred_poses_4x4 = self._to_4x4_poses(pred_poses)

        # Compute translation and rotation errors
        translation_errors = []
        rotation_errors = []

        for i in range(len(gt_poses_4x4)):
            gt_pose = gt_poses_4x4[i]
            pred_pose = pred_poses_4x4[i]

            # Translation error (Euclidean distance)
            gt_t = gt_pose[:3, 3]
            pred_t = pred_pose[:3, 3]
            t_error = np.linalg.norm(gt_t - pred_t)
            translation_errors.append(t_error)

            # Rotation error (angular distance)
            gt_R = gt_pose[:3, :3]
            pred_R = pred_pose[:3, :3]
            r_error = self._rotation_error(gt_R, pred_R)
            rotation_errors.append(r_error)

        translation_errors = np.array(translation_errors)
        rotation_errors = np.array(rotation_errors)

        # Compute statistics
        metrics = {
            'translation_error_mean': np.mean(translation_errors),
            'translation_error_median': np.median(translation_errors),
            'translation_error_std': np.std(translation_errors),
            'translation_error_max': np.max(translation_errors),
            'translation_error_min': np.min(translation_errors),
            'rotation_error_mean': np.mean(rotation_errors),
            'rotation_error_median': np.median(rotation_errors),
            'rotation_error_std': np.std(rotation_errors),
            'rotation_error_max': np.max(rotation_errors),
            'rotation_error_min': np.min(rotation_errors),
            'num_poses': len(gt_poses_4x4),
            'translation_errors': translation_errors,
            'rotation_errors': rotation_errors
        }

        return metrics

    def _to_4x4_poses(self, poses: np.ndarray) -> np.ndarray:
        """Convert poses to 4x4 format."""
        if poses.shape[-2:] == (4, 4):
            return poses
        elif poses.shape[-2:] == (3, 4):
            # Add bottom row [0, 0, 0, 1]
            N = poses.shape[0]
            poses_4x4 = np.zeros((N, 4, 4))
            poses_4x4[:, :3, :] = poses
            poses_4x4[:, 3, 3] = 1
            return poses_4x4
        else:
            raise ValueError(f"Unsupported pose shape: {poses.shape}")

    def _rotation_error(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """Compute rotation error in degrees."""
        try:
            # Compute relative rotation
            R_rel = R1.T @ R2

            # Convert to angle-axis representation
            r_rel = R.from_matrix(R_rel)
            angle = r_rel.magnitude()

            # Convert to degrees
            return np.degrees(angle)
        except Exception as e:
            logger.warning(f"Failed to compute rotation error: {e}")
            return np.nan

    def _get_empty_pose_metrics(self) -> Dict[str, Any]:
        """Return empty pose metrics dictionary."""
        return {
            'translation_error_mean': np.nan,
            'translation_error_median': np.nan,
            'translation_error_std': np.nan,
            'translation_error_max': np.nan,
            'translation_error_min': np.nan,
            'rotation_error_mean': np.nan,
            'rotation_error_median': np.nan,
            'rotation_error_std': np.nan,
            'rotation_error_max': np.nan,
            'rotation_error_min': np.nan,
            'num_poses': 0,
            'translation_errors': np.array([]),
            'rotation_errors': np.array([])
        }


class SceneEvaluator:
    """Combined scene evaluation including depth and pose metrics."""

    def __init__(self,
                 depth_alignment: str = "median",
                 depth_clip_range: Optional[Tuple[float, float]] = (0.1, 100.0)):
        """
        Initialize scene evaluator.

        Args:
            depth_alignment: Depth alignment method
            depth_clip_range: Depth clipping range
        """
        self.depth_evaluator = DepthEvaluator(
            alignment=depth_alignment,
            clip_pred_depth=depth_clip_range
        )
        self.pose_evaluator = PoseEvaluator()

    def evaluate_scene(self, gt_data: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate complete scene reconstruction.

        Args:
            gt_data: Ground truth data dictionary
            predictions: Model predictions dictionary

        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {
            'depth_metrics': {},
            'pose_metrics': {},
            'summary': {}
        }

        # Evaluate depth if available
        if 'gt_depth' in gt_data and 'depth' in predictions:
            logger.info("Evaluating depth predictions...")

            gt_depths = gt_data['gt_depth']  # (N, H, W)
            pred_depths = predictions['depth']  # (N, H, W)

            # Evaluate each frame
            frame_depth_metrics = []
            for i in range(len(gt_depths)):
                frame_metrics = self.depth_evaluator.evaluate_depth(
                    gt_depths[i], pred_depths[i]
                )
                frame_metrics['frame_id'] = i
                frame_depth_metrics.append(frame_metrics)

            # Aggregate metrics
            results['depth_metrics'] = self._aggregate_depth_metrics(frame_depth_metrics)
            results['depth_metrics']['per_frame'] = frame_depth_metrics

        # Evaluate poses if available
        if 'gt_extrinsic' in gt_data and 'extrinsic' in predictions:
            logger.info("Evaluating pose predictions...")

            gt_poses = gt_data['gt_extrinsic']  # (N, 3, 4)
            pred_poses = predictions['extrinsic']  # (N, 3, 4)

            results['pose_metrics'] = self.pose_evaluator.evaluate_poses(gt_poses, pred_poses)

        # Create summary
        results['summary'] = self._create_summary(results)

        return results

    def _aggregate_depth_metrics(self, frame_metrics: list) -> Dict[str, float]:
        """Aggregate depth metrics across frames."""
        if not frame_metrics:
            return {}

        # Get all metric keys (excluding non-numeric ones)
        numeric_keys = ['absrel', 'inliers103', 'pred_depth_density', 'mae', 'rmse',
                       'delta_1', 'delta_2', 'delta_3', 'valid_ratio']

        aggregated = {}

        for key in numeric_keys:
            values = [m[key] for m in frame_metrics if key in m and np.isfinite(m[key])]
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_median'] = np.median(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_min'] = np.min(values)
                aggregated[f'{key}_max'] = np.max(values)

        # Total valid pixels
        total_valid = sum(m['valid_pixels'] for m in frame_metrics)
        total_pixels = sum(m['total_pixels'] for m in frame_metrics)
        aggregated['total_valid_pixels'] = total_valid
        aggregated['total_pixels'] = total_pixels
        aggregated['overall_valid_ratio'] = total_valid / total_pixels if total_pixels > 0 else 0

        return aggregated

    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create evaluation summary."""
        summary = {}

        # Depth summary
        if 'depth_metrics' in results and results['depth_metrics']:
            depth_metrics = results['depth_metrics']
            summary['depth'] = {
                'absrel': depth_metrics.get('absrel_mean', np.nan),
                'inliers103': depth_metrics.get('inliers103_mean', np.nan),
                'pred_depth_density': depth_metrics.get('pred_depth_density_mean', np.nan),
                'mae': depth_metrics.get('mae_mean', np.nan),
                'rmse': depth_metrics.get('rmse_mean', np.nan),
                'delta_1': depth_metrics.get('delta_1_mean', np.nan),
                'valid_ratio': depth_metrics.get('overall_valid_ratio', 0)
            }

        # Pose summary
        if 'pose_metrics' in results and results['pose_metrics']:
            pose_metrics = results['pose_metrics']
            summary['pose'] = {
                'translation_error': pose_metrics.get('translation_error_mean', np.nan),
                'rotation_error': pose_metrics.get('rotation_error_mean', np.nan),
                'num_poses': pose_metrics.get('num_poses', 0)
            }

        return summary

    def save_evaluation_report(self, results: Dict[str, Any], save_path: str):
        """Save detailed evaluation report."""
        import json

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        serializable_results = convert_numpy(results)

        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Evaluation report saved to {save_path}")

    def print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "="*60)
        print("SCENE EVALUATION SUMMARY")
        print("="*60)

        summary = results.get('summary', {})

        # Depth metrics
        if 'depth' in summary:
            depth = summary['depth']
            print(f"\nDEPTH METRICS:")
            print(f"  AbsRel:     {depth['absrel']:.4f}%")
            print(f"  Inliers103: {depth['inliers103']:.4f}%")
            print(f"  Pred Density: {depth['pred_depth_density']:.4f}%")
            print(f"  MAE:        {depth['mae']:.4f}")
            print(f"  RMSE:       {depth['rmse']:.4f}")
            print(f"  δ < 1.25:   {depth['delta_1']:.4f}%")
            print(f"  Valid ratio: {depth['valid_ratio']:.4f}")

        # Pose metrics
        if 'pose' in summary:
            pose = summary['pose']
            print(f"\nPOSE METRICS:")
            print(f"  Translation error: {pose['translation_error']:.4f} m")
            print(f"  Rotation error:    {pose['rotation_error']:.4f} deg")
            print(f"  Number of poses:   {pose['num_poses']}")

        print("\n" + "="*60)
