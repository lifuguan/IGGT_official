# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
from scipy.spatial import cKDTree as KDTree

from iggt.utils.misc import invalid_to_zeros, invalid_to_nans
from iggt.utils.device import to_numpy

def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid

def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    # Invalid any depth > 80m
    valid_mask = valid_mask
    return X_cam, valid_mask


def depthmap_to_pts3d(depth, pseudo_focal, pp=None, **_):
    """
    Args:
        - depthmap (BxHxW array):
        - pseudo_focal: [B,H,W] ; [B,2,H,W] or [B,1,H,W]
    Returns:
        pointmap of absolute coordinates (BxHxWx3 array)
    """

    if len(depth.shape) == 4:
        B, H, W, n = depth.shape
    else:
        B, H, W = depth.shape
        n = None

    if len(pseudo_focal.shape) == 3:  # [B,H,W]
        pseudo_focalx = pseudo_focaly = pseudo_focal
    elif len(pseudo_focal.shape) == 4:  # [B,2,H,W] or [B,1,H,W]
        pseudo_focalx = pseudo_focal[:, 0]
        if pseudo_focal.shape[1] == 2:
            pseudo_focaly = pseudo_focal[:, 1]
        else:
            pseudo_focaly = pseudo_focalx
    else:
        raise NotImplementedError("Error, unknown input focal shape format.")

    assert pseudo_focalx.shape == depth.shape[:3]
    assert pseudo_focaly.shape == depth.shape[:3]
    grid_x, grid_y = xy_grid(W, H, cat_dim=0, device=depth.device)[:, None]

    # set principal point
    if pp is None:
        grid_x = grid_x - (W - 1) / 2
        grid_y = grid_y - (H - 1) / 2
    else:
        grid_x = grid_x.expand(B, -1, -1) - pp[:, 0, None, None]
        grid_y = grid_y.expand(B, -1, -1) - pp[:, 1, None, None]

    if n is None:
        pts3d = torch.empty((B, H, W, 3), device=depth.device)
        pts3d[..., 0] = depth * grid_x / pseudo_focalx
        pts3d[..., 1] = depth * grid_y / pseudo_focaly
        pts3d[..., 2] = depth
    else:
        pts3d = torch.empty((B, H, W, 3, n), device=depth.device)
        pts3d[..., 0, :] = depth * (grid_x / pseudo_focalx)[..., None]
        pts3d[..., 1, :] = depth * (grid_y / pseudo_focaly)[..., None]
        pts3d[..., 2, :] = depth
    return pts3d

def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, z_far=0, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)
    if z_far > 0:
        valid_mask = valid_mask & (depthmap < z_far)

    X_world = X_cam # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
) -> np.ndarray:
    """
    Unproject a batch of depth maps to 3D world coordinates.

    Args:
        depth_map (np.ndarray): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (np.ndarray): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (np.ndarray): Batch of camera intrinsic matrices of shape (S, 3, 3)

    Returns:
        np.ndarray: Batch of 3D world coordinates of shape (S, H, W, 3)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    z_far: float = 100.0,
    eps=1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a depth map to world coordinates (HxWx3) given the camera extrinsic and intrinsic.
    Returns both the world coordinates and the intermediate camera coordinates,
    as well as a mask for valid depth.

    Args:
        depth_map (np.ndarray):
            Depth map of shape (H, W).
        extrinsic (np.ndarray):
            Extrinsic matrix of shape (3, 4), representing the camera pose in OpenCV convention (camera-to-world).
        intrinsic (np.ndarray):
            Intrinsic matrix of shape (3, 3).
        eps (float):
            Small epsilon for thresholding valid depth.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (world_coords_points, cam_coords_points, point_mask)

            - world_coords_points: (H, W, 3) array of 3D points in world frame.
            - cam_coords_points: (H, W, 3) array of 3D points in camera frame.
            - point_mask: (H, W) boolean array where True indicates valid (non-zero) depth.
    """
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps
    if z_far > 0:
        point_mask = point_mask & (depth_map < z_far)

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)
    # The extrinsic is camera-from-world, so invert it to transform camera->world
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]
    # cam_to_world_extrinsic = extrinsic
    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # Apply the rotation and translation to the camera coordinates
    world_coords_points = (
        np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world
    ) # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a depth map to camera coordinates.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W).
        intrinsic (np.ndarray): Camera intrinsic matrix of shape (3, 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Camera coordinates (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def normalize_pointcloud(pts, norm_mode='avg_dis', valid=None, ret_factor=False):
    """ renorm pointmaps pts1, pts2 with norm_mode
    """
    assert pts.ndim >= 3 and pts.shape[-1] == 3
    norm_mode, dis_mode = norm_mode.split('_')

    if norm_mode == 'avg':
        # gather all points together (joint normalization)
        nan_pts, nnz = invalid_to_zeros(pts, valid, ndim=3)
        all_pts = nan_pts

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)
        if dis_mode == 'dis':
            pass  # do nothing
        elif dis_mode == 'log1p':
            all_dis = torch.log1p(all_dis)
        elif dis_mode == 'warp-log1p':
            # actually warp input points before normalizing them
            log_dis = torch.log1p(all_dis)
            warp_factor = log_dis / all_dis.clip(min=1e-8)
            H1, W1 = pts.shape[1:-1]
            pts = pts * warp_factor[:, :W1 * H1].view(-1, H1, W1, 1)
            all_dis = log_dis  # this is their true distance afterwards
        else:
            raise ValueError(f'bad {dis_mode=}')

        norm_factor = all_dis.sum(dim=1) / (nnz + 1e-8)
    else:
        # gather all points together (joint normalization)
        nan_pts = invalid_to_nans(pts, valid, ndim=3)
        all_pts = nan_pts

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)

        if norm_mode == 'avg':
            norm_factor = all_dis.nanmean(dim=1)
        elif norm_mode == 'median':
            norm_factor = all_dis.nanmedian(dim=1).values.detach()
        elif norm_mode == 'sqrt':
            norm_factor = all_dis.sqrt().nanmean(dim=1)**2
        else:
            raise ValueError(f'bad {norm_mode=}')

    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts.ndim:
        norm_factor.unsqueeze_(-1)

    res = pts / norm_factor
    if ret_factor:
        res = res + (norm_factor,)
    return res


@torch.no_grad()
def get_joint_pointcloud_depth(z1, z2, valid_mask1, valid_mask2=None, quantile=0.5):
    # set invalid points to NaN
    _z1 = invalid_to_nans(z1, valid_mask1).reshape(len(z1), -1)
    _z2 = invalid_to_nans(z2, valid_mask2).reshape(len(z2), -1) if z2 is not None else None
    _z = torch.cat((_z1, _z2), dim=-1) if z2 is not None else _z1

    # compute median depth overall (ignoring nans)
    if quantile == 0.5:
        shift_z = torch.nanmedian(_z, dim=-1).values
    else:
        shift_z = torch.nanquantile(_z, quantile, dim=-1)
    return shift_z  # (B,)


@torch.no_grad()
def get_joint_pointcloud_center_scale(pts1, pts2, valid_mask1=None, valid_mask2=None, z_only=False, center=True):
    # set invalid points to NaN
    _pts1 = invalid_to_nans(pts1, valid_mask1).reshape(len(pts1), -1, 3)
    _pts2 = invalid_to_nans(pts2, valid_mask2).reshape(len(pts2), -1, 3) if pts2 is not None else None
    _pts = torch.cat((_pts1, _pts2), dim=1) if pts2 is not None else _pts1

    # compute median center
    _center = torch.nanmedian(_pts, dim=1, keepdim=True).values  # (B,1,3)
    if z_only:
        _center[..., :2] = 0  # do not center X and Y

    # compute median norm
    _norm = ((_pts - _center) if center else _pts).norm(dim=-1)
    scale = torch.nanmedian(_norm, dim=1).values
    return _center[:, None, :, :], scale[:, None, None, None]


def find_reciprocal_matches(P1, P2):
    """
    returns 3 values:
    1 - reciprocal_in_P2: a boolean array of size P2.shape[0], a "True" value indicates a match
    2 - nn2_in_P1: a int array of size P2.shape[0], it contains the indexes of the closest points in P1
    3 - reciprocal_in_P2.sum(): the number of matches
    """
    tree1 = KDTree(P1)
    tree2 = KDTree(P2)

    _, nn1_in_P2 = tree2.query(P1, workers=8)
    _, nn2_in_P1 = tree1.query(P2, workers=8)

    reciprocal_in_P1 = (nn2_in_P1[nn1_in_P2] == np.arange(len(nn1_in_P2)))
    reciprocal_in_P2 = (nn1_in_P2[nn2_in_P1] == np.arange(len(nn2_in_P1)))
    assert reciprocal_in_P1.sum() == reciprocal_in_P2.sum()
    return reciprocal_in_P2, nn2_in_P1, reciprocal_in_P2.sum()


def get_med_dist_between_poses(poses):
    from scipy.spatial.distance import pdist
    return np.median(pdist([to_numpy(p[:3, 3]) for p in poses]))



def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')

def expand_extrinsic(extrinsic_3x4):
    """
    将 (B, N, 3, 4) 的外参矩阵扩展为 (B, N, 4, 4)。

    参数:
    extrinsic_3x4 (numpy.ndarray 或 torch.Tensor): 形状为 (B, N, 3, 4) 的外参矩阵。

    返回:
    numpy.ndarray 或 torch.Tensor: 形状为 (B, N, 4, 4) 的扩展外参矩阵。
    """
    if extrinsic_3x4.ndim != 4 or extrinsic_3x4.shape[-2:] != (3, 4):
        raise ValueError("输入的外参矩阵必须是 (B, N, 3, 4) 形状")

    B, N = extrinsic_3x4.shape[:2]  # 取 batch 和 N 维度

    # 创建一个 (B, N, 4, 4) 的单位矩阵
    if isinstance(extrinsic_3x4, np.ndarray):
        extrinsic_4x4 = np.tile(np.eye(4), (B, N, 1, 1))
    elif isinstance(extrinsic_3x4, torch.Tensor):
        extrinsic_4x4 = torch.eye(4, device=extrinsic_3x4.device).repeat(B, N, 1, 1)
    else:
        raise TypeError("输入必须是 numpy.ndarray 或 torch.Tensor")

    # 复制原始 3x4 矩阵的数据
    extrinsic_4x4[:, :, :3, :4] = extrinsic_3x4

    return extrinsic_4x4
