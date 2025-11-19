# --------------------------------------------------------
# utilitary functions for DUSt3R
# --------------------------------------------------------
import torch
import cv2
import numpy as np
from iggt.utils.vo_eval import save_trajectory_tum_format
from PIL import Image
import matplotlib.cm as cm
from jaxtyping import Float
from torch import Tensor
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from typing import Union, Tuple
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_mean

try:
    from cuml.cluster.hdbscan import HDBSCAN
except:
    from hdbscan import HDBSCAN

def knn_avg_features_pyg(points_batch, features_batch, k, device='cpu'):
    """
    使用 PyTorch Geometric 对一批(batch)网格状数据进行k-NN特征平均。

    参数:
    points_batch (torch.Tensor or np.ndarray):
        点云坐标批次，形状为 (N, H, W, 3)。
    features_batch (torch.Tensor or np.ndarray):
        点云特征批次，形状为 (N, H, W, F)，F是特征维度。
    k (int):
        邻居数量。
    device (str):
        计算设备, e.g., 'cuda' or 'cpu'.

    返回:
    torch.Tensor: 平滑后的新特征批次，形状为 (N, H, W, F)。
    """
    # 1. 数据类型转换和设备转移
    # 确保输入是PyTorch Tensor并转移到指定设备
    if isinstance(points_batch, np.ndarray):
        points_batch = torch.from_numpy(points_batch).float()
    if isinstance(features_batch, np.ndarray):
        features_batch = torch.from_numpy(features_batch).float()

    points_batch = points_batch.to(device)
    features_batch = features_batch.to(device)

    # 2. 获取原始形状并重塑 (Reshape)
    N, H, W, F = features_batch.shape
    num_points_per_sample = H * W

    # 将 (N, H, W, 3) -> (N*H*W, 3)
    points_flat = points_batch.reshape(-1, 3)
    # 将 (N, H, W, F) -> (N*H*W, F)
    features_flat = features_batch.reshape(-1, F)

    # 3. 创建批次索引张量 (Batch Index Tensor) - 这是关键步骤
    # 这个张量告诉knn_graph每个点属于哪个样本
    # e.g., for N=2, H*W=3: [0, 0, 0, 1, 1, 1]
    batch_idx = torch.zeros(points_flat.shape[0], dtype=torch.long, device=device)

    # 4. 使用knn_graph找到每个点的k个近邻
    # `batch=batch_idx`确保近邻搜索只在各自的样本内部进行
    edge_index = knn_graph(points_flat, k=k, batch=batch_idx, loop=False)

    source_nodes, center_nodes = edge_index

    # 5. 提取邻居特征并使用scatter_mean进行高效平均
    neighbor_features = features_flat[source_nodes]
    smoothed_features_flat = scatter_mean(neighbor_features, center_nodes, dim=0, dim_size=N * H * W)

    # 6. 将结果重塑回原始的批次形状
    smoothed_features_batch = smoothed_features_flat.view(N, H, W, F)

    return smoothed_features_batch


def cluster_features_to_masks_mv(
    feature_map: Union[torch.Tensor, np.ndarray],
    apply_colormap: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    对多视图特征图进行聚类，并选择性地生成彩色可视化掩码。

    现在支持将所有视图的特征合并后一起做DBSCAN聚类，这样同类别的物体在不同视图中会分配到相同的类别ID和颜色。

    Args:
        feature_map (Union[torch.Tensor, np.ndarray]):
            输入的特征图，形状应为 (N, H, W, C)。
        apply_colormap (bool, optional):
            如果为True，函数将额外返回一个用于可视化的彩色掩码图。默认为False。
        **kwargs:
            传递给聚类算法的参数。
            - 若 method='kmeans', 则需要提供 n_clusters (例如: n_clusters=5)。
            - 若 method='dbscan', 则需要提供 eps 和 min_samples (例如: eps=0.5, min_samples=50)。

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - 如果 apply_colormap=False (默认):
              返回一个 np.ndarray，形状为 (N, H, W)，其中的整数值代表类别ID。
            - 如果 apply_colormap=True:
              返回一个元组 (masks, colored_masks)，其中：
                - masks: (N, H, W) 的整数类别ID掩码。
                - colored_masks: (N, H, W, 3) 的彩色RGB掩码，`uint8` 类型 (0-255)。
    """
    if not (isinstance(feature_map, (torch.Tensor, np.ndarray)) and feature_map.ndim == 4):
        raise ValueError("输入特征图必须是形状为 (N, H, W, C) 的4D Tensor或NumPy数组。")

    n, h, w, c = feature_map.shape

    if isinstance(feature_map, torch.Tensor):
        feature_map_np = feature_map.cpu().numpy()
    else:
        feature_map_np = feature_map

    # 合并所有视图的特征做DBSCAN
    print(f"对所有视图特征合并后做DBSCAN聚类 (N={n}, H={h}, W={w}, C={c}) ...")
    all_pixels = feature_map_np.reshape(-1, c)  # (N*H*W, C)
    hdbscan_cluster = HDBSCAN(
        cluster_selection_epsilon=kwargs.get("eps"),
        min_samples=kwargs.get("min_samples"),
        min_cluster_size=kwargs.get("min_cluster_size"),
        allow_single_cluster=False,
    ).fit(all_pixels)
    all_labels = hdbscan_cluster.labels_  # (N*H*W,)
    invalid_label_mask = all_labels == -1

    # 利用KNN对无效标签(-1)的像素进行掩码分配
    if invalid_label_mask.sum() > 0:
        if invalid_label_mask.sum() == len(invalid_label_mask):
            # 全部为无效点，全部设为0
            all_labels = np.zeros_like(all_labels)
        else:
            valid_pixels = all_pixels[~invalid_label_mask]
            invalid_pixels = all_pixels[invalid_label_mask]
            valid_labels = all_labels[~invalid_label_mask]
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(valid_pixels)
            distances, indices = nbrs.kneighbors(invalid_pixels)
            indices = indices[:, 0]
            all_labels[invalid_label_mask] = valid_labels[indices]

    # 还原为 (N, H, W)
    masks = all_labels.reshape(n, h, w)
    print("DBSCAN聚类完成。")

    if not apply_colormap:
        return masks
    else:
        print("正在应用颜色图...")
        # 全局分配颜色，保证同一label在所有视图中颜色一致
        unique_labels = np.unique(masks)
        unique_labels_no_noise = unique_labels[unique_labels != -1]
        n_colors = len(unique_labels_no_noise)
        cmap = plt.colormaps.get_cmap('jet')
        color_map = {
            label: list(cmap(j / (n_colors - 1))[:3]) if n_colors > 1 else list(cmap(0.5)[:3])
            for j, label in enumerate(unique_labels_no_noise)
        }
        color_map[-1] = [0, 0, 0]
        colored_masks = np.zeros((n, h, w, 3), dtype=np.uint8)
        for i in range(n):
            labels = masks[i].reshape(-1)
            colored_pixels = np.array([color_map[label] for label in labels])
            colored_pixels_uint8 = (colored_pixels * 255).astype(np.uint8)
            colored_masks[i] = colored_pixels_uint8.reshape(h, w, 3)
        return masks, colored_masks
    


def cluster_features_to_masks(
    feature_map: Union[torch.Tensor, np.ndarray],
    method: str = "kmeans",
    apply_colormap: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    对多视图特征图进行聚类，并选择性地生成彩色可视化掩码。

    如果 method='dbscan'，则对每张图单独做DBSCAN聚类。
    如果 method='kmeans'，则对所有视图像素合并后整体聚类。

    Args:
        feature_map (Union[torch.Tensor, np.ndarray]):
            输入的特征图，形状应为 (N, H, W, C)。
        method (str, optional):
            聚类方法，可选 "kmeans" 或 "dbscan"。默认为 "kmeans"。
        apply_colormap (bool, optional):
            如果为True，函数将额外返回一个用于可视化的彩色掩码图。默认为False。
        **kwargs:
            传递给聚类算法的参数。
            - 若 method='kmeans', 则需要提供 n_clusters (例如: n_clusters=5)。
            - 若 method='dbscan', 则需要提供 eps 和 min_samples (例如: eps=0.5, min_samples=50)。

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            - 如果 apply_colormap=False (默认):
              返回一个 np.ndarray，形状为 (N, H, W)，其中的整数值代表类别ID。
            - 如果 apply_colormap=True:
              返回一个元组 (masks, colored_masks)，其中：
                - masks: (N, H, W) 的整数类别ID掩码。
                - colored_masks: (N, H, W, 3) 的彩色RGB掩码，`uint8` 类型 (0-255)。
    """
    if not (isinstance(feature_map, (torch.Tensor, np.ndarray)) and feature_map.ndim == 4):
        raise ValueError("输入特征图必须是形状为 (N, H, W, C) 的4D Tensor或NumPy数组。")

    n, h, w, c = feature_map.shape

    if isinstance(feature_map, torch.Tensor):
        feature_map_np = feature_map.cpu().numpy()
    else:
        feature_map_np = feature_map

    # 改为每张图单独做dbscan
    print(f"对每张图单独做DBSCAN聚类 (N={n}, H={h}, W={w}, C={c}) ...")
    masks = np.zeros((n, h, w), dtype=np.int32)
    for i in range(n):
        pixels = feature_map_np[i].reshape(-1, c)
        dbascan_cluster = HDBSCAN(
            cluster_selection_epsilon=kwargs.get("eps"),
            min_samples=kwargs.get("min_samples"),
            min_cluster_size=kwargs.get("min_cluster_size"),
            allow_single_cluster=False,
        ).fit(pixels)
        labels = dbascan_cluster.labels_
        invalid_label_mask = labels == -1

        # 利用KNN对无效标签(-1)的像素进行掩码分配
        # 输入 pixels, shape 为 (H, W, C)
        if invalid_label_mask.sum() > 0:
            if invalid_label_mask.sum() == len(invalid_label_mask):
                # 全部为无效点，全部设为0
                labels = np.zeros_like(labels)
            else:
                # 直接用 pixels 作为特征，pixels shape: (H*W, C)
                pixels_flat = pixels  # pixels 已经是 (H*W, C)
                valid_pixels = pixels_flat[~invalid_label_mask]
                invalid_pixels = pixels_flat[invalid_label_mask]
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(valid_pixels)
                distances, indices = nbrs.kneighbors(invalid_pixels)
                indices = indices[:, 0]
                labels[invalid_label_mask] = labels[~invalid_label_mask][indices]

        masks[i] = labels.reshape(h, w)
        print("DBSCAN聚类完成。")

    if not apply_colormap:
        return masks
    else:
        print("正在应用颜色图...")
        colored_masks = np.zeros((n, h, w, 3), dtype=np.uint8)
        for i in range(n):
            labels = masks[i].reshape(-1)
            unique_labels = np.unique(labels)
            unique_labels_no_noise = unique_labels[unique_labels != -1]
            n_colors = len(unique_labels_no_noise)
            cmap = plt.colormaps.get_cmap('jet')
            color_map = {
                label: list(cmap(j / (n_colors - 1))[:3]) if n_colors > 1 else list(cmap(0.5)[:3])
                for j, label in enumerate(unique_labels_no_noise)
            }
            color_map[-1] = [0, 0, 0]
            colored_pixels = np.array([color_map[label] for label in labels])
            colored_pixels_uint8 = (colored_pixels * 255).astype(np.uint8)
            colored_masks[i] = colored_pixels_uint8.reshape(h, w, 3)
        return masks, colored_masks
    

def apply_pca_colormap(image: Float[Tensor, "N H W C"]):
    """通过PCA将一批特征图像转换为3通道RGB图像。

    该函数处理形状为 (N, H, W, C) 的4D输入，其中N是
    视图的数量。为了确保多视图的一致性，它会展平所有视图
    以计算单个PCA变换，然后将输出重塑
    以保留N个独立的视图。

    此版本使用百分位归一化来增强对比度，使颜色映射更加鲜明。

    Args:
        image: 形状为 (N, H, W, C) 的特征图像4D张量。

    Returns:
        Tensor: 形状为 (N, H, W, 3) 的彩色图像4D张量。
    """
    # 1. 存储原始的多视图维度，并展平以便进行一致的PCA
    n, h, w, c = image.shape
    # 从 (N, H, W, C) 重塑为 (N*H*W, C)。
    # 这会将所有视图中的所有像素连接成一个批次。
    image_flat = image.reshape(-1, c)

    # 2. 对所有视图的组合数据计算PCA
    # 主成分 (v) 将代表整个数据集。为了结果更稳定，可以适当增加q的值。
    _, _, v = torch.pca_lowrank(image_flat, q=min(c, 256))

    # 3. 将数据投影到前3个主成分上
    # 这会将C维特征转换为3维的“颜色”特征。
    image_colored = torch.matmul(image_flat, v[:, :3])

    # 4. 对每个通道进行百分位归一化以增强对比度
    # 定义用于拉伸的百分位范围。
    # 使用较低和较高的百分位（例如，2%和98%）有助于
    # 忽略极端异常值，并增强主要数据范围内的对比度，
    # 从而产生更“大”或更明显的颜色映射。
    low_p = 0.02
    high_p = 0.98

    # 遍历每个颜色通道 (R, G, B)
    for i in range(3):
        channel = image_colored[:, i]
        
        # 计算低百分位和高百分位的值
        v_low = torch.quantile(channel, low_p)
        v_high = torch.quantile(channel, high_p)
        
        # 避免除以零（如果通道没有变化）
        if v_high > v_low:
            # 归一化通道：将 [v_low, v_high] 范围映射到 [0, 1]
            image_colored[:, i] = (channel - v_low) / (v_high - v_low)
        else:
            # 如果没有变化，通道是恒定的，可以设为中间值（灰色）
            image_colored[:, i] = 0.5

    # 5. 将值裁剪到 [0, 1] 范围
    # 这确保了百分位范围之外的值被裁剪为纯黑或纯白。
    image_colored = torch.clamp(image_colored, 0, 1)
    
    # 6. 将着色后的数据重塑回原始的多视图格式
    # 这会恢复N个独立的视图，将 (N*H*W, 3) 转换回 (N, H, W, 3)。
    return image_colored.view(n, h, w, 3)

def get_stride_distribution(strides, dist_type='uniform'):

    # input strides sorted by descreasing order by default
    
    if dist_type == 'uniform':
        dist = np.ones(len(strides)) / len(strides)
    elif dist_type == 'exponential':
        lambda_param = 1.0
        dist = np.exp(-lambda_param * np.arange(len(strides)))
    elif dist_type.startswith('linear'): # e.g., linear_1_2
        try:
            start, end = map(float, dist_type.split('_')[1:])
            dist = np.linspace(start, end, len(strides))
        except ValueError:
            raise ValueError(f'Invalid linear distribution format: {dist_type}')
    else:
        raise ValueError('Unknown distribution type %s' % dist_type)

    # normalize to sum to 1
    return dist / np.sum(dist)


def fill_default_args(kwargs, func):
    import inspect  # a bit hacky but it works reliably
    signature = inspect.signature(func)

    for k, v in signature.parameters.items():
        if v.default is inspect.Parameter.empty:
            continue
        kwargs.setdefault(k, v.default)

    return kwargs


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


def is_symmetrized(gt1, gt2):
    x = gt1['instance']
    y = gt2['instance']
    if len(x) == len(y) and len(x) == 1:
        return False  # special case of batchsize 1
    ok = True
    for i in range(0, len(x), 2):
        ok = ok and (x[i] == y[i + 1]) and (x[i + 1] == y[i])
    return ok


def flip(tensor):
    """ flip so that tensor[0::2] <=> tensor[1::2] """
    return torch.stack((tensor[1::2], tensor[0::2]), dim=1).flatten(0, 1)


def interleave(tensor1, tensor2):
    res1 = torch.stack((tensor1, tensor2), dim=1).flatten(0, 1)
    res2 = torch.stack((tensor2, tensor1), dim=1).flatten(0, 1)
    return res1, res2


def transpose_to_landscape(head, activate=True):
    """ Predict in the correct aspect-ratio,
        then transpose the result in landscape 
        and stack everything back together.
    """
    def wrapper_no(decout, true_shape):
        B = len(true_shape)
        assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'
        H, W = true_shape[0].cpu().tolist()
        res = head(decout, (H, W))
        return res

    def wrapper_yes(decout, true_shape):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            return head(decout, (H, W))
        if is_portrait.all():
            return transposed(head(decout, (W, H)))

        # batch is a mix of both portraint & landscape
        def selout(ar): return [d[ar] for d in decout]
        l_result = head(selout(is_landscape), (H, W))
        p_result = transposed(head(selout(is_portrait), (W, H)))

        # allocate full result
        result = {}
        for k in l_result | p_result:
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        return result

    return wrapper_yes if activate else wrapper_no


def transposed(dic):
    return {k: v.swapaxes(1, 2) for k, v in dic.items()}


def invalid_to_nans(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = float('nan')
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr


def invalid_to_zeros(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = 0
        nnz = valid_mask.view(len(valid_mask), -1).sum(1)
    else:
        nnz = arr.numel() // len(arr) if len(arr) else 0  # number of point per image
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr, nnz

def save_tum_poses(traj, path):
    # traj = self.get_tum_poses()
    save_trajectory_tum_format(traj, path)
    return traj[0] # return the poses

def save_focals(focals, path):
    # convert focal to txt
    # focals = self.get_focals()
    np.savetxt(path, focals.detach().cpu().numpy(), fmt='%.6f')
    return focals

def save_intrinsics(K_raw, path):
    # K_raw = self.get_intrinsics()
    K = K_raw.reshape(-1, 9)
    np.savetxt(path, K.detach().cpu().numpy(), fmt='%.6f')
    return K_raw

def save_conf_maps(conf, path):
    # conf = self.get_conf()
    for i, c in enumerate(conf):
        np.save(f'{path}/conf_{i}.npy', c.detach().cpu().numpy())
    return conf

def save_rgb_imgs(imgs, path):
    # imgs = self.imgs
    for i, img in enumerate(imgs):
        # convert from rgb to bgr
        img = img[..., ::-1]
        cv2.imwrite(f'{path}/frame_{i:04d}.png', img*255)
    return imgs

def save_dynamic_masks(dynamic_masks, path):
    # dynamic_masks = self.dynamic_masks
    for i, dynamic_mask in enumerate(dynamic_masks):
        cv2.imwrite(f'{path}/dynamic_mask_{i}.png', (dynamic_mask * 255).detach().cpu().numpy().astype(np.uint8))
    return dynamic_masks

def save_depth_maps(depth_maps, path):
    images = []
    for i, depth_map in enumerate(depth_maps):
        depth_map_colored = cv2.applyColorMap((depth_map * 255).detach().cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
        img_path = f'{path}/frame_{(i):04d}.png'
        cv2.imwrite(img_path, depth_map_colored)
        images.append(Image.open(img_path))
        # Save npy file
        np.save(f'{path}/frame_{(i):04d}.npy', depth_map.detach().cpu().numpy())
    
    # Save gif using Pillow
    images[0].save(f'{path}/_depth_maps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
    return depth_maps

def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, list):
        return [to_cpu(xx) for xx in x]


def select_first_batch(inputs, weight_dtype=None):
    """
    移除 inputs 字典中所有张量的 batch 维度，并根据需要修改张量的数据类型。
    
    参数：
        inputs (dict): 包含多个带 batch 维度的张量的字典。
        keys (list, optional): 需要处理的键。如果为 None，则默认处理字典中的所有张量。
        weight_dtype (torch.dtype, optional): 若指定，则将所有张量转换为该数据类型。
    
    返回：
        dict: 移除 batch 维度并转换数据类型后的新字典。
    """
    new_dicts = {}
    keys = ["pose_enc", "depth", "world_points", "images", "extrinsic", "intrinsic", "world_points_from_depth", 'depth_conf', 'world_points_conf', "sem_color", "sem_color_conf"]
    for key, value in inputs.items():
        # 只处理 Tensor 类型且符合 keys 中条件的元素
        if isinstance(value, torch.Tensor) and (keys is None or key in keys):
            # 移除 batch 维度
            try:
                value = value[:1, ...].detach()
            except:
                value = value[:1, ...]
            
            # 如果指定了 weight_dtype，则转换数据类型
            if weight_dtype:
                value = value.to(weight_dtype)
        
        new_dicts[key] = value
    
    return new_dicts

def closed_form_inverse_se3(T):
    """
    兼容形如 [..., 4, 4] 的 T。
    返回与 T 相同形状的 T_inv。
    """
    # 假设 T 的形状是 [..., 4, 4]
    R = T[..., :3, :3]            # => [..., 3, 3]
    t = T[..., :3, 3]             # => [..., 3]
    R_inv = R.transpose(-1, -2)   # => [..., 3, 3]
    
    # 有时可先将 t reshape 成 [..., 3, 1] 再做矩阵乘法
    # => [B, S, 3, 1]
    t_expanded = t.unsqueeze(-1)
    t_inv = -(R_inv @ t_expanded).squeeze(-1)  # => [..., 3]
    
    # 填回 4x4
    I = torch.eye(4, device=T.device, dtype=T.dtype)
    # 先将 I broadcast 成 [..., 4, 4] 的形状
    out_shape = T.shape
    I_broadcast = I.unsqueeze(0).expand(out_shape[:-2] + (4, 4)).clone()
    
    # 写回 R_inv, t_inv
    I_broadcast[..., :3, :3] = R_inv
    I_broadcast[..., :3, 3]  = t_inv
    return I_broadcast

def normalize_camera_extrinsics_and_points_batch(
    extrinsics,
    cam_points=None,
    world_points=None,
    depths=None,
    scale_by_points=True,
    point_masks=None,
    mode="mean",
    seq_name=None,
):
    # Note this assumes we use cpu
    # extrinsics: (B, S, 3, 4)
    # world_points: (B, S, H, W, 3) or (*,3)
    # cam_points: same shape as world_points or something consistent
    # point_masks: (B, S, H, W) boolean mask if provided

    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    dtype = extrinsics.dtype


    # Convert extrinsics to homogeneous form: (B, N,4,4)
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)

    # Force the first camera to be the identity
    # Do we really need this? Close it now
    # identity_4x4 = torch.eye(4, device=device, dtype=dtype)
    # new_extrinsics[:, 0] = identity_4x4


    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None


    if scale_by_points:
        if cam_points is not None:
            new_cam_points = cam_points.clone()
        else:
            new_cam_points = None
        new_depths = depths.clone()

        dist = new_world_points.norm(dim=-1)
        dist_sum = (dist * point_masks).sum(dim=[1,2,3])
        valid_count = point_masks.sum(dim=[1,2,3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-3, max=1e3)

        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
            
        return new_extrinsics[:, :, :3], new_cam_points, new_world_points, new_depths
    else:
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths


def save_images_from_tensor(tensor, save_dir="frames", prefix="frame"):
    """
    将形状 (1, 4, H, W, 3) 的 Tensor 逐帧保存为图片。

    参数：
        tensor (torch.Tensor): 形状 (1, 4, H, W, 3) 的图像数据，值范围应在 [0, 1] 或 [0, 255]。
        save_dir (str): 保存图片的目录。
        prefix (str): 保存文件名前缀。
    """
    import os

    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 去掉 batch 维度 (1, 4, H, W, 3) -> (4, H, W, 3)
    tensor = tensor.squeeze(0)

    # 逐帧保存
    for i, img_tensor in enumerate(tensor):
        img_array = img_tensor.permute(1,2,0).cpu().numpy()  # 转 NumPy
        img_array = (img_array * 255).astype(np.uint8)  # 如果值在 [0,1]，转换到 [0,255]

        # 转换为 PIL 图片
        img = Image.fromarray(img_array)

        # 保存图片
        img.save(f"{save_dir}/{prefix}_{i}.png")
        print(f"Saved: {save_dir}/{prefix}_{i}.png")

def closed_form_inverse_se3(T):
    """
    兼容形如 [..., 4, 4] 的 T。
    返回与 T 相同形状的 T_inv。
    """
    # 假设 T 的形状是 [..., 4, 4]
    R = T[..., :3, :3]            # => [..., 3, 3]
    t = T[..., :3, 3]             # => [..., 3]
    R_inv = R.transpose(-1, -2)   # => [..., 3, 3]
    
    # 有时可先将 t reshape 成 [..., 3, 1] 再做矩阵乘法
    # => [B, S, 3, 1]
    t_expanded = t.unsqueeze(-1)
    t_inv = -(R_inv @ t_expanded).squeeze(-1)  # => [..., 3]
    
    # 填回 4x4
    I = torch.eye(4, device=T.device, dtype=T.dtype)
    # 先将 I broadcast 成 [..., 4, 4] 的形状
    out_shape = T.shape
    I_broadcast = I.unsqueeze(0).expand(out_shape[:-2] + (4, 4)).clone()
    
    # 写回 R_inv, t_inv
    I_broadcast[..., :3, :3] = R_inv
    I_broadcast[..., :3, 3]  = t_inv
    return I_broadcast

def normalize_camera_extrinsics_and_points_batch(
    extrinsics,
    cam_points=None,
    world_points=None,
    depths=None,
    scale_by_points=True,
    point_masks=None,
    mode="mean",
    seq_name=None,
):
    # Note this assumes we use cpu
    # extrinsics: (B, S, 3, 4)
    # world_points: (B, S, H, W, 3) or (*,3)
    # cam_points: same shape as world_points or something consistent
    # point_masks: (B, S, H, W) boolean mask if provided
        
    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    dtype = extrinsics.dtype


    # Convert extrinsics to homogeneous form: (B, N,4,4)
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)

    # Force the first camera to be the identity
    # Do we really need this? Close it now
    # identity_4x4 = torch.eye(4, device=device, dtype=dtype)
    # new_extrinsics[:, 0] = identity_4x4


    
    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        world_points = world_points.to(torch.float32)
        R = R.to(torch.float32)
        t = t.to(torch.float32)
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None


    if scale_by_points:
        if cam_points is not None:
            new_cam_points = cam_points.clone()
        else:
            new_cam_points = None
        new_depths = depths.clone()

        dist = new_world_points.norm(dim=-1)
        dist_sum = (dist * point_masks).sum(dim=[1,2,3])
        valid_count = point_masks.sum(dim=[1,2,3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-3, max=1e3)

        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
            
        return new_extrinsics[:, :, :3], new_cam_points, new_world_points, depths
    else:
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths

def visualize_mask_with_colormap(batch_masks, cmap_name="tab20"):
    masks = batch_masks[0]  # [N, H, W]
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    N, H, W = masks.shape
    label_map = np.zeros((H, W), dtype=np.int32)

    for i in range(N):
        label_map[masks[i] > 0] = i + 1  # 保证每个 mask 是唯一 index

    # 映射 colormap（归一化到 [0, 1]，然后转为 RGB）
    cmap = cm.get_cmap(cmap_name, N + 1)  # e.g. "tab20", "gist_ncar", etc.
    color_mask = cmap(label_map / (N + 1))[:, :, :3]  # 去掉 alpha

    return color_mask

def visualize_mask2former_topk(class_logits, mask_logits, cmap_name="tab20"):
    """
    class_logits: Tensor [num_queries, num_classes]
    mask_logits: Tensor [num_queries, H, W]
    """
    class_probs = class_logits.softmax(dim=-1)  # [num_queries, num_classes]
    scores, labels = class_probs.max(dim=-1)    # [num_queries] - confidence scores
    topk = 5

    # 选取 topk 高置信度的 query
    topk_indices = torch.topk(scores, k=topk).indices
    selected_masks = mask_logits[topk_indices]  # [topk, H, W]

    # 转为 numpy，生成 label map
    selected_masks = selected_masks.detach().sigmoid().cpu().numpy()  # 转为 [0,1]
    label_map = np.zeros_like(selected_masks[0], dtype=np.int32)  # [H, W]

    for i, m in enumerate(selected_masks):
        binary_mask = m > 0.5  # 可调阈值
        label_map[binary_mask] = i + 1  # i+1 避免和背景 0 冲突

    # colormap 映射
    cmap = cm.get_cmap(cmap_name, topk + 1)
    color_mask = cmap(label_map / (topk + 1))[:, :, :3]  # 去掉 alpha 通道
    return color_mask

def visualize_mask2former_pixelwise_assignment(class_logits, mask_logits, cmap_name="tab20"):
    """
    class_logits: Tensor [num_queries, num_classes]
    mask_logits: Tensor [num_queries, H, W]
    """
    class_probs = class_logits.softmax(dim=-1)  # [num_queries, num_classes]
    scores, labels = class_probs.max(dim=-1)    # [num_queries] - confidence scores

    # Get soft masks and multiply by query score
    soft_masks = mask_logits.sigmoid() * scores[:, None, None]  # [num_queries, H, W]

    # 进行 pixel-wise 最大得分选取（即每个像素来自哪个 query）
    soft_masks_np = soft_masks.detach().cpu().numpy()
    assigned_indices = np.argmax(soft_masks_np, axis=0) + 1  # [H, W], avoid 0 as background
    max_scores = np.max(soft_masks_np, axis=0)               # [H, W]

    # 抑制低得分区域（可选）
    assigned_indices[max_scores < 0.5] = 0  # 设置背景（可调）

    # 映射到颜色图
    num_queries = mask_logits.shape[0]
    cmap = cm.get_cmap(cmap_name, num_queries + 1)
    color_mask = cmap(assigned_indices / (num_queries + 1))[:, :, :3]  # RGB
    return color_mask
