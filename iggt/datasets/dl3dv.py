import sys
sys.path.append('.')
import os
import torch,cv2
import numpy as np
import os.path as osp
import glob
import torchvision.transforms as tvf
import random
from PIL import Image
import PIL
import json
import joblib
from tqdm import tqdm

import pycocotools.mask as mask_util
import iggt.datasets.utils.cropping as cropping
from iggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from iggt.utils.geometry import depthmap_to_absolute_camera_coordinates, depth_to_world_coords_points,closed_form_inverse_se3
from iggt.datasets.base.base_stereo_view_dataset import is_good_type, view_name, transpose_to_landscape
from iggt.datasets.utils.image_ranking import compute_ranking
from iggt.datasets.utils.misc import threshold_depth_map
from dataset_preprocess.read_write_model import read_images_binary
from iggt.datasets.utils.cropping  import ImageList, camera_matrix_of_crop, bbox_from_intrinsics_in_out
from iggt.datasets.sav import show_anns

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC
    
np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

ToTensor = tvf.ToTensor()

def qvec_to_rotmat(qvec):
    """
    Convert quaternion to rotation matrix.
    qvec is a 4-element array: [q0, q1, q2, q3] (w, x, y, z)
    """
    q0, q1, q2, q3 = qvec

    # Calculate the rotation matrix from the quaternion
    R = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)],
        [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)]
    ])
    
    return R

def get_camera_pose(qvec, tvec):
    """
    Convert quaternion and translation vector into a 4x4 camera pose matrix.
    """
    # Convert quaternion to rotation matrix
    R = qvec_to_rotmat(qvec)
    
    # Create the 4x4 transformation matrix (camera pose)
    camera_pose = np.eye(4)  # Identity matrix
    camera_pose[:3, :3] = R  # Rotation part
    camera_pose[:3, 3] = tvec  # Translation part
    
    return camera_pose

def parse_camera_params(json_data):
    # 相机内参
    camera_params = {
        "w": json_data["w"],  # 图像宽度
        "h": json_data["h"],  # 图像高度
        "fl_x": json_data["fl_x"],  # 焦距 x
        "fl_y": json_data["fl_y"],  # 焦距 y
        "cx": json_data["cx"],  # 主点 x
        "cy": json_data["cy"],  # 主点 y
        "k1": json_data["k1"],  # 畸变参数 k1
        "k2": json_data["k2"],  # 畸变参数 k2
        "p1": json_data["p1"],  # 畸变参数 p1
        "p2": json_data["p2"],  # 畸变参数 p2
        "camera_model": json_data["camera_model"]  # 相机模型
    }
    
    # 提取每一帧的外参
    frames = []
    for frame in json_data["frames"]:
        transform_matrix = np.array(frame["transform_matrix"])  # 转换矩阵
        frame_data = {
            "file_path": frame["file_path"],  # 图像路径
            "colmap_im_id": frame["colmap_im_id"],  # COLMAP 图像ID
            "transform_matrix": transform_matrix  # 变换矩阵 (外参)
        }
        frames.append(frame_data)

    return camera_params, frames

def scale_intrinsics(camera_params, target_width, target_height):
    # 目标图像的宽高
    w, h = camera_params["w"], camera_params["h"]
    
    # 按照目标分辨率缩放焦距和主点
    scale_x = target_width / w
    scale_y = target_height / h
    
    # 缩放焦距
    fl_x_new = camera_params["fl_x"] * scale_x
    fl_y_new = camera_params["fl_y"] * scale_y
    
    # 缩放主点
    cx_new = camera_params["cx"] * scale_x
    cy_new = camera_params["cy"] * scale_y
    
    # 返回新的相机内参
    new_camera_params = {
        "fl_x": fl_x_new,
        "fl_y": fl_y_new,
        "cx": cx_new,
        "cy": cy_new
    }
    
    return new_camera_params

def get_camera_intrinsic_matrix(camera_params):
    camera_params = scale_intrinsics(camera_params, target_width = 480, target_height = 270)
    # 组合成内参矩阵 K
    K = np.array([
        [camera_params['fl_x'], 0, camera_params['cx']],
        [0, camera_params['fl_y'], camera_params['cy']],
        [0, 0, 1]
    ])
    return K

class Dl3dv(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='datasets/processed_dl3dv_ours_parts/processed_dl3dv_ours',
                 mask_gt_location='datasets/processed_dl3dv_ours_parts/processed_dl3dv_ours',
                 use_cache = True,
                 dset='',
                 use_augs=False,
                 specify=False,
                 top_k = 256,
                 z_far=500,
                 quick=False,
                 verbose=False,
                 load_mask=True,
                 *args, 
                 **kwargs
                 ):

        print('loading DL3DV dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'DL3DV'
        self.dataset_location = dataset_location
        self.mask_gt_location = mask_gt_location
        self.split = dset
        self.top_k = top_k
        self.use_cache = use_cache
        self.specify = specify
        self.z_far = z_far
        
        self.verbose = verbose
        self.load_mask = load_mask

        self.use_augs = use_augs
        self.dset = dset

        self.full_idxs = []
        self.all_rgb_paths = []
        self.all_depth_paths = []
        self.all_traj_paths = []
        self.all_extrinsic = []
        self.all_intrinsic = []
        self.all_outlier_mask_paths = []
        self.all_sky_mask_paths = []
        self.all_annotation_paths = []
        self.all_mask_paths = []
        self.all_mask_index = []
        self.rank = dict()


        self.subdirs = []
        self.sequences = []
        self.subdirs.append(os.path.join(dataset_location, dset))

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*/")):
                self.sequences.append(seq)

        self.sequences = sorted(self.sequences)
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))
        
        ## load trajectories
        print('loading trajectories...')

        if quick:
           self.sequences = self.sequences[:5] 
        
        if self.use_cache:
            dataset_location = 'sam_annotations/dl3dv_annotations'
            all_rgb_paths_file = os.path.join(dataset_location, dset, 'rgb_paths.json')
            all_depth_paths_file = os.path.join(dataset_location, dset, 'depth_paths.json')
            all_outlier_mask_paths_file = os.path.join(dataset_location, dset, 'outlier_mask_paths.json')
            all_sky_mask_paths_file = os.path.join(dataset_location, dset, 'sky_mask_paths.json')
            
            with open(all_rgb_paths_file, 'r', encoding='utf-8') as file:
                self.all_rgb_paths = json.load(file)
            with open(all_depth_paths_file, 'r', encoding='utf-8') as file:
                self.all_depth_paths = json.load(file)    
            with open(all_outlier_mask_paths_file, 'r', encoding='utf-8') as file:
                self.all_outlier_mask_paths = json.load(file)       
            with open(all_sky_mask_paths_file, 'r', encoding='utf-8') as file:
                self.all_sky_mask_paths = json.load(file)                            
                
            self.all_rgb_paths = [self.all_rgb_paths[str(i)] for i in range(len(self.all_rgb_paths))]
            self.all_depth_paths = [self.all_depth_paths[str(i)] for i in range(len(self.all_depth_paths))]
            self.all_outlier_mask_paths = [self.all_outlier_mask_paths[str(i)] for i in range(len(self.all_outlier_mask_paths))]
            self.all_sky_mask_paths = [self.all_sky_mask_paths[str(i)] for i in range(len(self.all_sky_mask_paths))]
            self.full_idxs = list(range(len(self.all_rgb_paths)))
            self.rank = joblib.load(os.path.join(dataset_location, dset, 'rankings.joblib'))
            self.all_intrinsic = joblib.load(os.path.join(dataset_location, dset, 'intrinsics.joblib'))
            self.all_extrinsic = joblib.load(os.path.join(dataset_location, dset, 'extrinsics.joblib'))
            self.all_mask_paths = joblib.load(os.path.join(dataset_location, dset, 'mask_gts.joblib'))
            self.all_mask_index = joblib.load(os.path.join(dataset_location, dset, 'mask_index.joblib'))
            
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))            
        
        else:
            for seq in self.sequences:
                if self.verbose: 
                    print('seq', seq)

                if seq.split('/')[-2] == '001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f':
                    continue
                
                seq_name = seq.split('/')[-2]
                seq = os.path.join(self.dataset_location, dset, seq_name)
                mask_seq = os.path.join(self.mask_gt_location, dset, seq_name)
                mask_json_path = os.path.join(mask_seq,"auto_masks.json")
                if not os.path.exists(mask_json_path):
                    print(f"Mask JSON file not found for sequence {seq_name}: {mask_json_path}")
                    continue
                
                # assert os.path.exists(mask_json_path), f"Mask JSON file not found: {mask_json_path}"
                
                rgb_path = os.path.join(seq, "dense", "rgb")
                depth_path = os.path.join(seq, "dense", 'depth')
                caminfo_path = os.path.join(seq,'dense','cam')
                outlier_mask_path = os.path.join(seq, 'dense', 'outlier_mask')
                sky_mask_path = os.path.join(seq, 'dense', 'sky_mask')
                num_frames = len(glob.glob(os.path.join(rgb_path, '*.png')))

                if num_frames < 24:
                    print(f"Skipping sequence {seq} with only {num_frames} frames.")
                    continue

                new_sequence = list(len(self.full_idxs) + np.arange(num_frames))
                old_sequence_length = len(self.full_idxs)
                self.full_idxs.extend(new_sequence)
                self.all_rgb_paths.extend(sorted(glob.glob(os.path.join(rgb_path, 'frame_*.png')))) 
                self.all_depth_paths.extend(sorted(glob.glob(os.path.join(depth_path, 'frame_*.npy'))))
                self.all_outlier_mask_paths.extend(sorted(glob.glob(os.path.join(outlier_mask_path, 'frame_*.png'))))
                self.all_sky_mask_paths.extend(sorted(glob.glob(os.path.join(sky_mask_path, 'frame_*.png'))))
                seq_annotaions_path = sorted(glob.glob(os.path.join(caminfo_path, 'frame_*.npz')))
                self.all_annotation_paths.extend(seq_annotaions_path)
               
                with open(mask_json_path, 'r', encoding='utf-8') as file:
                    mask_gts = json.load(file)
                    masklets = mask_gts['masklet']

                self.all_mask_paths.extend([mask_json_path]*len(masklets))
                self.all_mask_index.extend(list(range(len(masklets))))
       
                N = len(self.full_idxs)
                
                assert len(self.all_rgb_paths) == N and \
                    len(self.all_outlier_mask_paths) == N and \
                    len(self.all_sky_mask_paths) == N and \
                    len(self.all_annotation_paths) == N and \
                    len(self.all_mask_paths) == N and \
                    len(self.all_depth_paths) == N, f"Number of images, depth maps, and annotations do not match in {seq}."
                
                extrinsics_seq = []  
                #load intrinsics and extrinsics
                for anno in seq_annotaions_path:
                    camera_info = np.load(anno)
                    pose = np.array(camera_info['pose'],dtype=np.float32)
                    intrinsics = np.array(camera_info['intrinsic'],dtype=np.float32)
                    self.all_extrinsic.extend([pose])
                    self.all_intrinsic.extend([intrinsics])
                    extrinsics_seq.append(pose)
                    
                all_extrinsic_numpy = np.array(extrinsics_seq)
                #compute ranking
                ranking, dists = compute_ranking(all_extrinsic_numpy, lambda_t=1.0, normalize=True, batched=True)
                ranking += old_sequence_length
                for ind, i in enumerate(range(old_sequence_length, len(self.full_idxs))):
                    self.rank[i] = ranking[ind] 
                    
            # 保存为 JSON 文件
            os.makedirs(f'sam_annotations/dl3dv_annotations/{self.dset}', exist_ok=True)
            # self._save_paths_to_json(self.all_rgb_paths, f'sam_annotations/dl3dv_annotations/{self.dset}/rgb_paths.json')
            # self._save_paths_to_json(self.all_depth_paths, f'sam_annotations/dl3dv_annotations/{self.dset}/depth_paths.json')
            # self._save_paths_to_json(self.all_outlier_mask_paths, f'sam_annotations/dl3dv_annotations/{self.dset}/outlier_mask_paths.json')
            # self._save_paths_to_json(self.all_sky_mask_paths, f'sam_annotations/dl3dv_annotations/{self.dset}/sky_mask_paths.json')
            # joblib.dump(self.rank, f'sam_annotations/dl3dv_annotations/{self.dset}/rankings.joblib')
            # joblib.dump(self.all_extrinsic, f'sam_annotations/dl3dv_annotations/{self.dset}/extrinsics.joblib')
            # joblib.dump(self.all_intrinsic, f'sam_annotations/dl3dv_annotations/{self.dset}/intrinsics.joblib')
            # joblib.dump(self.all_mask_paths, f'sam_annotations/dl3dv_annotations/{self.dset}/mask_gts.joblib')
            # joblib.dump(self.all_mask_index, f'sam_annotations/dl3dv_annotations/{self.dset}/mask_index.joblib')
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))
        
    
    def _save_paths_to_json(self, paths, filename):
        path_dict = {i: path for i, path in enumerate(paths)}
        with open(filename, 'w') as f:
            json.dump(path_dict, f, indent=4)

    def __len__(self):
        return len(self.full_idxs)
    
    def _get_views(self, index, num, resolution, rng):
        if num != 1:
            # get the top num frames of the anchor frame
            anchor_frame = self.full_idxs[index]
            top_k = self.top_k if len(self.rank[anchor_frame]) > self.top_k else len(self.rank[anchor_frame])
            rest_frame = self.rank[anchor_frame][:top_k]
            if self.specify:
                step = max(1, len(rest_frame) // (num - 1))
                rest_frame_indexs = [rest_frame[i] for i in range(0, len(rest_frame), step)][:num-1]
            else:
                rest_frame_indexs = random.sample(list(rest_frame), num-1) 
            full_idx = [anchor_frame] + rest_frame_indexs  
            
            rgb_paths = [self.all_rgb_paths[i] for i in full_idx]
            depth_paths = [self.all_depth_paths[i] for i in full_idx]
            camera_pose_list = [self.all_extrinsic[i] for i in full_idx]
            intrinsics_list = [self.all_intrinsic[i] for i in full_idx]
            sky_mask_paths = [self.all_sky_mask_paths[i] for i in full_idx]
            outlier_mask_paths = [self.all_outlier_mask_paths[i] for i in full_idx]
            mask_gt_paths = [self.all_mask_paths[i] for i in full_idx]
            mask_gt_indexs = [self.all_mask_index[i] for i in full_idx]
            
        else:
            full_idx = self.full_idxs[index]
            rgb_paths = [self.all_rgb_paths[full_idx]]
            depth_paths = [self.all_depth_paths[full_idx]]
            camera_pose_list = [self.all_extrinsic[full_idx]]
            intrinsics_list = [self.all_intrinsic[full_idx]]
            sky_mask_paths = [self.all_sky_mask_paths[full_idx]]
            outlier_mask_paths = [self.all_outlier_mask_paths[full_idx]]
            mask_gt_paths = [self.all_mask_paths[full_idx]]
            mask_gt_indexs = [self.all_mask_index[full_idx]]

        with open(mask_gt_paths[0], 'r', encoding='utf-8') as file:
            mask_gts = json.load(file)
            masklets = mask_gts['masklet']
            
        views = []
        for i in range(num):
            impath = rgb_paths[i]
            depthpath = depth_paths[i]
            camera_pose = camera_pose_list[i]
            intrinsics = intrinsics_list[i]
            outlier_mask_path = outlier_mask_paths[i]
            sky_mask_path = sky_mask_paths[i]
            mask_gt_index = mask_gt_indexs[i]

            # load image and depth
            rgb_image = Image.open(impath)
            rgb_image = rgb_image.convert("RGB")
            depthmap = np.load(depthpath)
            sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_GRAYSCALE) == 0
            outlier_mask = cv2.imread(outlier_mask_path, cv2.IMREAD_GRAYSCALE) == 0
            depthmap[sky_mask & outlier_mask] = depthmap[sky_mask & outlier_mask]  # 保留有效值
            depthmap[~(sky_mask & outlier_mask)] = 0  # 设置无效值为0
            depthmap = threshold_depth_map(depthmap, max_percentile=98, min_percentile=-1)
            mask = mask_util.decode(masklets[mask_gt_index]) 
                        
            rgb_image, depthmap, mask, intrinsics  = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, mask, resolution, rng=rng, info=impath)        
            
            mask = mask > 0
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                mask=mask,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=rgb_paths[i].split('/')[-3],
                instance=osp.split(rgb_paths[i])[1],
            ))

        return views
 
    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, mask, resolution, rng=None, info=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, mask, intrinsics = self.crop_image_depthmap_mask(image, depthmap, mask, intrinsics, crop_bbox)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1*W:
            # image is portrait mode
            # resolution = resolution[::-1]
            pass
            
        elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if rng.integers(2):
                # resolution = resolution[::-1]
                pass

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_focal:
            crop_scale = self.aug_focal + (1.0 - self.aug_focal) * np.random.beta(0.5, 0.5) # beta distribution, bi-modal
            image, depthmap, mask, intrinsics = self.center_crop_image_depthmap_mask(image, depthmap, mask, intrinsics, crop_scale)

        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, depthmap, mask, intrinsics = self.rescale_image_depthmap_mask(image, depthmap, mask, intrinsics, target_resolution) # slightly scale the image a bit larger than the target resolution

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, mask, intrinsics2 = self.crop_image_depthmap_mask(image, depthmap, mask, intrinsics, crop_bbox)

        return image, depthmap, mask, intrinsics2    
    
    def crop_image_depthmap_mask(self, image, depthmap, mask, camera_intrinsics, crop_bbox):
        """
        Return a crop of the input view.
        """
        image = ImageList(image)
        l, t, r, b = crop_bbox

        image = image.crop((l, t, r, b))
        depthmap = depthmap[t:b, l:r]
        mask = mask[t:b, l:r]

        camera_intrinsics = camera_intrinsics.copy()
        camera_intrinsics[0, 2] -= l
        camera_intrinsics[1, 2] -= t

        return image.to_pil(), depthmap, mask, camera_intrinsics

    def center_crop_image_depthmap_mask(self, image, depthmap, mask, camera_intrinsics, crop_scale):
        """
        Jointly center-crop an image and its depthmap, and adjust the camera intrinsics accordingly.

        Parameters:
        - image: PIL.Image or similar, the input image.
        - depthmap: np.ndarray, the corresponding depth map.
        - camera_intrinsics: np.ndarray, the 3x3 camera intrinsics matrix.
        - crop_scale: float between 0 and 1, the fraction of the image to keep.

        Returns:
        - cropped_image: PIL.Image, the center-cropped image.
        - cropped_depthmap: np.ndarray, the center-cropped depth map.
        - adjusted_intrinsics: np.ndarray, the adjusted camera intrinsics matrix.
        """
        # Ensure crop_scale is valid
        assert 0 < crop_scale <= 1, "crop_scale must be between 0 and 1"

        # Convert image to ImageList for consistent processing
        image = ImageList(image)
        input_resolution = np.array(image.size)  # (width, height)
        if depthmap is not None:
            # Ensure depthmap matches the image size
            assert depthmap.shape[:2] == tuple(image.size[::-1]), "Depthmap size must match image size"

        # Compute output resolution after cropping
        output_resolution = np.floor(input_resolution * crop_scale).astype(int)
        # get the correct crop_scale
        crop_scale = output_resolution / input_resolution

        # Compute margins (amount to crop from each side)
        margins = input_resolution - output_resolution
        offset = margins / 2  # Since we are center cropping

        # Calculate the crop bounding box
        l, t = offset.astype(int)
        r = l + output_resolution[0]
        b = t + output_resolution[1]
        crop_bbox = (l, t, r, b)

        # Crop the image and depthmap
        image = image.crop(crop_bbox)
        if depthmap is not None:
            depthmap = depthmap[t:b, l:r]

        # Adjust the camera intrinsics
        adjusted_intrinsics = camera_intrinsics.copy()

        # Adjust focal lengths (fx, fy)                         # no need to adjust focal lengths for cropping
        # adjusted_intrinsics[0, 0] /= crop_scale[0]  # fx
        # adjusted_intrinsics[1, 1] /= crop_scale[1]  # fy

        # Adjust principal point (cx, cy)
        adjusted_intrinsics[0, 2] -= l  # cx
        adjusted_intrinsics[1, 2] -= t  # cy

        return image.to_pil(), depthmap, mask, adjusted_intrinsics

    def rescale_image_depthmap_mask(self, image, depthmap, mask, camera_intrinsics, output_resolution, force=True):
        """ Jointly rescale a (image, depthmap) 
            so that (out_width, out_height) >= output_res
        """
        image = ImageList(image)
        input_resolution = np.array(image.size)  # (W,H)
        output_resolution = np.array(output_resolution)
        if depthmap is not None:
            # can also use this with masks instead of depthmaps
            assert tuple(depthmap.shape[:2]) == image.size[::-1]
        if mask is not None:
            assert mask.shape[:2] == image.size[::-1]
        # define output resolution
        assert output_resolution.shape == (2,)
        scale_final = max(output_resolution / image.size) + 1e-8
        if scale_final >= 1 and not force:  # image is already smaller than what is asked
            return (image.to_pil(), depthmap, camera_intrinsics)
        output_resolution = np.floor(input_resolution * scale_final).astype(int)

        # first rescale the image so that it contains the crop
        image = image.resize(tuple(output_resolution), resample=lanczos if scale_final < 1 else bicubic)
        if depthmap is not None:
            depthmap = cv2.resize(depthmap, output_resolution, fx=scale_final,
                                fy=scale_final, interpolation=cv2.INTER_NEAREST)
        if mask is not None:
            mask = cv2.resize(mask, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)

        # no offset here; simple rescaling
        camera_intrinsics = camera_matrix_of_crop(
            camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

        return image.to_pil(), depthmap, mask, camera_intrinsics

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if len(idx) == 2:
                # the idx is specifying the aspect-ratio
                idx, ar_idx = idx
                num = 1
            else:
                idx, ar_idx, num = idx
        else:
            assert len(self._resolutions) == 1
            num = 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, '_rng'):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, num, resolution, self._rng)
        assert len(views) == num

        # check data-types
        for v, view in enumerate(views):
            assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view['idx'] = (idx, ar_idx, v)

            # encode the image
            width, height = view['img'].size
            view['true_shape'] = np.int32((height, width))
            view['img'] = self.transform(view['img'])

            assert 'camera_intrinsics' in view
            if 'camera_pose' not in view:
                view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
            assert 'pts3d' not in view
            assert 'valid_mask' not in view
            assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
            view['z_far'] = self.z_far
            # world_coords_points, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            # view['world_coords_points'] = world_coords_points
            # view['point_mask'] = valid_mask & np.isfinite(world_coords_points).all(axis=-1)
            
            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']

            view['camera_pose'] = closed_form_inverse_se3(view['camera_pose'][None])[0]            
            world_coords_points, cam_coords_points, point_mask = (
                depth_to_world_coords_points(view['depthmap'], view['camera_pose'], view["camera_intrinsics"], z_far = self.z_far)
            )
            view['world_coords_points'] = world_coords_points
            view['cam_coords_points'] = cam_coords_points
            view['point_mask'] = point_mask


        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')
            
        # Initialize lists to store concatenated data for each field
        img_list = []
        depthmap_list = []
        camera_pose_list = []
        camera_intrinsics_list = []
        pts3d_list = []
        true_shape_list = []
        valid_mask_list = []
        label_list = []
        instance_list = []
        mask_list = []

        # Iterate over the views and concatenate each field
        for view in views:
            img_list.append(view['img'])
            depthmap_list.append(view['depthmap'][:, :, np.newaxis])
            camera_pose_list.append(view['camera_pose'][:3])
            camera_intrinsics_list.append(view['camera_intrinsics'])
            pts3d_list.append(view['world_coords_points'])
            true_shape_list.append(view['true_shape'])
            valid_mask_list.append(view['point_mask'])
            label_list.append(view['label'])
            instance_list.append(view['instance'])
            mask_list.append(view['mask'])

        # Concatenate the lists along the first dimension (n)
        img = torch.stack(img_list)
        depthmap = np.stack(depthmap_list)
        camera_pose = np.stack(camera_pose_list)
        camera_intrinsics = np.stack(camera_intrinsics_list)
        pts3d = np.stack(pts3d_list)
        true_shape = np.array(true_shape_list)
        valid_mask = np.stack(valid_mask_list)    
        mask = np.stack(mask_list)

        return dict(
                images=img, #(n, c, h, w)
                depth=depthmap, #(n, h, w, 1)
                extrinsic=camera_pose, #(n, 3, 4)
                intrinsic=camera_intrinsics, #(n, 3, 3)
                dataset=self.dataset_label,
                label=label_list,
                instance=instance_list,
                world_points=pts3d, #(n, h, w, 3)
                true_shape=true_shape,
                valid_mask=valid_mask,
                mask_gt = mask)

if __name__ == "__main__":

    from iggt.viz import SceneViz, auto_cam_size
    from iggt.utils.image import rgb

    use_augs = False
    num_views = 10
    n_views_list = range(num_views)
    top_k = 100
    quick = False  # Set to True for quick testing


    def visualize_scene(idx):
        views = dataset[idx]
        # assert len(views['images']) == num_views, f"Expected {num_views} views, got {len(views)}"
        viz = SceneViz()
        poses = views['extrinsic']
        views['extrinsic'] = closed_form_inverse_se3(views['extrinsic'])   
        cam_size = max(auto_cam_size(poses), 0.25)
        for view_idx in n_views_list:
            pts3d = views['world_points'][view_idx]
            valid_mask = views['valid_mask'][view_idx]
            colors = rgb(views['images'][view_idx])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=(views['extrinsic'][view_idx]),
                        focal=views['intrinsic'][view_idx][0, 0],
                        color=(255, 0, 0),
                        image=colors,
                        cam_size=cam_size)
        # return viz.show()
        viz.save_glb(f'dl3dv_{dataset.dset}_views_{num_views}.glb')
        return
    
    def visualize_mask(idx):
        batch = dataset[idx]
        manual_mask_colors = np.random.random((256, 3))
        colors = manual_mask_colors[:batch['mask_gt'].shape[-1]]
        mask_gt_list = [batch['mask_gt'][i] for i in range(batch['mask_gt'].shape[0])]
        canvas_bgr = []
        for i in range(len(batch['images'])):
            slices = [mask_gt_list[i][:, :, j] for j in range(mask_gt_list[i].shape[2])]
            canvas = show_anns(slices, colors=colors)
            canvas_rgb = (canvas[:, :, :3] * 255).astype(np.uint8)
            alpha = canvas[:, :, 3:4]
            # White background
            bg = np.ones_like(canvas_rgb, dtype=np.uint8) * 255
            canvas_rgb = (canvas_rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
            canvas_bgr.append(cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR))
            
        viz = SceneViz()
        poses = batch['extrinsic']
        batch['extrinsic'] = closed_form_inverse_se3(batch['extrinsic'])   
        cam_size = max(auto_cam_size(poses), 0.25)
        for view_idx in n_views_list:
            pts3d = batch['world_points'][view_idx]
            valid_mask = batch['valid_mask'][view_idx]
            colors = canvas_bgr[view_idx]
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=(batch['extrinsic'][view_idx]),
                        focal=batch['intrinsic'][view_idx][0, 0],
                        color=(255, 0, 0),
                        image=colors,
                        cam_size=cam_size)
        # return viz.show()
        viz.save_glb(f'dl3dv_{dataset.dset}_views_mask_{num_views}.glb')
        return

    dataset = Dl3dv(
        dataset_location="/mnt/juicefs/datasets/cut3r_processed/processed_dl3dv_ours_parts/processed_dl3dv_ours",
        mask_gt_location="/mnt/juicefs/sam2_results/",
        dset = '1K',
        use_cache = True,
        top_k= 50,
        quick=False,
        verbose=True,
        resolution=(512,512), 
        seed = 777,
        load_mask=False,
        aug_crop=16,
        use_augs=True,
        z_far = 200)

    print("Dataset loaded successfully.")
    # idx = random.randint(0, len(dataset)-1)
    # visualize_scene((100,0,num_views))
    visualize_mask((1000,0,16))
    # print(len(dataset))
