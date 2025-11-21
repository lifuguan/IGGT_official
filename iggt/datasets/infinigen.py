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


import iggt.datasets.utils.cropping as cropping
from iggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from iggt.utils.geometry import depthmap_to_absolute_camera_coordinates, depth_to_world_coords_points,closed_form_inverse_se3
from iggt.datasets.base.base_stereo_view_dataset import is_good_type, view_name, transpose_to_landscape
from iggt.datasets.utils.image_ranking import compute_ranking
from iggt.datasets.utils.misc import threshold_depth_map
from dataset_preprocess.read_write_model import read_images_binary
from iggt.datasets.utils.cropping  import ImageList, camera_matrix_of_crop, bbox_from_intrinsics_in_out

try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

ToTensor = tvf.ToTensor()


class Infinigen(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='datasets/infinigen',
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

        print('loading Infinigen dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'Infinigen'
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
        self.all_seg_mask_paths = []
        self.all_annotation_paths = []
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
           self.sequences = self.sequences[:2] 
        
        if self.use_cache:
            dataset_location = 'annotations/infinigen_annotations'
            all_rgb_paths_file = os.path.join(dataset_location, dset, 'rgb_paths.json')
            all_depth_paths_file = os.path.join(dataset_location, dset, 'depth_paths.json')
            all_seg_mask_paths_file = os.path.join(dataset_location, dset, 'seg_mask_paths.json')
            with open(all_rgb_paths_file, 'r', encoding='utf-8') as file:
                self.all_rgb_paths = json.load(file)
            with open(all_depth_paths_file, 'r', encoding='utf-8') as file:
                self.all_depth_paths = json.load(file)    
            with open(all_seg_mask_paths_file, 'r', encoding='utf-8') as file:
                self.all_seg_mask_paths = json.load(file)                             

            self.all_rgb_paths = [self.all_rgb_paths[str(i)] for i in range(len(self.all_rgb_paths))]
            self.all_depth_paths = [self.all_depth_paths[str(i)] for i in range(len(self.all_depth_paths))]
            self.all_seg_mask_paths = [self.all_seg_mask_paths[str(i)] for i in range(len(self.all_seg_mask_paths))]
            self.full_idxs = list(range(len(self.all_rgb_paths)))
            self.rank = joblib.load(os.path.join(dataset_location, dset, 'rankings.joblib'))
            self.all_intrinsic = joblib.load(os.path.join(dataset_location, dset, 'intrinsics.joblib'))
            self.all_extrinsic = joblib.load(os.path.join(dataset_location, dset, 'extrinsics.joblib'))
            
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))            
        
        else:
            for seq in self.sequences:
                if self.verbose: 
                    print('seq', seq)
                
                if not seq.split('/')[-2].startswith('scene'):
                    continue  # skip sequences that start with 'scene', as they are not part of the dataset
                
                sub_scenes = os.listdir(seq)
                for sub_seq in sub_scenes:
                    if not os.path.isdir(os.path.join(seq, sub_seq)):
                        continue
                    
                    if self.verbose: 
                        print('sub seq', sub_seq)

                    rgb_path = os.path.join(seq, sub_seq, 'frames', 'Image', 'camera_0' )
                    depth_path = os.path.join(seq, sub_seq, 'frames', 'Depth', 'camera_0')
                    seg_mask_path = os.path.join(seq, sub_seq, 'frames', 'ObjectSegmentation', 'camera_0')
                    annotaions_file_path = os.path.join(seq, sub_seq, 'frames', 'camview', 'camera_0')
                    num_frames = len(glob.glob(os.path.join(rgb_path, '*.png')))

                    if num_frames < 24:
                        print(f"Skipping sequence {seq} with only {num_frames} frames.")
                        continue

                    new_sequence = list(len(self.full_idxs) + np.arange(num_frames))
                    old_sequence_length = len(self.full_idxs)
                    self.full_idxs.extend(new_sequence)
                    self.all_rgb_paths.extend(sorted(glob.glob(os.path.join(rgb_path, 'Image_*.png')))) 
                    self.all_depth_paths.extend(sorted(glob.glob(os.path.join(depth_path, 'Depth_*.npy'))))
                    self.all_seg_mask_paths.extend(sorted(glob.glob(os.path.join(seg_mask_path, 'ObjectSegmentation_*.npy'))))
                    seq_annotaions_path = sorted(glob.glob(os.path.join(annotaions_file_path, 'camview_*.npz')))
                    self.all_annotation_paths.extend(seq_annotaions_path)
               
                    N = len(self.full_idxs)
                    
                    assert len(self.all_rgb_paths) == N and \
                        len(self.all_seg_mask_paths) == N and \
                        len(self.all_annotation_paths) == N and \
                        len(self.all_depth_paths) == N, f"Number of images, depth maps, and annotations do not match in {seq}."
                    
                    extrinsics_seq = []  
                    #load intrinsics and extrinsics
                    for anno in seq_annotaions_path:
                        camera_info = np.load(anno)
                        pose = np.array(camera_info['T'],dtype=np.float32)
                        intrinsics = np.array(camera_info['K'],dtype=np.float32)
                        assert pose.shape == (4, 4), f"Pose shape mismatch in {anno}: {pose.shape}"
                        assert intrinsics.shape == (3, 3), f"Intrinsics shape mismatch in {anno}: {intrinsics.shape}"
                        self.all_extrinsic.extend([pose])
                        self.all_intrinsic.extend([intrinsics])
                        extrinsics_seq.append(pose)
                        
                    all_extrinsic_numpy = np.array(extrinsics_seq)
                    #compute ranking
                    ranking, dists = compute_ranking(all_extrinsic_numpy, lambda_t=1.0, normalize=True, batched=True)
                    ranking += old_sequence_length
                    for ind, i in enumerate(range(old_sequence_length, len(self.full_idxs))):
                        self.rank[i] = ranking[ind] 
                    
            # # 保存为 JSON 文件
            # os.makedirs(f'annotations/infinigen_annotations/{self.dset}', exist_ok=True)
            # self._save_paths_to_json(self.all_rgb_paths, f'annotations/infinigen_annotations/{self.dset}/rgb_paths.json')
            # self._save_paths_to_json(self.all_depth_paths, f'annotations/infinigen_annotations/{self.dset}/depth_paths.json')
            # self._save_paths_to_json(self.all_seg_mask_paths, f'annotations/infinigen_annotations/{self.dset}/seg_mask_paths.json')
            # joblib.dump(self.rank, f'annotations/infinigen_annotations/{self.dset}/rankings.joblib')
            # joblib.dump(self.all_extrinsic, f'annotations/infinigen_annotations/{self.dset}/extrinsics.joblib')
            # joblib.dump(self.all_intrinsic, f'annotations/infinigen_annotations/{self.dset}/intrinsics.joblib')
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))
        
    
    def _save_paths_to_json(self, paths, filename):
        path_dict = {i: path for i, path in enumerate(paths)}
        with open(filename, 'w') as f:
            json.dump(path_dict, f, indent=4)

    def __len__(self):
        return len(self.full_idxs)
    
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
            seg_mask_paths = [self.all_seg_mask_paths[i] for i in full_idx]
            
        else:
            full_idx = self.full_idxs[index]
            rgb_paths = [self.all_rgb_paths[full_idx]]
            depth_paths = [self.all_depth_paths[full_idx]]
            camera_pose_list = [self.all_extrinsic[full_idx]]
            intrinsics_list = [self.all_intrinsic[full_idx]]
            seg_mask_paths = [self.all_seg_mask_paths[full_idx]]

        views = []
        for i in range(num):
            impath = rgb_paths[i]
            depthpath = depth_paths[i]
            camera_pose = camera_pose_list[i]
            intrinsics = intrinsics_list[i]
            seg_mask_path = seg_mask_paths[i]

            # load image and depth
            rgb_image = Image.open(impath)
            rgb_image = rgb_image.convert("RGB")
            depthmap = np.load(depthpath)
            seg_mask = np.load(seg_mask_path)
            depthmap = threshold_depth_map(depthmap, max_percentile=98, min_percentile=2)
                        
            rgb_image, depthmap, seg_mask, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, seg_mask, resolution, rng=rng, info=impath)        
            
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                seg_mask=seg_mask,
                dataset=self.dataset_label,
                label=rgb_paths[i].split('/')[-3],
                instance=osp.split(rgb_paths[i])[1],
            ))

        return views
        
        
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
        seg_mask_list = []

        # Iterate over the views and concatenate each field
        for view in views:
            img_list.append(view['img'])
            depthmap_list.append(view['depthmap'][:, :, np.newaxis])
            camera_pose_list.append(view['camera_pose'][:3])
            camera_intrinsics_list.append(view['camera_intrinsics'])
            pts3d_list.append(view['world_coords_points'])
            true_shape_list.append(view['true_shape'])
            valid_mask_list.append(view['point_mask'])
            seg_mask_list.append(view['seg_mask']) 
            label_list.append(view['label'])
            instance_list.append(view['instance'])

        # Concatenate the lists along the first dimension (n)
        img = torch.stack(img_list)
        depthmap = np.stack(depthmap_list)
        camera_pose = np.stack(camera_pose_list)
        camera_intrinsics = np.stack(camera_intrinsics_list)
        pts3d = np.stack(pts3d_list)
        true_shape = np.array(true_shape_list)
        valid_mask = np.stack(valid_mask_list)    

        # 将seg_mask_list (N个view, 每个view为(H, W)或(H, W, 1)) 转为 (N, H, W, C)的mask集合，C为所有view中独一无二的id总数（去除背景0）
        # 1. 找到所有view中出现过的非0 id
        all_ids = set()
        for seg_mask in seg_mask_list:
            if seg_mask.ndim == 3 and seg_mask.shape[2] == 1:
                seg_mask = seg_mask[..., 0]
            unique_ids = np.unique(seg_mask)
            all_ids.update(set(unique_ids.tolist()))
        all_ids.discard(0)  # 去除背景
        all_ids = sorted(list(all_ids))
        id2idx = {id_: idx for idx, id_ in enumerate(all_ids)}
        C = len(all_ids)
        N = len(seg_mask_list)
        H, W = seg_mask_list[0].shape[:2]
        # 2. 构建(N, H, W, C)的mask
        instance_mask = np.zeros((N, H, W, C), dtype=np.uint8)
        for n, seg_mask in enumerate(seg_mask_list):
            if seg_mask.ndim == 3 and seg_mask.shape[2] == 1:
                seg_mask = seg_mask[..., 0]
            for id_ in np.unique(seg_mask):
                if id_ == 0: continue
                idx = id2idx[id_]
                instance_mask[n, :, :, idx] = (seg_mask == id_).astype(np.uint8)

        return dict(
                images=img, #(n, c, h, w)
                depth=depthmap, #(n, h, w, 1)
                extrinsic=camera_pose, #(n, 3, 4)
                intrinsic=camera_intrinsics, #(n, 3, 3)
                mask_gt=instance_mask, #(n, h, w)
                dataset=self.dataset_label,
                label=label_list,
                instance=instance_list,
                world_points=pts3d, #(n, h, w, 3)
                true_shape=true_shape,
                valid_mask=valid_mask,)

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
        views['extrinsic'] = closed_form_inverse_se3(poses)
        cam_size = max(auto_cam_size(poses), 0.25)
        for view_idx in n_views_list:
            pts3d = views['world_points'][view_idx]
            valid_mask = views['valid_mask'][view_idx]
            colors = rgb(views['images'][view_idx])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views['extrinsic'][view_idx],
                        focal=views['intrinsic'][view_idx][0, 0],
                        color=(255, 0, 0),
                        image=colors,
                        cam_size=cam_size)
        # return viz.show()
        viz.save_glb(f'infinigen_{dataset.dset}_views_{num_views}.glb')
        return

    dataset = Infinigen(
        dataset_location="datasets/infinigen",
        dset = '',
        use_cache = True,
        use_augs=use_augs, 
        top_k= 50,
        quick=False,
        verbose=True,
        resolution=(512,224), 
        seed = 777,
        load_mask=False,
        aug_crop=16,
        z_far = 200)

    dataset[(101,0,16)]
    print("Dataset loaded successfully.")
    # idx = random.randint(0, len(dataset)-1)
    visualize_scene((1000,0,num_views))
    # print(len(dataset))
