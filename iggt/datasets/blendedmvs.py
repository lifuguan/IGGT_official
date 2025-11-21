# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed BlendedMVS
# dataset at https://github.com/YoYo000/BlendedMVS
# See datasets_preprocess/preprocess_blendedmvs.py
# --------------------------------------------------------
import sys
sys.path.append('.')
import os
import torch
import glob
import os.path as osp
import numpy as np
import torchvision.transforms as tvf
import random
from PIL import Image
import joblib
from tqdm import tqdm
import json

from iggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from iggt.utils.image import imread_cv2
from iggt.utils.geometry import depthmap_to_absolute_camera_coordinates, depth_to_world_coords_points, closed_form_inverse_se3
from iggt.datasets.base.base_stereo_view_dataset import is_good_type, view_name, transpose_to_landscape
from iggt.datasets.utils.image_ranking import compute_ranking
from iggt.datasets.utils.misc import threshold_depth_map

class BlendedMVS (BaseStereoViewDataset):
    """ Dataset of outdoor street scenes, 5 images each time
    """

    def __init__(self, 
                 dataset_location='datasets/processed_blendedmvs',
                 use_cache = True,
                 z_far = 1000,
                 use_augs=False,
                 dset='',
                 top_k = 256,
                 quick=False,
                 verbose=False,
                 *args,
                 **kwargs):
        
        print('loading blendedmvs dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'BlendedMVS'
        self.split = dset
        self.verbose = verbose
        self.top_k = top_k
        self.use_cache = use_cache
        self.z_far = z_far

        self.all_rgb_paths = []
        self.all_depth_paths = []
        self.all_annotations = []
        self.all_extrinsic = []
        self.all_intrinsic = []
        self.all_annotation_paths = []
        self.full_idxs = []
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

        if quick:
           self.sequences = self.sequences[0:1] 
        
        if self.use_cache:
            dataset_location = "annotations/blendedmvs_annotations"
            all_rgb_paths_file = os.path.join(dataset_location, dset, 'rgb_paths.json')
            all_depth_paths_file = os.path.join(dataset_location, dset, 'depth_paths.json')
            all_annotation_paths_file = os.path.join(dataset_location, dset, 'annotation_paths.json')
            with open(all_rgb_paths_file, 'r', encoding='utf-8') as file:
                self.all_rgb_paths = json.load(file)
            with open(all_depth_paths_file, 'r', encoding='utf-8') as file:
                self.all_depth_paths = json.load(file)    
            with open(all_annotation_paths_file, 'r', encoding='utf-8') as file:
                self.all_annotations = json.load(file)    
            self.all_rgb_paths = [self.all_rgb_paths[str(i)] for i in range(len(self.all_rgb_paths))]
            self.all_depth_paths = [self.all_depth_paths[str(i)] for i in range(len(self.all_depth_paths))]
            self.all_annotations = [self.all_annotations[str(i)] for i in range(len(self.all_annotations))]
            self.full_idxs = list(range(len(self.all_rgb_paths)))
            self.rank = joblib.load(os.path.join(dataset_location, dset, 'rankings.joblib'))
            
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))
            
        else:
            for seq in self.sequences:
                if self.verbose: 
                    print('seq', seq)

                rgb_path = seq 
                depth_path = seq 
                caminfo_path = seq 
                num_frames = len(glob.glob(os.path.join(rgb_path, '*.jpg')))

                if num_frames < 24:
                    print(f"Skipping sequence {seq} with only {num_frames} frames.")
                    continue
                
                new_sequence = list(len(self.full_idxs) + np.arange(num_frames))
                old_sequence_length = len(self.full_idxs)
                self.full_idxs.extend(new_sequence)
                self.all_rgb_paths.extend(sorted(glob.glob(os.path.join(rgb_path, '*.jpg')))) 
                self.all_depth_paths.extend(sorted(glob.glob(os.path.join(depth_path, '*.exr'))))
                self.all_annotations.extend(sorted(glob.glob(os.path.join(caminfo_path, '*.npz'))))
                N = len(self.full_idxs)
                assert len(self.all_rgb_paths) == N and \
                    len(self.all_depth_paths) == N and \
                    len(self.all_annotations) == N, f"Number of images, depth maps, and annotations do not match in {seq}."
                annotations = sorted(glob.glob(os.path.join(caminfo_path, '*.npz')))
                extrinsics_seq = []
                for anno in annotations:
                    camera_params = np.load(anno)
                    camera_pose = np.eye(4, dtype=np.float32)
                    camera_pose[:3, :3] = camera_params['R_cam2world']
                    camera_pose[:3, 3] = camera_params['t_cam2world']
                    intrinsics = np.float32(camera_params['intrinsics'])
                    self.all_extrinsic.extend([camera_pose])
                    self.all_intrinsic.extend([intrinsics])
                    extrinsics_seq.append(camera_pose)
                all_extrinsic_numpy = np.array(extrinsics_seq)
                
                ranking, dists = compute_ranking(all_extrinsic_numpy, lambda_t=1.0, normalize=True, batched=True)
                ranking = np.array(ranking, dtype=np.int32)
                ranking += old_sequence_length
                for ind, i in enumerate(range(old_sequence_length, len(self.full_idxs))):
                    self.rank[i] = ranking[ind] 
            
                    # # 保存为 JSON 文件
            # os.makedirs(f'annotations/blendedmvs_annotations', exist_ok=True)
            # self._save_paths_to_json(self.all_rgb_paths, f'annotations/blendedmvs_annotations/rgb_paths.json')
            # self._save_paths_to_json(self.all_depth_paths, f'annotations/blendedmvs_annotations/depth_paths.json')
            # self._save_paths_to_json(self.all_annotations, f'annotations/blendedmvs_annotations/annotation_paths.json')
            # joblib.dump(self.rank, f'annotations/blendedmvs_annotations/rankings.joblib')
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
            rest_frame_indexs = random.sample(list(rest_frame), num-1)     
            full_idx = [anchor_frame] + rest_frame_indexs  
            
            rgb_paths = [self.all_rgb_paths[i] for i in full_idx]
            depth_paths = [self.all_depth_paths[i] for i in full_idx]
            annotations = [self.all_annotations[i] for i in full_idx]
            
            
        else:
            full_idx = self.full_idxs[index]
            rgb_paths = [self.all_rgb_paths[full_idx]]
            depth_paths = [self.all_depth_paths[full_idx]]
            annotations = [self.all_annotations[full_idx]]


        views = []

        for i in range(num):
            impath = rgb_paths[i]
            depthpath = depth_paths[i]
            annotation = annotations[i]
            
            camera_params = np.load(annotation)
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = camera_params['R_cam2world']
            camera_pose[:3, 3] = camera_params['t_cam2world']
            intrinsics = np.float32(camera_params['intrinsics'])
            
            rgb_image = Image.open(impath)
            rgb_image = rgb_image.convert("RGB")
            depthmap = imread_cv2(depthpath)

            depthmap = threshold_depth_map(depthmap, max_percentile=98, min_percentile=-1)
            
            image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng, info=impath)    
            
            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=rgb_paths[i].split('/')[-2],
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
            # pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            # view['world_coords_points'] = pts3d
            # view['point_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)
            
            
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

        # Concatenate the lists along the first dimension (n)
        img = torch.stack(img_list)
        depthmap = np.stack(depthmap_list)
        camera_pose = np.stack(camera_pose_list)
        camera_intrinsics = np.stack(camera_intrinsics_list)
        pts3d = np.stack(pts3d_list)
        true_shape = np.array(true_shape_list)
        valid_mask = np.stack(valid_mask_list)    

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
                valid_mask=valid_mask,)

if __name__ == '__main__':
    from iggt.viz import SceneViz, auto_cam_size
    from iggt.utils.image import rgb


    num_views = 12
    use_augs = False
    n_views_list = range(num_views)

    def visualize_scene(idx):
        views = dataset[idx]
        viz = SceneViz()
        poses = views['extrinsic']
        cam_size = max(auto_cam_size(poses), 0.25)
        for view_idx in n_views_list:
            pts3d = views['world_points'][view_idx]
            valid_mask = views['valid_mask'][view_idx]
            colors = rgb(views['images'][view_idx])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=np.vstack((views['extrinsic'][view_idx],np.array([0, 0, 0, 1]))),
                            focal=views['intrinsic'][view_idx][0, 0],
                            color=(255, 0, 0),
                            image=colors,
                            cam_size=cam_size)
        # return viz.show()
        viz.save_glb('blendedmvs_scene.glb')
        return

    dataset = BlendedMVS(
        dataset_location="datasets/processed_blendedmvs",
        use_cache = True,
        dset='',
        use_augs=use_augs,
        top_k= 100,
        quick=False,
        verbose=True,
        resolution=(512,384), 
        aug_crop=16)
    
    dataset[0]
    # for i in tqdm(np.linspace(0, len(dataset), 10000)):
    #     try:
    #         dataset[i]
    #     except:
    #         print(f"Error processing index {i}, skipping...")
    #         continue
    # visualize_scene((1000,0,num_views))
    print("ok")