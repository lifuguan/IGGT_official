import sys
sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
import glob
import cv2
import random
from PIL import Image
import json
import joblib

from iggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from iggt.utils.geometry import depthmap_to_absolute_camera_coordinates, depth_to_world_coords_points, closed_form_inverse_se3
from iggt.datasets.base.base_stereo_view_dataset import is_good_type, view_name, transpose_to_landscape
from iggt.datasets.utils.image_ranking import compute_ranking
from iggt.datasets.utils.misc import threshold_depth_map

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

class Habitat(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='/mnt/juicefs/datasets/cut3r_processed/habitat_processed/hm3d',
                 use_cache = True,
                 dset='train',
                 use_augs=False,
                 top_k = 5,      
                 quick=False,
                 verbose=False,
                 *args, 
                 **kwargs
                 ):

        print('loading Habitat dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'Habitat'
        self.split = dset
        self.verbose = verbose
        self.top_k = top_k

        self.use_augs = use_augs
        self.dset = dset
        self.use_cache = use_cache

        
        self.full_idxs = []
        self.all_rgb_paths = []
        self.all_depth_paths = []
        self.all_normal_paths = []
        self.all_extrinsic = []
        self.all_intrinsic = []
        self.all_annotation_paths = []
        self.rank = dict()

        self.subdirs = []
        self.sequences = []
        self.subdirs.append(os.path.join(dataset_location, dset))

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*/")):
                seq_name = seq.split('/')[-1]
                self.sequences.append(seq)

        self.sequences = sorted(self.sequences)
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))
        
        ## load trajectories
        print('loading trajectories...')

        if quick:
           self.sequences = self.sequences[0:2] 
        
        if self.use_cache:
            dataset_location = '/mnt/juicefs/datasets/cut3r_processed/habitat_annotations/hm3d'
            all_rgb_paths_file = os.path.join(dataset_location, dset, 'rgb_paths.json')
            all_depth_paths_file = os.path.join(dataset_location, dset, 'depth_paths.json')
            all_annotation_paths_file = os.path.join(dataset_location, dset, 'annotation_paths.json')
            with open(all_rgb_paths_file, 'r', encoding='utf-8') as file:
                self.all_rgb_paths = json.load(file)
            with open(all_depth_paths_file, 'r', encoding='utf-8') as file:
                self.all_depth_paths = json.load(file)    
            with open(all_annotation_paths_file, 'r', encoding='utf-8') as file:
                self.all_annotation_paths = json.load(file)    
            self.all_rgb_paths = [self.all_rgb_paths[str(i)] for i in range(len(self.all_rgb_paths))]
            self.all_depth_paths = [self.all_depth_paths[str(i)] for i in range(len(self.all_depth_paths))]
            self.all_annotation_paths = [self.all_annotation_paths[str(i)] for i in range(len(self.all_annotation_paths))]
            self.full_idxs = list(range(len(self.all_rgb_paths)))
            self.rank = joblib.load(os.path.join(dataset_location, dset, 'rankings.joblib'))
            
            print('found %d frames in %s (dset=%s)' % (len(self.full_idxs), dataset_location, dset))
            
        else:
            for seq in self.sequences:
                if self.verbose: 
                    print('seq', seq)

                seq_name = seq.split('-')[-1][:-1] + ".basis"
                rgb_path = os.path.join(seq, seq_name)
                
                image_paths = sorted(glob.glob(os.path.join(rgb_path, '*.jpeg')))
                self.all_rgb_paths.extend(image_paths)
                
                num_frames = len(image_paths)
                new_sequence = list(len(self.full_idxs) + np.arange(num_frames))
                old_sequence_length = len(self.full_idxs)
                self.full_idxs.extend(new_sequence)
                
                self.all_depth_paths.extend(sorted(glob.glob(os.path.join(rgb_path, '*.exr'))))
                annotations = sorted(glob.glob(os.path.join(rgb_path, '*.json')))
                self.all_annotation_paths.extend(annotations)
                
                N = len(self.full_idxs)
                assert len(self.all_rgb_paths) == N and \
                    len(self.all_depth_paths) == N and \
                    len(self.all_annotation_paths) == N, f"Number of images, depth maps, and annotations do not match in {seq}."
                
                extrinsics = []
                intrinsics = []
                for anno in annotations:
                    with open(anno, 'r') as f:
                        camera_params = json.load(f)
                    intrinsic = np.float32(camera_params['camera_intrinsics'])
                    camera_pose = np.eye(4, dtype=np.float32)
                    camera_pose[:3, :3] = camera_params['R_cam2world']
                    camera_pose[:3, 3] = camera_params['t_cam2world']
                    extrinsics.append(camera_pose)
                    intrinsics.append(intrinsic)
                    
                self.all_extrinsic.extend(extrinsics)
                self.all_intrinsic.extend(intrinsics)
                ranking, dists = compute_ranking(extrinsics, lambda_t=1.0, normalize=True, batched=True)
                ranking = np.array(ranking, dtype=np.int32)
                ranking += old_sequence_length
                for ind, i in enumerate(range(old_sequence_length, len(self.full_idxs))):
                    self.rank[i] = ranking[ind]
                    
            # # 保存为 JSON 文件
            # self._save_paths_to_json(self.all_rgb_paths, f'/mnt/juicefs/datasets/cut3r_processed/habitat_annotations/hm3d/{dset}/rgb_paths.json')
            # self._save_paths_to_json(self.all_depth_paths, f'/mnt/juicefs/datasets/cut3r_processed/habitat_annotations/hm3d/{dset}/depth_paths.json')
            # self._save_paths_to_json(self.all_annotation_paths, f'/mnt/juicefs/datasets/cut3r_processed/habitat_annotations/hm3d/{dset}/annotation_paths.json')
            # joblib.dump(self.rank, f'/mnt/juicefs/datasets/cut3r_processed/habitat_annotations/hm3d/{dset}/rankings.joblib')
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
            top_k = self.top_k if len(self.rank[anchor_frame]) >= self.top_k else len(self.rank[anchor_frame])
            rest_frame = self.rank[anchor_frame][:top_k]
            rest_frame_indexs = random.sample(list(rest_frame), num-1)     
            full_idx = [anchor_frame] + rest_frame_indexs  # 用 list 替代 tuple
            
            rgb_paths = [self.all_rgb_paths[i] for i in full_idx]
            depth_paths = [self.all_depth_paths[i] for i in full_idx]
            annotation_paths = [self.all_annotation_paths[i] for i in full_idx]
            
    
        else:
            full_index = self.full_idxs[index]
            rgb_paths = [self.all_rgb_paths[full_index]]
            depth_paths = [self.all_depth_paths[full_index]]
            annotation_paths = [self.all_annotation_paths[full_index]]

        views = []
        for i in range(num):
            
            impath = rgb_paths[i]
            depthpath = depth_paths[i]
            annotation_path = annotation_paths[i]

            # load camera params
            with open(annotation_path, 'r') as f:
                camera_params = json.load(f)
                intrinsic = np.float32(camera_params['camera_intrinsics'])
                camera_pose = np.eye(4, dtype=np.float32)
                camera_pose[:3, :3] = camera_params['R_cam2world']
                camera_pose[:3, 3] = camera_params['t_cam2world']
                
            intrinsic = intrinsic
            extrinsic = camera_pose

            # load image and depth
            rgb_image = Image.open(impath)
            rgb_image = rgb_image.convert("RGB")
            depthmap = cv2.imread(depthpath, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
            
            depthmap = threshold_depth_map(depthmap, max_percentile=99, min_percentile=-1)   
            
            rgb_image, depthmap, intrinsic = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsic, resolution, rng=rng, info=impath)
            
  
            
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=extrinsic,
                camera_intrinsics=intrinsic,
                dataset=self.dataset_label,
                label=rgb_paths[i].split('/')[-3],
                instance=osp.split(rgb_paths[i])[1],
            ))
        return views
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if len(idx) == 2:
                idx, ar_idx = idx
                num = 1
                # the idx is specifying the aspect-ratio
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

            # view['pts3d'] = pts3d
            # view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']

            
            world_coords_points, cam_coords_points, point_mask = (
                depth_to_world_coords_points(view['depthmap'], view['camera_pose'], view["camera_intrinsics"], z_far = self.z_far)
            )
            view['world_coords_points'] = world_coords_points
            view['cam_coords_points'] = cam_coords_points
            view['point_mask'] = point_mask
            view['camera_pose'] = closed_form_inverse_se3(view['camera_pose'][None])[0]
            
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

if __name__ == "__main__":
    from iggt.datasets.base.base_stereo_view_dataset import view_name
    from iggt.viz import SceneViz, auto_cam_size
    from iggt.utils.image import rgb
    import gradio as gr
    import random

    dataset_location = '/mnt/juicefs/datasets/cut3r_processed/habitat_processed/hm3d'  # Change this to the correct path
    dset = 'val'
    use_augs = False
    num_views = 10
    n_views_list = range(num_views)
    quick = True  # Set to True for quick testing

    def visualize_scene(idx):
        views = dataset[idx]
        assert len(views['images']) == num_views, f"Expected {num_views} views, got {len(views)}"
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
        # os.makedirs('./tmp/po', exist_ok=True)
        return viz.show()

    dataset = Habitat(
        dataset_location=dataset_location,
        dset=dset,
        use_augs=use_augs,
        top_k = 50,
        quick=quick,
        verbose=True,
        resolution=[(518,154)], 
        aug_crop=16,
        aug_focal=1,
        z_far=80,
        seed=985)

    dataset[0]
    dataset[(0,0,2)]
    batch = dataset[(0, 0, 4)]
    print("Dataset loaded successfully.")
    # idx = random.randint(0, len(dataset)-1)
    # print(f"Visualizing scene {idx}...")
    # visualize_scene((0,0,num_views))
    # for i in range(len(dataset)):
    #     batch = dataset[(i, 0, 4)]
    #     if batch['intrinsic'][0,0,0] == 0:
    #         print("Error: Intrinsics are zero.")
