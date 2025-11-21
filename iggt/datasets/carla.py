import sys
sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
import glob
import torchvision.transforms as tvf
import random
from PIL import Image
import json
from copy import copy

from scipy.spatial.transform import Rotation as R
import cv2
from iggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from iggt.utils.geometry import depthmap_to_absolute_camera_coordinates
from iggt.datasets.base.base_stereo_view_dataset import is_good_type, view_name, transpose_to_landscape
from iggt.datasets.utils.image_ranking import compute_ranking
from dataset_preprocess.read_write_model import read_images_binary

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')

ToTensor = tvf.ToTensor()

def scale_intrinsics(K, old_size, new_size):
    new_width, new_height = new_size
    old_width, old_height = old_size
    scale_x = new_width / old_width
    scale_y = new_height / old_height

    K_scaled = copy(K) 
    K_scaled[0,0] *= scale_x
    K_scaled[0,2] *= scale_x
    K_scaled[1,1] *= scale_y
    K_scaled[1,2] *= scale_y
    
    return K_scaled

def pose_unreal2opencv(c2w_mat):
    translation = c2w_mat[:3, 3]
    rot = R.from_matrix(c2w_mat[:3, :3])
    rot_vec = rot.as_rotvec()

    rot_vec_new = rot_vec[[1, 2, 0]]
    rot_vec_new[0] *= -1
    rot_vec_new[2] *= -1

    rot = R.from_rotvec(rot_vec_new)
    
    translation_new = translation[[1, 2, 0]]
    translation_new[1] *= -1

    c2w_mat = np.eye(4)
    c2w_mat[:3, :3] = rot.as_matrix()
    c2w_mat[:3, 3] = translation_new

    rot = np.eye(4)
    rot[1,1]=-1
    rot[2, 2] = -1
    c2w_mat =  rot @ c2w_mat
    return c2w_mat

# 需要确保 PNG_SCALE 与保存时一致，假设 PNG_SCALE = 1000.0
MAX_DEPTH_METERS = 1000.0          # CARLA 默认深度上限
PNG_SCALE       = 65535.0 / MAX_DEPTH_METERS 

def read_depth_png(file_path: str) -> np.ndarray:
    """
    从 PNG 图片中读取深度数据并转换为深度矩阵（单位：米）
    
    :param file_path: PNG 图片文件路径
    :return: 深度矩阵，单位米
    """
    # 读取PNG图像，注意读取为灰度图（0 - 65535）
    depth_uint16 = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    # 确保读取的图像是uint16类型
    if depth_uint16 is None or depth_uint16.dtype != np.uint16:
        raise ValueError("无法读取深度PNG图像，确保图像为PNG格式且包含uint16深度数据")
    
    # 将 uint16 转换为深度值（单位：米），除以 PNG_SCALE 来还原
    depth_meters = depth_uint16.astype(np.float32) / PNG_SCALE
    
    return depth_meters

def read_params_from_json(root_path, files, if_scale=False, old_size=(1920,1080), new_size=(512, 288)):
    intrinsics = []
    extrinsics = []
    for parmas_file in files:
        file_path = os.path.join(root_path, parmas_file)
        # 读取 JSON
        with open(file_path, "r") as f:
            data = json.load(f)
        K = np.around(np.array(data["intrinsic"]["K"]),decimals=4)
        T = np.around(np.array(data["extrinsic"]["T"]),decimals=4)
        if if_scale:
            K = scale_intrinsics(K, old_size, new_size)
        T = pose_unreal2opencv(T)
        intrinsics.append(K)
        extrinsics.append(T)
    return intrinsics, extrinsics

class Carla(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='datasets/carla',
                 dset='',
                 use_augs=False,
                 top_k = 100,
                 quick=False,
                 verbose=False,
                 load_mask=True,
                 *args, 
                 **kwargs
                 ):

        print('loading Carla dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'Carla'
        self.split = dset
        self.top_k = top_k
        
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
           self.sequences = self.sequences[:1] 
        
        for seq in self.sequences:
            if self.verbose: 
                print('seq', seq)
            time_stamp = sorted(os.listdir(seq))
            time_stamp.pop()
            caminfo_path = os.path.join(seq,'params')
            caminfo_files = sorted(os.listdir(caminfo_path))
            intrinsics, extrinsics = read_params_from_json(root_path=caminfo_path, files=caminfo_files, if_scale=False)
            
            for time_index in time_stamp:
                rgb_path = os.path.join(seq, time_index, "rgb")
                depth_path = os.path.join(seq, time_index,'depth')

                num_frames = len(glob.glob(os.path.join(rgb_path, '*.png')))

                new_sequence = list(len(self.full_idxs) + np.arange(num_frames))
                old_sequence_length = len(self.full_idxs)
                self.full_idxs.extend(new_sequence)
                self.all_rgb_paths.extend(sorted(glob.glob(os.path.join(rgb_path, 'camera_*.png')))) 
                self.all_depth_paths.extend(sorted(glob.glob(os.path.join(depth_path, 'camera_*.png'))))
                N = len(self.full_idxs)
                
                assert len(self.all_rgb_paths) == N and \
                    len(self.all_depth_paths) == N, f"Number of images, depth maps, and annotations do not match in {seq}."
                
                #load intrinsics and extrinsics
                self.all_intrinsic.extend(np.array(intrinsics).astype(np.float32))
                self.all_extrinsic.extend(np.array(extrinsics).astype(np.float32))
                
                all_extrinsic_numpy = np.array(extrinsics)
                #compute ranking
                ranking, dists = compute_ranking(all_extrinsic_numpy, lambda_t=1.0, normalize=True, batched=True)
                ranking += old_sequence_length
                for ind, i in enumerate(range(old_sequence_length, len(self.full_idxs))):
                    self.rank[i] = ranking[ind] 
                
        print('loaded %d frames' % len(self.full_idxs))

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
            camera_pose_list = [self.all_extrinsic[i] for i in full_idx]
            intrinsics_list = [self.all_intrinsic[i] for i in full_idx]
            
        else:
            full_idx = self.full_idxs[index]
            rgb_paths = [self.all_rgb_paths[full_idx]]
            depth_paths = [self.all_depth_paths[full_idx]]
            camera_pose_list = [self.all_extrinsic[full_idx]]
            intrinsics_list = [self.all_intrinsic[full_idx]]

        views = []
        for i in range(num):
            impath = rgb_paths[i]
            depthpath = depth_paths[i]
            camera_pose = camera_pose_list[i]
            intrinsics = intrinsics_list[i]

            # load image and depth
            rgb_image = Image.open(impath)
            rgb_image = rgb_image.convert("RGB")
            depthmap = read_depth_png(depthpath)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)
            
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
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
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)
            
            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view['camera_intrinsics']

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
            pts3d_list.append(view['pts3d'])
            true_shape_list.append(view['true_shape'])
            valid_mask_list.append(view['valid_mask'])
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

    from iggt.viz import SceneViz, auto_cam_size
    from iggt.utils.image import rgb

    use_augs = False
    num_views = 8
    n_views_list = range(num_views)
    top_k = 10
    quick = False  # Set to True for quick testing


    def visualize_scene(idx):
        views = dataset[idx]
        # assert len(views['images']) == num_views, f"Expected {num_views} views, got {len(views)}"
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
        return viz.show()

    dataset = CarlaDUSt3R(
        dataset_location="datasets/carla",
        use_augs=use_augs, 
        top_k= 50,
        quick=False,
        verbose=True,
        resolution=(512,224), 
        seed = 777,
        load_mask=True,
        aug_crop=16)

    # dataset[(0,0,num_views)]
    # dataset[0]
    print("Dataset loaded successfully.")
    # idx = random.randint(0, len(dataset)-1)
    visualize_scene((10,0,num_views))
    # print(len(dataset))
