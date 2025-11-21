import sys
sys.path.append('.')
import os
import torch
import numpy as np
import os.path as osp
import glob
import PIL.Image
import torchvision.transforms as tvf
import random
from PIL import Image

from iggt.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from iggt.utils.geometry import depthmap_to_absolute_camera_coordinates
from iggt.datasets.base.base_stereo_view_dataset import is_good_type, view_name, transpose_to_landscape
from iggt.datasets.utils.image_ranking import compute_ranking

np.random.seed(125)
torch.multiprocessing.set_sharing_strategy('file_system')
TAG_FLOAT = 202021.25

ToTensor = tvf.ToTensor()

def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))
    return depth

def cam_read(filename):
    """ Read camera data, return (M,N) tuple.
    
    M is the intrinsic matrix, N is the extrinsic matrix, so that

    x = M*N*X,
    with x being a point in homogeneous image pixel coordinates, X being a
    point in homogeneous world coordinates.
    """
    f = open(filename,'rb')
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' cam_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    M = np.fromfile(f,dtype='float64',count=9).reshape((3,3))
    N = np.fromfile(f,dtype='float64',count=12).reshape((3,4))
    return M,N

class SintelDUSt3R(BaseStereoViewDataset):
    def __init__(self,
                 dataset_location='data/sintel/training',
                 dset='clean',
                 use_augs=False,
                 top_k = 100,
                 quick=False,
                 verbose=False,
                 load_dynamic_mask=True,
                 *args, 
                 **kwargs
                 ):

        print('loading sintel dataset...')
        super().__init__(*args, **kwargs)
        self.dataset_label = 'sintel'
        self.split = dset
        self.top_k = top_k
        
        self.verbose = verbose
        self.load_dynamic_mask = load_dynamic_mask

        self.use_augs = use_augs
        self.dset = dset

        self.full_idxs = []
        self.all_rgb_paths = []
        self.all_depth_paths = []
        self.all_traj_paths = []
        self.all_extrinsic = []
        self.all_intrinsic = []
        self.all_annotation_paths = []
        self.all_dynamic_mask_paths = []
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
           self.sequences = self.sequences[1:2] 
        
        for seq in self.sequences:
            if self.verbose: 
                print('seq', seq)

            rgb_path = seq
            depth_path = seq.replace(dset,'depth')
            caminfo_path = seq.replace(dset,'camdata_left')
            dynamic_mask_path = seq.replace(dset,'dynamic_label_perfect')
            num_frames = len(glob.glob(os.path.join(rgb_path, '*.png')))

            new_sequence = list(len(self.full_idxs) + np.arange(num_frames))
            old_sequence_length = len(self.full_idxs)
            self.full_idxs.extend(new_sequence)
            self.all_rgb_paths.extend(sorted(glob.glob(os.path.join(rgb_path, 'frame_*.png')))) 
            self.all_depth_paths.extend(sorted(glob.glob(os.path.join(depth_path, 'frame_*.dpt'))))
            self.all_annotation_paths.extend(sorted(glob.glob(os.path.join(caminfo_path, 'frame_*.cam'))))
            self.all_dynamic_mask_paths.extend(sorted(glob.glob(os.path.join(dynamic_mask_path, 'frame_*.png'))))
            N = len(self.full_idxs)
            self.all_dynamic_mask_paths.extend([None for i in range(N - len(self.all_dynamic_mask_paths))])
            assert len(self.all_rgb_paths) == N and \
                   len(self.all_depth_paths) == N and \
                   len(self.all_annotation_paths) == N and \
                   len(self.all_dynamic_mask_paths) == N, f"Number of images, depth maps, and annotations do not match in {seq}."
            annotations = sorted(glob.glob(os.path.join(caminfo_path, 'frame_*.cam')))
            extrinsics_seq = []
            for anno in annotations:
                intrinsics, extrinsics = cam_read(anno)
                intrinsics, extrinsics = np.array(intrinsics, dtype=np.float32), np.array(extrinsics, dtype=np.float32)
                R = extrinsics[:3,:3]
                t = extrinsics[:3,3]
                camera_pose = np.eye(4, dtype=np.float32)
                camera_pose[:3,:3] = R.T
                camera_pose[:3,3] = -R.T @ t
                self.all_extrinsic.extend([camera_pose])
                self.all_intrinsic.extend([intrinsics])
                extrinsics_seq.append(camera_pose)
            all_extrinsic_numpy = np.array(extrinsics_seq)
            
            ranking, dists = compute_ranking(all_extrinsic_numpy, lambda_t=1.0, normalize=True, batched=True)
            ranking += old_sequence_length
            for ind, i in enumerate(range(old_sequence_length, len(self.full_idxs))):
                self.rank[i] = ranking[ind] 
                
    

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
            dynamic_mask_paths = [self.all_dynamic_mask_paths[i] for i in full_idx]
            
        else:
            full_idx = self.full_idxs[index]
            rgb_paths = [self.all_rgb_paths[full_idx]]
            depth_paths = [self.all_depth_paths[full_idx]]
            camera_pose_list = [self.all_extrinsic[full_idx]]
            intrinsics_list = [self.all_intrinsic[full_idx]]
            dynamic_mask_paths = [self.all_dynamic_mask_paths[full_idx]]

        views = []
        for i in range(num):
            impath = rgb_paths[i]
            depthpath = depth_paths[i]
            dynamic_mask_path = dynamic_mask_paths[i]
            camera_pose = camera_pose_list[i]
            intrinsics = intrinsics_list[i]

            # load image and depth
            rgb_image = Image.open(impath)
            rgb_image = rgb_image.convert("RGB")
            depthmap = depth_read(depthpath)

            # load dynamic mask
            if dynamic_mask_path is not None and os.path.exists(dynamic_mask_path):
                dynamic_mask = PIL.Image.open(dynamic_mask_path).convert('L')
                dynamic_mask = ToTensor(dynamic_mask).sum(0).numpy()
                _, dynamic_mask, _ = self._crop_resize_if_necessary(
                rgb_image, dynamic_mask, intrinsics, resolution, rng=rng, info=impath)
                dynamic_mask = dynamic_mask > 0.5
                assert not np.all(dynamic_mask), f"Dynamic mask is all True for {impath}"
            else:
                dynamic_mask = np.ones((resolution[1],resolution[0]), dtype=bool)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)
            
            if self.load_dynamic_mask:
                views.append(dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
                    camera_intrinsics=intrinsics,
                    dataset=self.dataset_label,
                    label=rgb_paths[i].split('/')[-2],
                    instance=osp.split(rgb_paths[i])[1],
                    dynamic_mask=dynamic_mask,
                ))
            else:
                views.append(dict(
                    img=rgb_image,
                    depthmap=depthmap,
                    camera_pose=camera_pose,
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
        dynamic_mask_list = []
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
            dynamic_mask_list.append(view['dynamic_mask'])
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
        dynamic_mask = np.stack(dynamic_mask_list)
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
                dynamic_mask=dynamic_mask,
                world_points=pts3d, #(n, h, w, 3)
                true_shape=true_shape,
                valid_mask=valid_mask,)

if __name__ == "__main__":

    from iggt.viz import SceneViz, auto_cam_size
    from iggt.utils.image import rgb

    use_augs = False
    num_views = 24
    n_views_list = range(num_views)
    top_k = 100
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

    dataset = SintelDUSt3R(
        dataset_location="datasets/sintel/training",
        use_augs=use_augs, 
        top_k= top_k,
        quick=quick,
        verbose=False,
        resolution=(512,224), 
        seed = 777,
        aug_crop=16)

    dataset[(0,0,num_views)]
    print("Dataset loaded successfully.")
    # idx = random.randint(0, len(dataset)-1)
    # visualize_scene((1,num_views,0))
    # print(len(dataset))
