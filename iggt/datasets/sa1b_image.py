"""
Author: Wouter Van Gansbeke

Dataset class for COCO Panoptic Segmentation
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import json
import torch
import os.path as op
import numpy as np
import torch.utils.data as data
from PIL import Image
from typing import Optional, Any, Tuple
import random
from collections import defaultdict
import pycocotools.mask as mask_util
from io import BytesIO
from torchvision.transforms import Normalize, Resize, ToTensor
import torch.nn as nn

from detectron2.data import transforms as T
from detectron2.structures import Instances, Boxes, PolygonMasks,BoxMode
from detectron2.data import detection_utils as utils

from iggt.datasets.utils.tsv import TSVFile, img_from_base64, generate_lineidx, FileProgressingbar
from tqdm import tqdm
import base64

_M_RGB2YUV = [[0.299, 0.587, 0.114], [-0.14713, -0.28886, 0.436], [0.615, -0.51499, -0.10001]]
def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)


    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    return annotation
def convert_PIL_to_numpy(image, format):
    """
    Convert PIL image to numpy array of target format.

    Args:
        image (PIL.Image): a PIL image
        format (str): the format of output image

    Returns:
        (np.ndarray): also see `read_image`
    """
    if format is not None:
        # PIL only supports RGB, so convert to RGB and flip channels over below
        conversion_format = format
        if format in ["BGR", "YUV-BT.601"]:
            conversion_format = "RGB"
        image = image.convert(conversion_format)
    image = np.asarray(image)
    # PIL squeezes out the channel dimension for "L", so make it HWC
    if format == "L":
        image = np.expand_dims(image, -1)

    # handle formats not supported by PIL
    elif format == "BGR":
        # flip channels if needed
        image = image[:, :, ::-1]
    elif format == "YUV-BT.601":
        image = image / 255.0
        image = np.dot(image, np.array(_M_RGB2YUV).T)

    return image

def img_from_base64(imagestring):
    jpgbytestring = base64.b64decode(imagestring)
    image = BytesIO(jpgbytestring)
    image = Image.open(image).convert("RGB")
    return image
def check_image_size(dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            raise ValueError("Mismatched image shape{}, got {}, expect {}.".format(
                    (
                        " for image " + dataset_dict["file_name"]
                        if "file_name" in dataset_dict
                        else ""
                    ),
                    image_wh,
                    expected_wh,
                ) + " Please check the width/height in your annotation.")

    # To ensure bbox always remap to original image size
    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]

def load_sam_index(tsv_file):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.
    """
    dataset_dicts = []
    tsv_id = 0
    files = os.listdir(tsv_file)
    # process all tsvs in current local file
    start = int(os.getenv("SAM_SUBSET_START", "0"))
    end = int(os.getenv("SAM_SUBSET_END", "100"))
    # if len(files)>0 and 'part' in files[0]:  # for hgx
    files = [f for f in files if '.tsv' in f and int(f.split('.')[0].split('_')[-1])>=start and int(f.split('.')[0].split('_')[-1])<end]

    for tsv in files:
        if op.splitext(tsv)[1] == '.tsv':
            print('register tsv to create index', "tsv_id", tsv_id, tsv)
            lineidx = os.path.join(tsv_file, op.splitext(tsv)[0] + '.lineidx')
            line_name = op.splitext(tsv)[0] + '.lineidx'
            
            with open(lineidx, 'r') as fp:
                lines = fp.readlines()
                _lineidx = [int(i.strip().split()[0]) for i in lines]

            dataset_dict =[{'idx': (tsv_id, i)} for i in range(len(_lineidx))]
            dataset_dicts = dataset_dicts + dataset_dict
            tsv_id += 1
    return dataset_dicts

class MyPath(object):
    @staticmethod
    def db_root_dir(database='', prefix='./datasets/'):

        db_names = {'lvis', 'cityscapes','sa1b_tsv_chunks'}
        assert (database in db_names), 'Database {} not available.'.format(database)

        return os.path.join(prefix, database)

class SA1BDataset(data.Dataset):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """
    def __init__(
        self,
        config,
        is_train=True,
        image_format = "RGB",
        only_train_adaptor=False,
    ):
        self.only_train_adaptor = only_train_adaptor
        self.img_format = image_format
        self.is_train = is_train
        self.training = is_train
        self.augmentation = self.build_transform_gen(config)

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.sam2_transforms = Normalize(self.mean, self.std)

        _root = os.getenv("SAM_DATASETS", "no")
        totoal_images = 0
        self.current_tsv_id = -1
        tsv_file = f"{_root}/"
        self.tsv = {}
        print("start dataset mapper, get tsv_file from ", tsv_file)
        files = os.listdir(tsv_file)
        # print('files ', files)

        start = int(os.getenv("SAM_SUBSET_START", "0"))
        end = int(os.getenv("SAM_SUBSET_END", "100"))
        files = [f for f in files if '.tsv' in f and int(f.split('.')[0].split('_')[-1])>=start and int(f.split('.')[0].split('_')[-1])<end]
        self.total_tsv_num = len(files)
        for i, tsv in enumerate(tqdm(files, desc="Loading TSV files")):
            if tsv.split('.')[-1] == 'tsv':
                self.tsv[i] = TSVFile(f"{_root}/{tsv}")
                totoal_images += self.tsv[i].num_rows()
        print('totoal_images', totoal_images)

        tsv_id = 0
        self.dataset_dicts = []
        for tsv in files:
            if op.splitext(tsv)[1] == '.tsv':
                print('register tsv to create index', "tsv_id", tsv_id, tsv)
                lineidx = os.path.join(tsv_file, op.splitext(tsv)[0] + '.lineidx')
                line_name = op.splitext(tsv)[0] + '.lineidx'
                
                with open(lineidx, 'r') as fp:
                    lines = fp.readlines()
                    _lineidx = [int(i.strip().split()[0]) for i in lines]
                if self.training:
                    dataset_dict =[{'idx': (tsv_id, i)} for i in range(len(_lineidx))]
                else:
                    dataset_dict =[{'idx': (tsv_id, i)} for i in range(min(20,len(_lineidx)))]
                self.dataset_dicts = self.dataset_dicts + dataset_dict
                tsv_id += 1

        self.copy_flay = 0

    def build_transform_gen(self, cfg):
        """
        Create a list of default :class:`Augmentation` from config.
        Now it includes resizing and flipping.
        Returns:
            list[Augmentation]
        """

        cfg_input = cfg['INPUT']
        image_size = cfg_input['IMAGE_SIZE']
        min_scale = cfg_input['MIN_SCALE']
        max_scale = cfg_input['MAX_SCALE']

        augmentation = []

        if cfg_input['RANDOM_FLIP'] != "none":
            augmentation.append(
                T.RandomFlip(
                    horizontal=cfg_input['RANDOM_FLIP'] == "horizontal",
                    vertical=cfg_input['RANDOM_FLIP'] == "vertical",
                )
            )

        augmentation.extend([
            T.ResizeScale(
                min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
            ),
            T.FixedSizeCrop(crop_size=(image_size, image_size)),
        ])

        return augmentation
    
    def read_img(self, row):
        img = img_from_base64(row[-1])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        return img

    def read_json(self, row):
        anno=json.loads(row[1])
        return anno

    def __len__(self):
        return len(self.dataset_dicts)
    
    def __getitem__(self, index):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # self.init_copy()
        idx= self.dataset_dicts[index]['idx']
        # if idx == 0:   # read the next tsv file now
        current_tsv_id = idx[0]
        current_idx = idx[1]
        # print('before seek ', current_tsv_id, current_idx)
        row = self.tsv[current_tsv_id].seek(current_idx)
        # print('after seed')
        dataset_dict=self.read_json(row)
        if len(dataset_dict['annotations'])==0:
            print("encounter image with empty annotations, choose the first image in the first tsv file")
            current_tsv_id = 0
            current_idx = 0
            row = self.tsv[current_tsv_id].seek(current_idx)
        dataset_dict=self.read_json(row)
            
        image = self.read_img(row)
        image = utils.convert_PIL_to_numpy(image,"RGB")
        ori_shape = image.shape[:2]
        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        
        dataset_dict.update(dataset_dict['image'])
        for anno in dataset_dict['annotations']:
            anno["bbox_mode"] = BoxMode.XYWH_ABS
            anno["category_id"] = 0

        utils.check_image_size(dataset_dict, image)

        padding_mask = np.zeros(image.shape[:2])
        image, transforms = T.apply_transform_gens(self.augmentation, image)

        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)
        image_shape = image.shape[:2]

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = ToTensor()(np.ascontiguousarray(image))
        dataset_dict["sam_image"] = self.sam2_transforms(dataset_dict["image"])
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        
        # if self.only_train_adaptor is True:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                anno.pop("keypoints", None)
            mask_shape = ori_shape
            if len(dataset_dict['annotations'])>0 and 'segmentation' in dataset_dict['annotations'][0].keys() and 'size' in dataset_dict['annotations'][0]['segmentation'].keys():
                mask_shape = dataset_dict['annotations'][0]['segmentation']['size']
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape,mask_format='bitmask')
            point_coords=[]
            for ann in annos:
                assert len(ann['point_coords'])==1
                point_coords.extend(ann['point_coords'])
            point_coords=torch.as_tensor(point_coords)
            point_coords=torch.cat([point_coords-3.,point_coords+3.],dim=1)
            point_coords=transforms.apply_box(point_coords)
            point_coords=torch.as_tensor(point_coords,device=instances.gt_boxes.device)
            instances.point_coords=point_coords
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if not instances.has('gt_masks'): 
                instances.gt_masks = PolygonMasks([])  # for negative examples
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)

            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            # if hasattr(instances, 'gt_masks'):
            #     gt_masks = instances.gt_masks
            #     gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            #     instances.gt_masks = gt_masks
            # import ipdb; ipdb.set_trace()
            ####
            # instances.gt_classes = torch.tensor([])
            ###
            dataset_dict["instances"] = instances
        
        return dataset_dict