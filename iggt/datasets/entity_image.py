import os
import random
from typing import Optional

from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class RandomResizeOrCrop:
    def __init__(self, height, width, crop_prob=0.3):
        self.height = height
        self.width = width
        self.crop_prob = crop_prob

        self.random_crop = transforms.RandomResizedCrop(
            (self.height, self.width), scale=(0.8, 1.0)
        )
        self.simple_resize = transforms.Resize((self.height, self.width))

    def __call__(self, img):
        if random.random() < self.crop_prob:
            return self.random_crop(img)
        else:
            return self.simple_resize(img)


class EntityDataset(Dataset):
    def __init__(
        self,
        root_path: str = '/mnt/juicefs/datasets/cut3r_processed/entityseg/',
        annotation_json: str = '/mnt/juicefs/datasets/cut3r_processed/entityseg/annotations/entity_segmentation/entityseg_train.json',
        mask_subdir: str = 'rgb_masks',
        height: int = 336,
        width: int = 518,
        random_crop_prob: float = 0.3,  # 控制 crop 和 resize 的触发比例
    ) -> None:
        super().__init__()

        self.mask_subdir = mask_subdir
        self.root_path = root_path
        self.coco = COCO(annotation_json)
        self.img_ids = self.coco.getImgIds()

        self.height = height
        self.width = width

        self.transforms = transforms.Compose([
            RandomResizeOrCrop(height, width, crop_prob=random_crop_prob),
            transforms.ToTensor(),  # 将 PIL 图像转为 [C, H, W] Tensor，范围 [0, 1]
        ])

    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):
        while True:
            try:
                img_info = self.coco.loadImgs(self.img_ids[index])[0]
                file_name = img_info['file_name']
                image_path = os.path.join(self.root_path, file_name)

                # 获取 mask 路径，假设命名为 xxx_mask.png
                base_name = os.path.splitext(file_name)[0]
                mask_path = os.path.join(self.root_path, self.mask_subdir, f"{base_name}_mask.png")

                # 加载图像和 mask（RGB）
                image = Image.open(image_path).convert('RGB')
                mask = Image.open(mask_path).convert('RGB')

                # 同步变换（假设你有这个函数）
                image_tensor = self.transforms(image)
                mask_tensor = self.transforms(mask)
            
                return dict(images=image_tensor, masks=mask_tensor)
 
            except Exception as e:
                print(f"[Dataset Error] Failed loading: {file_name}")
                print(f"Error: {e}")
                # print(traceback.format_exc())  # 如果需要更详细的错误信息可取消注释

                # 随机选一个新的 index 重试
                index = random.randint(0, len(self.img_ids) - 1)

        return dict(images=image_tensor, masks=mask_tensor)
