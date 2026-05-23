import torch
from PIL import Image
import os
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import json

def read_json(path):
    with open(path,'r',encoding='utf-8') as f:
        return json.load(f)


class OurDataset(Dataset):
    """
    ZJU-LEAPER 数据集用于自编码器训练

    """

    def __init__(self,
                 all_pics_root=r"/media/share/WX/Sync/ZJU-Leaper/Images",
                 all_pics_seg=r"/media/share/WX/Sync/ZJU-Leaper/ImageSets",
                 mask_root=r"/media/share/WX/Sync/ZJU-Leaper/Annotations/masks",
                 phase='Train',
                 divide_by_groups=True,
                 num_of_group=1,
                 divide_by_patterns=False,
                 num_of_pattern=1,
                 resize=256,
                 cropsize=256,
                 train_sample_ratio=0.7):
        self.all_pics_root = all_pics_root
        self.all_pics_seg = all_pics_seg
        self.mask_root = mask_root
        self.phase = phase
        self.resize = resize
        self.cropsize = cropsize

        # Step 1: Load raw file list based on config
        if divide_by_groups:
            json_file = read_json(os.path.join(all_pics_seg, 'Groups', f'group{num_of_group}.json'))
        elif divide_by_patterns:
            json_file = read_json(os.path.join(all_pics_seg, 'Patterns', f'pattern{num_of_pattern}.json'))
        else:
            raise ValueError("Either divide_by_groups or divide_by_patterns must be True")

        if self.phase == 'Train':
            raw_file_list = json_file["normal"]['train']
            if train_sample_ratio < 1.0:
                np.random.seed(42)
                sample_size = int(len(raw_file_list) * train_sample_ratio)
                raw_file_list = list(np.random.choice(raw_file_list, size=sample_size, replace=False))
        elif self.phase == 'Test':
            self.defect_pics_file = json_file['defect']['test']
            raw_file_list = json_file['defect']['test'] + json_file['normal']['test']
        elif self.phase == 'PreTest':
            raw_file_list = json_file["defect"]['train']
        else:
            raise ValueError('this phase was not include')

        # Step 2: Filter out all-black images
        self.file_list = []
        for fid in raw_file_list:
            img_path = os.path.join(self.all_pics_root, f'{fid}.jpg')
            # Check if image is all black
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            if img_array.max() > 0:  # Not all black
                self.file_list.append(fid)
            # else: skip this image

        # Optional: print how many were filtered
        num_filtered = len(raw_file_list) - len(self.file_list)
        if num_filtered > 0:
            print(f"[Warning] Filtered out {num_filtered} all-black images in phase '{phase}'.")

        # Define transforms
        self.transform_x = T.Compose([
            T.Resize(resize),
            T.CenterCrop(cropsize),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.transform_mask = T.Compose([
            T.Resize(resize, Image.NEAREST),
            T.CenterCrop(cropsize),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        '''
        train:
        返回正常样本,label,mask
        val:
        返回正常样本,label,mask
        test:
        返回混合样本,label,mask
        '''
        if self.phase=='Train':#训练和验证阶段只使用正常图片
            img_path=os.path.join(self.all_pics_root,f'{self.file_list[idx]}.jpg')
            image=Image.open(img_path).convert('RGB')
            x=self.transform_x(image)
            y=0
            mask=torch.zeros([1,self.cropsize,self.cropsize])
        elif self.phase=='Test':#测试阶段使用缺陷图(这里我们只使用了测试集的)以及正常图
            img_path = os.path.join(self.all_pics_root, f"{self.file_list[idx]}.jpg")
            image = Image.open(img_path).convert('RGB')
            x = self.transform_x(image)
            #判读图片是否是缺陷图
            if self.file_list[idx] in self.defect_pics_file:

                mask_path = os.path.join(self.mask_root, f"{self.file_list[idx]}.png")
                mask = Image.open(mask_path).convert('L')  # 注意：mask 通常是单通道！
                mask = self.transform_mask(mask)
                y = 1
            else:
                mask = torch.zeros([1,self.cropsize, self.cropsize])
                y=0
        # pretest phase we use train set defect pics finding the best threshold value
        elif self.phase=="PreTest":
            img_path = os.path.join(self.all_pics_root, f"{self.file_list[idx]}.jpg")
            image = Image.open(img_path).convert('RGB')
            x = self.transform_x(image)
            #判读图片是否是缺陷图
            mask_path = os.path.join(self.mask_root, f"{self.file_list[idx]}.png")
            mask = Image.open(mask_path).convert('L')  # 注意：mask 通常是单通道！
            mask = self.transform_mask(mask)
            y=1

        else:
            raise ValueError(f"Unexpected phase: {self.phase}. Expected one of: 'Train', 'Test', 'PreTest'")

        return x,y,mask
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import json
import random

class OurDataset_(Dataset):#会随机反转进行dataaugmentation
    """
    ZJU-LEAPER 数据集用于自编码器训练
    """

    def __init__(self,
                 all_pics_root=r"/media/share/WX/2_Dataset/ZJU-Leaper/Images",
                 all_pics_seg=r"/media/share/WX/2_Dataset/ZJU-Leaper/ImageSets",
                 mask_root=r"/media/share/WX/2_Dataset/ZJU-Leaper/Annotations/masks",
                 phase='Train',
                 divide_by_groups=True,
                 num_of_group=1,
                 divide_by_patterns=False,
                 num_of_pattern=1,
                 resize=256,
                 cropsize=256,
                 train_sample_ratio=0.7):
        self.all_pics_root = all_pics_root
        self.all_pics_seg = all_pics_seg
        self.mask_root = mask_root
        self.phase = phase
        self.resize = resize
        self.cropsize = cropsize

        # Step 1: Load raw file list based on config
        if divide_by_groups:
            json_file = read_json(os.path.join(all_pics_seg, 'Groups', f'group{num_of_group}.json'))
        elif divide_by_patterns:
            json_file = read_json(os.path.join(all_pics_seg, 'Patterns', f'pattern{num_of_pattern}.json'))
        else:
            raise ValueError("Either divide_by_groups or divide_by_patterns must be True")

        if self.phase == 'Train':
            raw_file_list = json_file["normal"]['train']
            if train_sample_ratio < 1.0:
                np.random.seed(42)
                sample_size = int(len(raw_file_list) * train_sample_ratio)
                raw_file_list = list(np.random.choice(raw_file_list, size=sample_size, replace=False))
        elif self.phase == 'Test':
            self.defect_pics_file = json_file['defect']['test']
            raw_file_list = json_file['defect']['test'] + json_file['normal']['test']
        elif self.phase == 'PreTest':
            raw_file_list = json_file["defect"]['train']
        else:
            raise ValueError('this phase was not include')

        # Step 2: Filter out all-black images
        self.file_list = []
        for fid in raw_file_list:
            img_path = os.path.join(self.all_pics_root, f'{fid}.jpg')
            img = Image.open(img_path).convert('RGB')
            img_array = np.array(img)
            if img_array.max() > 0:  # Not all black
                self.file_list.append(fid)

        num_filtered = len(raw_file_list) - len(self.file_list)
        if num_filtered > 0:
            print(f"[Warning] Filtered out {num_filtered} all-black images in phase '{phase}'.")

        # 定义基础变换（不含随机几何增强）
        self.to_tensor_norm = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.to_tensor_mask = T.ToTensor()

        # Resize and CenterCrop 参数
        self.resize_size = resize
        self.crop_size = cropsize

    def _transform_train(self, image, mask=None):
        """
        Apply synchronized data augmentation for training.
        image: PIL RGB
        mask: PIL L (grayscale), optional
        Returns: transformed image (tensor), mask (tensor, same size)
        """
        # Step 1: Resize
        image = TF.resize(image, self.resize_size)
        if mask is not None:
            mask = TF.resize(mask, self.resize_size, interpolation=TF.InterpolationMode.NEAREST)

        # Step 2: No random flip (removed)

        # Step 4: Center crop
        image = TF.center_crop(image, self.crop_size)
        if mask is not None:
            mask = TF.center_crop(mask, self.crop_size)

        # Step 5: ToTensor + Normalize
        image = self.to_tensor_norm(image)
        if mask is not None:
            mask = self.to_tensor_mask(mask)

        return image, mask

    def _transform_test(self, image, mask=None):
        """No augmentation, only deterministic transforms"""
        image = TF.resize(image, self.resize_size)
        image = TF.center_crop(image, self.crop_size)
        image = self.to_tensor_norm(image)

        if mask is not None:
            mask = TF.resize(mask, self.resize_size, interpolation=TF.InterpolationMode.NEAREST)
            mask = TF.center_crop(mask, self.crop_size)
            mask = self.to_tensor_mask(mask)

        return image, mask

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fid = self.file_list[idx]
        img_path = os.path.join(self.all_pics_root, f'{fid}.jpg')
        image = Image.open(img_path).convert('RGB')

        if self.phase == 'Train':
            x, _ = self._transform_train(image, mask=None)
            y = 0
            mask = torch.zeros(1, self.crop_size, self.crop_size)
            return x, y, mask

        elif self.phase == 'Test':
            # Test: apply deterministic transform
            x, _ = self._transform_test(image, mask=None)
            if fid in self.defect_pics_file:
                mask_path = os.path.join(self.mask_root, f"{fid}.png")
                mask_img = Image.open(mask_path).convert('L')
                _, mask = self._transform_test(image, mask=mask_img)  # 注意：这里 image 未被修改，仅用于尺寸参考；实际 mask 单独处理
                y = 1
            else:
                mask = torch.zeros(1, self.crop_size, self.crop_size)
                y = 0
            return x, y, mask

        elif self.phase == 'PreTest':
            x, _ = self._transform_test(image, mask=None)
            mask_path = os.path.join(self.mask_root, f"{fid}.png")
            mask_img = Image.open(mask_path).convert('L')
            _, mask = self._transform_test(image, mask=mask_img)
            y = 1
            return x, y, mask

        else:
            raise ValueError(f"Unexpected phase: {self.phase}")
