# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import cv2
import git
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from neuralfeels.datasets import redwood_depth_noise_model as noise_model
import torchvision.transforms as T

root = git.Repo(".", search_parent_directories=True).working_tree_dir
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop
import random
from neuralfeels.contrib.tactile_transformer.custom_augmentation import ToMask
from PIL import Image

class RandomAugment:
    def __init__(self, p_flip=0.5, p_crop=0.3, p_rot=0.2, resize=[224,224]):
        self.p_flip = p_flip
        self.p_crop = p_crop
        self.p_rot = p_rot
        self.target_size = resize

    def __call__(self, image, depth, mask):
        # 1) Horizontal flip
        if random.random() < self.p_flip:
            image = TF.hflip(image)
            depth = TF.hflip(depth)
            mask = TF.hflip(mask)

        # 2) Random crop
        if random.random() < self.p_crop:
            i, j, h, w = RandomCrop.get_params(image, output_size=(image.shape[1] - 20, image.shape[2] - 20))
            image = TF.crop(image, i, j, h, w)
            depth = TF.crop(depth, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

        # 3) Random rotation
        if random.random() < self.p_rot:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)
            depth = TF.rotate(depth, angle)
            mask = TF.rotate(mask, angle)
        image = TF.resize(image, self.target_size)
        depth = TF.resize(depth, self.target_size)
        mask  = TF.resize(mask, self.target_size)
        return image, depth, mask


def get_transforms(config):
    im_size = tuple(config['Dataset']['transforms']['resize'])
    transform_image = transforms.Compose([
        transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_depth = transforms.Compose([
        transforms.Resize(im_size),
        transforms.Grayscale(num_output_channels=1) ,
        transforms.ToTensor()
    ])
    transform_seg = transforms.Compose([
        transforms.Resize(im_size, interpolation=transforms.InterpolationMode.NEAREST),
        ToMask(config['Dataset']['classes']),
    ])
    return transform_image, transform_depth, transform_seg

class VisionDataset(torch.utils.data.Dataset):
    """Realsense data loader for neuralfeels dataset"""

    def __init__(
        self,
        root_dir: str,
        gt_seg: bool,
        sim_noise_iters: float,
        col_ext: str = ".jpg",
    ):
        # pre-load depth data
        depth_file = os.path.join(root_dir, "depth.npz")
        depth_loaded = np.load(depth_file, fix_imports=True, encoding="latin1")
        self.depth_data = depth_loaded["depth"]
        self.depth_scale = depth_loaded["depth_scale"]
        self.depth_data = self.depth_data.astype(np.float32)
        self.depth_data = self.depth_data * self.depth_scale

        if sim_noise_iters > 0:
            # add noise to the clean simulation depth data
            # At 1 meter distance an accuracy of 2.5 mm to 5 mm  (https://github.com/IntelRealSense/librealsense/issues/7806).
            # We operate at roughly 0.5 meter, we empirally pick 2mm as the noise std.
            # Adding the noise here allows us to ablate the effect of depth noise on the performance of the system.
            self.dist_model = np.load(
                os.path.join(root, "data", "feelsight", "redwood-depth-dist-model.npy")
            )
            self.dist_model = self.dist_model.reshape(80, 80, 5)
            for i, depth in enumerate(tqdm(self.depth_data)):
                depth = noise_model._simulate(-depth, self.dist_model, sim_noise_iters)
                self.depth_data[i, :, :] = -depth

        self.rgb_dir = os.path.join(root_dir, "image")
        self.seg_dir = os.path.join(root_dir, "seg")
        self.col_ext = col_ext
        self.gt_seg = gt_seg

    def __len__(self):
        return len(os.listdir(self.rgb_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb_file = os.path.join(self.rgb_dir, f"{idx}" + self.col_ext)
        image = cv2.imread(rgb_file)
        depth = self.depth_data[idx]

        if self.gt_seg:
            mask = self.get_gt_seg(idx)
            depth = depth * mask  # mask depth with gt segmentation

        return image, depth

    def get_avg_seg_area(self):
        """
        Returns the average segmentation area of the dataset
        """
        seg_area = 0.0
        for i in range(len(self)):
            mask = self.get_gt_seg(i)
            seg_area += mask.sum() / mask.size
        seg_area /= len(self)
        return seg_area

    def get_gt_seg(self, idx: int):
        """
        Returns a binary mask of the segmentation ground truth
        """
        seg_file = os.path.join(self.seg_dir, f"{idx}" + self.col_ext)
        mask = cv2.imread(seg_file, 0).astype(np.int64)
        # round every pixel to either 0, 255/2, 255
        mask = np.round(mask / 127.5) * 127.5
        # check if there exists three classes, if not return empty mask
        if np.unique(mask).size != 3:
            mask = np.zeros_like(mask)
        else:
            mask = mask == 255
        return mask


class TactileDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        gt_depth: bool,
        col_ext: str = ".jpg",
        resize = (224,224),
        cfg = None
    ):
        """DIGIT dataset loader for neuralfeels dataset"""
        self.rgb_dir = os.path.join(root_dir, "image")
        self.depth_dir = os.path.join(root_dir, "depth")
        self.mask_dir = os.path.join(root_dir, "mask")
        self.col_ext = col_ext
        self.gt_depth = gt_depth

        # 按focusondepth配置
        self.augment = RandomAugment(p_flip=0.5, p_crop=0.3, p_rot=0.2)

        if resize != None:
            self.image_size = resize
            print(f"images in dataset will resize to {resize}")
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize(resize),
                T.ToTensor(),  # 输出 float32, 范围 [0,1]
            ])
        
        self.transform_image, self.transform_depth, self.transform_seg = get_transforms(cfg)
    def __len__(self):
        return len(os.listdir(self.rgb_dir))

    def __getitem__(self, idx):
        # 旧版
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # rgb_file = os.path.join(self.rgb_dir, f"{idx:05d}" + self.col_ext)
        # image = cv2.imread(rgb_file)

        # depth = None
        # if self.gt_depth:
        #     depth_file = os.path.join(self.depth_dir, f"{idx:05d}" + self.col_ext)
        #     mask_file = os.path.join(self.mask_dir, f"{idx:05d}" + self.col_ext)
        #     depth = cv2.imread(depth_file, 0).astype(np.int64)

        #     depth[depth < 0] = 0

        #     mask = cv2.imread(mask_file, 0).astype(np.int64)
        #     mask = mask > 255 / 2
        #     if mask.sum() / mask.size < 0.01:
        #         # tiny mask, ignore
        #         mask *= False

        #     depth = depth * mask  # apply contact mask
        #     # if idx < 5:        # 只看前几张
        #     #     print("depth min/max:", depth.min(), depth.max())
        # # image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.
        # return image, depth

        # 新版 集成了原tensor_loader的归一化 , resize, 及dummy segmentation包装功能
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # H, W = self.image_size
        # name = f"{idx:05d}{self.col_ext}"
        # rgb_path   = os.path.join(self.rgb_dir,   name)
        # depth_path = os.path.join(self.depth_dir, name)
        # mask_path  = os.path.join(self.mask_dir,  name)

        # # 2) 读 RGB + resize + to tensor
        # img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        # if img is None:
        #     raise FileNotFoundError(f"RGB not found: {rgb_path}")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        # image_tensor = torch.from_numpy(img).permute(2,0,1).float().div(255.0)  # (3,H,W)

        # # 3) 读 Depth + mask 后处理
        # depth_np = np.zeros((H, W), dtype=np.float32)
        # if self.gt_depth:
        #     raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        #     if raw is None:
        #         raise FileNotFoundError(f"Depth not found: {depth_path}")
        #     raw = raw.astype(np.int64)
        #     raw[raw < 0] = 0

        #     m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        #     if m is None:
        #         raise FileNotFoundError(f"Mask not found: {mask_path}")
        #     m = (m.astype(np.int64) > 127)  # bool mask
        #     # 如果接触区域太小，就当全无接触
        #     if m.sum() / m.size < 0.01:
        #         m[:] = False

        #     masked = raw * m
        #     resized = cv2.resize(masked.astype(np.float32), (W, H),
        #                          interpolation=cv2.INTER_NEAREST)
        #     depth_np = resized / 255.0

        # depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).float()  # (1,H,W)

        # # 4) 读 Mask → segmentation label  
        # #    0=background, 1=contact
        # m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # if m is None:
        #     raise FileNotFoundError(f"Mask not found for segmentation: {mask_path}")
        # m = (m.astype(np.int64) > 127).astype(np.uint8)
        # m_resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        # seg_tensor = torch.from_numpy(m_resized).long().unsqueeze(0)  # (1,H,W)
        # # 数据增强
        # # image_tensor, depth_tensor, mask_tensor = self.augment(image_tensor, depth_tensor, seg_tensor)
        # return image_tensor, depth_tensor, seg_tensor

        # 最新版  
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. 读取并转换 RGB 图像
        rgb_file = os.path.join(self.rgb_dir, f"{idx:05d}" + self.col_ext)
        image = cv2.imread(rgb_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = self.transform_image(image)

        depth = None
        if self.gt_depth:
            # 2. 读取并转换 depth（先不转换为 Image）
            depth_file = os.path.join(self.depth_dir, f"{idx:05d}" + self.col_ext)
            mask_file = os.path.join(self.mask_dir, f"{idx:05d}" + self.col_ext)

            raw_depth = cv2.imread(depth_file, 0).astype(np.uint8)  # ✅ 转为 uint8，PIL 支持
            raw_mask = cv2.imread(mask_file, 0).astype(np.uint8)

            depth = self.transform_depth(Image.fromarray(raw_depth))       # Tensor: (1,H,W)
            mask = self.transform_seg(Image.fromarray(raw_mask))           # Tensor: (1,H,W) or (H,W)

            # 3. 处理 mask（确保是 bool 型，0/1）
            mask = (mask > 0).float()
            if mask.sum() / mask.numel() < 0.01:
                mask.zero_()  # 如果太小则全 0

            # 4. 应用 mask 到 depth
            # depth = depth * mask

        # 5. segmentation（不依赖于 gt_depth）
        segmentation_file = os.path.join(self.mask_dir, f"{idx:05d}" + self.col_ext)
        segmentation = self.transform_seg(Image.open(segmentation_file))

        return image, depth, segmentation
                
        # 官方版 不可用 接口和trainer对不上
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # rgb_file = os.path.join(self.rgb_dir, f"{idx:05d}" + self.col_ext)
        # image = cv2.imread(rgb_file)
        # image = self.transform_image(image)
        # depth = None
        # if self.gt_depth:
        #     depth_file = os.path.join(self.depth_dir, f"{idx:05d}" + self.col_ext)
        #     mask_file = os.path.join(self.mask_dir, f"{idx:05d}" + self.col_ext)
        #     depth = cv2.imread(depth_file, 0).astype(np.int64)
        #     depth = self.transform_depth(depth)
        #     depth[depth < 0] = 0

        #     mask = cv2.imread(mask_file, 0).astype(np.int64)
        #     mask = mask > 255 / 2
        #     if mask.sum() / mask.size < 0.01:
        #         # tiny mask, ignore
        #         mask *= False

        #     depth = depth * mask  # apply contact mask
        
        # segmentation_file = os.path.join(self.mask_dir, f"{idx:05d}" + self.col_ext)
        # segmentation = cv2.imread(segmentation_file, 0).astype(np.int64)

        # return image, depth, segmentation

        # 最新版 不返回masked depth 和 seg 而是raw depth 和mask
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        # H, W = self.image_size
        # name = f"{idx:05d}{self.col_ext}"
        # rgb_path   = os.path.join(self.rgb_dir,   name)
        # depth_path = os.path.join(self.depth_dir, name)
        # mask_path  = os.path.join(self.mask_dir,  name)

        # # 1) 读 RGB + resize + to tensor
        # img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        # if img is None:
        #     raise FileNotFoundError(f"RGB not found: {rgb_path}")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        # image_tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)  # (3,H,W)

        # # 2) 读 Depth（保留原始值，0～255） + resize
        # depth_np = np.zeros((H, W), dtype=np.float32)
        # if self.gt_depth:
        #     raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        #     if raw is None:
        #         raise FileNotFoundError(f"Depth not found: {depth_path}")
        #     raw = raw.astype(np.float32)
        #     raw[raw < 0] = 0  # 清理负值
        #     resized_depth = cv2.resize(raw, (W, H), interpolation=cv2.INTER_NEAREST)
        #     depth_np = resized_depth / 255.0  # 归一化

        # depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).float()  # (1,H,W)

        # # 3) 读接触 Mask（0/1），resize 后转为 tensor
        # m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # if m is None:
        #     raise FileNotFoundError(f"Mask not found: {mask_path}")
        # m = (m.astype(np.int64) > 127).astype(np.uint8)  # binary mask
        # # 如果 mask 面积太小，设置为全 0
        # if m.sum() / m.size < 0.01:
        #     m[:] = 0
        # m_resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        # mask_tensor = torch.from_numpy(m_resized).unsqueeze(0).long()  # (1,H,W), 0/1

        # # 4) 数据增强（保持原始结构）
        # image_tensor, depth_tensor, mask_tensor = self.augment(image_tensor, depth_tensor, mask_tensor)

        # return image_tensor, depth_tensor, mask_tensor

        # focus on depth版 测试中 TODO: 修bug
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # image = self.transform_image(Image.open(os.path.join(self.rgb_dir, f"{idx:05d}" + self.col_ext)))
        # depth = self.transform_depth(Image.open(os.path.join(self.depth_dir, f"{idx:05d}" + self.col_ext)))
        # segmentation = self.transform_seg(Image.open(os.path.join(self.mask_dir, f"{idx:05d}" + self.col_ext)))
        # imgorig = image.clone()

        # if random.random() < self.p_flip:
        #     image = TF.hflip(image)
        #     depth = TF.hflip(depth)
        #     segmentation = TF.hflip(segmentation)

        # if random.random() < self.p_crop:
        #     random_size = random.randint(256, self.resize-1)
        #     max_size = self.resize - random_size
        #     left = int(random.random()*max_size)
        #     top = int(random.random()*max_size)
        #     image = TF.crop(image, top, left, random_size, random_size)
        #     depth = TF.crop(depth, top, left, random_size, random_size)
        #     segmentation = TF.crop(segmentation, top, left, random_size, random_size)
        #     image = transforms.Resize((self.resize, self.resize))(image)
        #     depth = transforms.Resize((self.resize, self.resize))(depth)
        #     segmentation = transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.NEAREST)(segmentation)

        # if random.random() < self.p_rot:
        #     #rotate
        #     random_angle = random.random()*20 - 10 #[-10 ; 10]
        #     mask = torch.ones((1,self.resize,self.resize)) #useful for the resize at the end
        #     mask = TF.rotate(mask, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
        #     image = TF.rotate(image, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
        #     depth = TF.rotate(depth, random_angle, interpolation=transforms.InterpolationMode.BILINEAR)
        #     segmentation = TF.rotate(segmentation, random_angle, interpolation=transforms.InterpolationMode.NEAREST)
        #     #crop to remove black borders due to the rotation
        #     left = torch.argmax(mask[:,0,:]).item()
        #     top = torch.argmax(mask[:,:,0]).item()
        #     coin = min(left,top)
        #     size = self.resize - 2*coin
        #     image = TF.crop(image, coin, coin, size, size)
        #     depth = TF.crop(depth, coin, coin, size, size)
        #     segmentation = TF.crop(segmentation, coin, coin, size, size)
        #     #Resize
        #     image = transforms.Resize((self.resize, self.resize))(image)
        #     depth = transforms.Resize((self.resize, self.resize))(depth)
        #     segmentation = transforms.Resize((self.resize, self.resize), interpolation=transforms.InterpolationMode.NEAREST)(segmentation)
        # # show([imgorig, image, depth, segmentation])
        # # exit(0)
        # return image, depth, segmentation