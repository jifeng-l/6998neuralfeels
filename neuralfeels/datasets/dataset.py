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

root = git.Repo(".", search_parent_directories=True).working_tree_dir


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
        resize = None
    ):
        """DIGIT dataset loader for neuralfeels dataset"""
        self.rgb_dir = os.path.join(root_dir, "image")
        self.depth_dir = os.path.join(root_dir, "depth")
        self.mask_dir = os.path.join(root_dir, "mask")
        self.col_ext = col_ext
        self.gt_depth = gt_depth

        if resize != None:
            self.image_size = resize
            print(f"images in dataset will resize to {resize}")

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
        if torch.is_tensor(idx):
            idx = idx.tolist()
        H, W = self.image_size
        name = f"{idx:05d}{self.col_ext}"
        rgb_path   = os.path.join(self.rgb_dir,   name)
        depth_path = os.path.join(self.depth_dir, name)
        mask_path  = os.path.join(self.mask_dir,  name)

        # 2) 读 RGB + resize + to tensor
        img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"RGB not found: {rgb_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        image_tensor = torch.from_numpy(img).permute(2,0,1).float().div(255.0)  # (3,H,W)

        # 3) 读 Depth + mask 后处理
        depth_np = np.zeros((H, W), dtype=np.float32)
        if self.gt_depth:
            raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise FileNotFoundError(f"Depth not found: {depth_path}")
            raw = raw.astype(np.int64)
            raw[raw < 0] = 0

            m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if m is None:
                raise FileNotFoundError(f"Mask not found: {mask_path}")
            m = (m.astype(np.int64) > 127)  # bool mask
            # 如果接触区域太小，就当全无接触
            if m.sum() / m.size < 0.01:
                m[:] = False

            masked = raw * m
            resized = cv2.resize(masked.astype(np.float32), (W, H),
                                 interpolation=cv2.INTER_NEAREST)
            depth_np = resized / 255.0

        depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).float()  # (1,H,W)

        # 4) 读 Mask → segmentation label  
        #    0=background, 1=contact
        m = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(f"Mask not found for segmentation: {mask_path}")
        m = (m.astype(np.int64) > 127).astype(np.uint8)
        m_resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        seg_tensor = torch.from_numpy(m_resized).long().unsqueeze(0)  # (1,H,W)

        return image_tensor, depth_tensor, seg_tensor
