# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/antocad/FocusOnDepth and https://github.com/isl-org/DPT

# Utility functions for tactile transformer

import errno
import os
from glob import glob

import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib import cm
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms

from neuralfeels.contrib.tactile_transformer.custom_augmentation import ToMask
from neuralfeels.contrib.tactile_transformer.loss import ScaleAndShiftInvariantLoss


def get_total_paths(path, ext):
    return glob(os.path.join(path, "*" + ext))


def get_splitted_dataset(
    config, split, dataset_name, path_images, path_depths, path_segmentation
):
    list_files = [os.path.basename(im) for im in path_images]
    np.random.seed(config["General"]["seed"])
    np.random.shuffle(list_files)
    if split == "train":
        selected_files = list_files[
            : int(len(list_files) * config["Dataset"]["splits"]["split_train"])
        ]
    elif split == "val":
        selected_files = list_files[
            int(len(list_files) * config["Dataset"]["splits"]["split_train"]) : int(
                len(list_files) * config["Dataset"]["splits"]["split_train"]
            )
            + int(len(list_files) * config["Dataset"]["splits"]["split_val"])
        ]
    else:
        selected_files = list_files[
            int(len(list_files) * config["Dataset"]["splits"]["split_train"])
            + int(len(list_files) * config["Dataset"]["splits"]["split_val"]) :
        ]

    path_images = [
        os.path.join(
            config["Dataset"]["paths"]["path_dataset"],
            dataset_name,
            config["Dataset"]["paths"]["path_images"],
            im[:-4] + config["Dataset"]["extensions"]["ext_images"],
        )
        for im in selected_files
    ]
    path_depths = [
        os.path.join(
            config["Dataset"]["paths"]["path_dataset"],
            dataset_name,
            config["Dataset"]["paths"]["path_depths"],
            im[:-4] + config["Dataset"]["extensions"]["ext_depths"],
        )
        for im in selected_files
    ]
    path_segmentation = [
        os.path.join(
            config["Dataset"]["paths"]["path_dataset"],
            dataset_name,
            config["Dataset"]["paths"]["path_segmentations"],
            im[:-4] + config["Dataset"]["extensions"]["ext_segmentations"],
        )
        for im in selected_files
    ]
    return path_images, path_depths, path_segmentation


def get_transforms(config):
    im_size = config["Dataset"]["transforms"]["resize"]
    transform_image = transforms.Compose(
        [
            transforms.Resize((im_size[0], im_size[1])),
            transforms.ToTensor(),  # converts to [0 - 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    transform_depth = transforms.Compose(
        [
            transforms.Resize((im_size[0], im_size[1])),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),  # converts to [0 - 1]
        ]
    )
    transform_seg = transforms.Compose(
        [
            transforms.Resize(
                (im_size[0], im_size[1]),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            ToMask(config["Dataset"]["classes"]),
        ]
    )
    return transform_image, transform_depth, transform_seg


def get_losses(config):
    def NoneFunction(a, b):
        return 0

    loss_depth = NoneFunction
    loss_segmentation = NoneFunction
    type = config["General"]["type"]
    if type == "full" or type == "depth":
        if config["General"]["loss_depth"] == "mse":
            # loss_depth = nn.L1Loss()
            loss_depth = nn.MSELoss()
        elif config["General"]["loss_depth"] == "ssi":
            loss_depth = ScaleAndShiftInvariantLoss()
    if type == "full" or type == "segmentation":
        if config["General"]["loss_segmentation"] == "ce":
            loss_segmentation = nn.CrossEntropyLoss()
    return loss_depth, loss_segmentation


def create_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_optimizer(config, net):
    names = set([name.split(".")[0] for name, _ in net.named_modules()]) - set(
        ["", "transformer_encoders"]
    )
    params_backbone = net.transformer_encoders.parameters()
    params_scratch = list()
    for name in names:
        params_scratch += list(eval("net." + name).parameters())

    if config["General"]["optim"] == "adam":
        optimizer_backbone = optim.Adam(
            params_backbone, lr=config["General"]["lr_backbone"]
        )
        optimizer_scratch = optim.Adam(
            params_scratch, lr=config["General"]["lr_scratch"]
        )
    elif config["General"]["optim"] == "sgd":
        optimizer_backbone = optim.SGD(
            params_backbone,
            lr=config["General"]["lr_backbone"],
            momentum=config["General"]["momentum"],
        )
        optimizer_scratch = optim.SGD(
            params_scratch,
            lr=config["General"]["lr_scratch"],
            momentum=config["General"]["momentum"],
        )
    return optimizer_backbone, optimizer_scratch


def get_schedulers(optimizers):
    return [
        ReduceLROnPlateau(optimizer, verbose=True, factor=0.8)
        for optimizer in optimizers
    ]


def concat_images(images, direction="horizontal"):
    widths, heights = zip(*(img.size for img in images))

    if direction == "horizontal":
        total_width = sum(widths)
        max_height = max(heights)
        new_image = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            new_image.paste(img, (x_offset, 0))
            x_offset += img.width
    elif direction == "vertical":
        total_height = sum(heights)
        max_width = max(widths)
        new_image = Image.new("RGB", (max_width, total_height))
        y_offset = 0
        for img in images:
            new_image.paste(img, (0, y_offset))
            y_offset += img.height
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'")

    return new_image


def apply_jet_colormap(image):
    # Convert to grayscale if not already
    grayscale_image = image.convert("L")

    # Convert grayscale image to numpy array
    image_np = np.array(grayscale_image)

    # Normalize image data to range [0, 1] for colormap
    image_normalized = image_np / 255.0

    # Apply the jet colormap
    colormap = cm.get_cmap("jet")
    colored_image = colormap(image_normalized)

    # Convert back to 8-bit per channel RGB
    colored_image = (colored_image[:, :, :3] * 255).astype(np.uint8)

    # Convert numpy array back to PIL image
    return Image.fromarray(colored_image)
