# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Miscellaneous utility functions

import gc
import os
import shutil
from typing import Dict

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from termcolor import cprint

from pyvirtualdisplay import Display

class OptionalDisplay:
    def __init__(self, size=(1900, 1084), use_xauth=True, active=False):
        self.display = None
        if active:
            self.display = Display(size=size, use_xauth=use_xauth)

    def __enter__(self):
        if self.display is not None:
            self.display.__enter__()
            print(f"Display created at :{self.display.display}.")

    def __exit__(self, *args, **kwargs):
        if self.display is not None:
            self.display.__exit__()

def print_once(string, bucket=[]):
    """
    Print statement only once: https://stackoverflow.com/a/75484543
    """
    if string not in bucket:
        print(string)
        bucket.append(string)
    if len(bucket) > 50:
        del bucket[:-1]


def gpu_usage_check():
    available, total = torch.cuda.mem_get_info("cuda:0")
    availableGb = available / (1024**3)
    ratioGb = available / total
    if ratioGb < 0.1:
        cprint(f"WARNING: {availableGb}GB available on GPU", color="red")
        gc.collect()
        torch.cuda.empty_cache()


def remove_and_mkdir(results_path: str) -> None:
    """
    Remove directory (if exists) and create
    """
    if os.path.exists(results_path):
        shutil.rmtree(results_path)
    os.makedirs(results_path)


def pose_from_config(cfg: Dict):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(
        [
            cfg["rotation"]["x"],
            cfg["rotation"]["y"],
            cfg["rotation"]["z"],
            cfg["rotation"]["w"],
        ]
    ).as_matrix()
    T[:3, 3] = np.array(
        [cfg["translation"]["x"], cfg["translation"]["y"], cfg["translation"]["z"]]
    )
    return T
