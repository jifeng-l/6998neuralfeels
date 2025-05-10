#!/usr/bin/env python3
import os
import yaml
import argparse
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader

from neuralfeels.datasets.dataset import TactileDataset
from neuralfeels.contrib.tactile_transformer.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Tactile Transformer")
    parser.add_argument(
        "--config", "-c", required=True, help="Path to tactile_config.yaml"
    )
    return parser.parse_args()

        
def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    # 确保 lr 是 float
    cfg["General"]["lr_backbone"] = float(cfg["General"]["lr_backbone"])
    cfg["General"]["lr_scratch"]  = float(cfg["General"]["lr_scratch"])

    device = cfg["General"].get("device", "cuda:0")
    print("Running on device:", device)

    # 数据路径 & 参数
    root    = cfg["Dataset"]["root"]
    col_ext = cfg["Dataset"].get("col_ext", ".jpg")
    gt_d    = cfg["Dataset"].get("gt_depth", True)
    resize  = tuple(cfg["Dataset"]["transforms"]["resize"])

    # PyTorch Dataset + DataLoader
    # train_ds = TactileDataset(os.path.join(root, "train"), gt_d, col_ext)
    # val_ds   = TactileDataset(os.path.join(root,   "val"),   gt_d, col_ext)

    # 测试集成版dataset 需要传入resize
    train_ds = TactileDataset(os.path.join(root, "train"), gt_d, col_ext, tuple(resize))
    val_ds   = TactileDataset(os.path.join(root,   "val"),   gt_d, col_ext, tuple(resize))


    bs = cfg["General"]["batch_size"]
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=4,     # 调试期先零线程，定位问题。确认无误后可改回 4
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"intializing dataloaders, train loader {len(train_loader)} batches, val loader {len(val_loader)} batches")
    # Trainer 实例化
    trainer = Trainer(cfg)

    # 进入训练
    # trainer.train(
    #     tensor_loader(train_loader, image_size=resize, device=device),
    #     tensor_loader(val_loader,   image_size=resize, device=device),
    # )

    # 测试新dataset get item 无需tensor_loader
    # trainer.train(
    #     train_loader, 
    #     val_loader)


    # 最后在验证集做一次完整 eval 并输出
    print("\n=== Final evaluation on validation set ===")
    final_loss = trainer.run_eval(
        val_loader
    )
    print(f"Validation loss: {final_loss:.4f}")

    # 保存最终模型快照
    trainer.save_model("final")
    print("Saved final model snapshot.")

if __name__ == "__main__":
    main()
