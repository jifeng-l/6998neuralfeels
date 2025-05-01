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

def tensor_loader(loader, image_size=(384,384), device="cuda:0"):
    """
    归一化values 并添加dummy segmentation label 以适配DPT model的需要 （原dataset无segmentation label）
    loader: DataLoader yielding (list_of_images, list_of_depths)
    输出: generator yielding (Xb, Yd, Ys)
      Xb: (B,3,H,W) RGB in [0,1]
      Yd: (B,1,H,W) depth in [0,1]
      Ys: dummy seg (全 0)
    """
    H, W = image_size
    for batch in loader:
        imgs_np, deps = batch
        rgb_tensors, dep_tensors = [], []

        for img_np, dep in zip(imgs_np, deps):
            # --------- RGB ----------
            arr = img_np.cpu().numpy() if isinstance(img_np, torch.Tensor) else img_np
            arr = cv2.resize(arr, (W, H))
            t_img = torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)
            rgb_tensors.append(t_img)

            # --------- DEPTH --------
            dn = None
            try:
                # print(f"[DEBUG] Depth input type: {type(dep)}, shape: {getattr(dep, 'shape', 'N/A')}")

                if isinstance(dep, torch.Tensor):
                    dn = dep.cpu().numpy()
                elif isinstance(dep, np.ndarray):
                    dn = dep
                else:
                    raise ValueError("Unsupported depth input type")

                dn = np.squeeze(dn)  # remove singleton dimensions
                if dn.ndim != 2:
                    raise ValueError(f"Expected 2D depth, got shape {dn.shape}")

                # convert dtype if necessary
                if dn.dtype == np.bool_:
                    dn = dn.astype(np.uint8) * 255
                elif dn.dtype not in [np.uint8, np.float32]:
                    dn = dn.astype(np.float32)

                dn = cv2.resize(dn, (W, H))

            except Exception as e:
                print(f"[⚠️ Depth fallback] {e}, replacing with zeros")
                dn = np.zeros((H, W), dtype=np.float32)

            t_dep = torch.from_numpy(dn).unsqueeze(0).float().div(255.0)
            dep_tensors.append(t_dep)

        if len(rgb_tensors) == 0:
            continue

        Xb = torch.stack(rgb_tensors, dim=0).to(device)
        Yd = torch.stack(dep_tensors, dim=0).to(device)
        Ys = torch.zeros_like(Yd, dtype=torch.long, device=device)
        yield Xb, Yd, Ys
        
        
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
    trainer.train(
        train_loader, 
        val_loader)


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
