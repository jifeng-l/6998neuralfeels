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

    # Ensure lr is float
    cfg["General"]["lr_backbone"] = float(cfg["General"]["lr_backbone"])
    cfg["General"]["lr_scratch"]  = float(cfg["General"]["lr_scratch"])

    device = cfg["General"].get("device", "cuda:0")
    print("Running on device:", device)

    # Data path & parameters
    # root    = cfg["Dataset"]["root"]
    # col_ext = cfg["Dataset"].get("col_ext", ".jpg")
    # gt_d    = cfg["Dataset"].get("gt_depth", True)
    # resize  = tuple(cfg["Dataset"]["transforms"]["resize"])

    # new versiton
    root    = cfg["Dataset"]["paths"]["root"]
    gt_d = cfg["General"]["gt_depth"]
    col_ext = ".jpg"
    resize = tuple(cfg["Dataset"]["transforms"]["resize"])

    # Test integrated version dataset requires passing resize
    # train_ds = TactileDataset(os.path.join(root, "train"), gt_d, col_ext, tuple(resize))
    # val_ds   = TactileDataset(os.path.join(root,   "val"),   gt_d, col_ext, tuple(resize))

    # new versiton
    train_ds = TactileDataset(os.path.join(root, "train"), gt_d, col_ext, resize, cfg)
    val_ds   = TactileDataset(os.path.join(root,   "val"), gt_d, col_ext, resize, cfg)


    bs = cfg["General"]["batch_size"]
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=4,     # Debugging: Set to 0 first to locate issues. Once confirmed, set back to 4
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
    # Initialize Trainer
    trainer = Trainer(cfg)

    # # Enter training
    # trainer.train(
    #     train_loader,
    #     val_loader
    # )

    # Test new dataset get item without tensor_loader
    trainer.train(train_loader, val_loader)


    # Final evaluation on validation set
    print("\n=== Final evaluation on validation set ===")
    final_loss = trainer.run_eval(
        val_loader
    )
    print(f"Validation loss: {final_loss:.4f}")

    # Save final model snapshot
    trainer.save_model("final")
    print("Saved final model snapshot.")

if __name__ == "__main__":
    main()
