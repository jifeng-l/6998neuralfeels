# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/antocad/FocusOnDepth and https://github.com/isl-org/DPT

# Tactile transformer training code

import os

import cv2
import numpy as np
import torch
import wandb
from numpy.core.numeric import Inf
from tqdm import tqdm
from utils import create_dir, get_losses, get_optimizer, get_schedulers

from neuralfeels.contrib.tactile_transformer.dpt_model import DPTModel


class Trainer(object):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.type = self.config["General"]["type"]

        self.device = torch.device(
            self.config["General"]["device"] if torch.cuda.is_available() else "cpu"
        )
        print("device: %s" % self.device)
        resize = config["Dataset"]["transforms"]["resize"]
        self.model = DPTModel(
            image_size=(3, resize[0], resize[1]),
            emb_dim=config["General"]["emb_dim"],
            resample_dim=config["General"]["resample_dim"],
            read=config["General"]["read"],
            nclasses=len(config["Dataset"]["classes"]),
            hooks=config["General"]["hooks"],
            model_timm=config["General"]["model_timm"],
            type=self.type,
            patch_size=config["General"]["patch_size"],
            model_type = config["General"]["backbone_type"],
        )

        self.model.to(self.device)

        # path_model = os.path.join(config["General"]["path_model"], "DPTModel111.p")
        # path_model = os.path.join(config["General"]["path_model"], "dpt_real.p")
        path_model = config["General"]["path_model"]
        if path_model is not None:
            # 检查模型文件夹中是否存在模型文件
            model_files = [f for f in os.listdir(path_model) if f.endswith('.pth') or f.endswith('.pt') or f.endswith('.p')]
            if model_files:
                # 使用最新的模型文件
                latest_model = sorted(model_files)[-1]
                model_path = os.path.join(path_model, latest_model)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device)["model_state_dict"])
                print(f"✅ Loaded pretrained model from {model_path}")
            else:
                print(f"⚠️ No model files found in {path_model}, training from scratch.")
        else:
            print("ℹ️ No model path specified, training from scratch.")

        # path_model = config["General"]["path_model"]
        # if os.path.exists(path_model):
        #     self.model.load_state_dict(torch.load(path_model, map_location=self.device)["model_state_dict"])
        #     print(f"✅ Loaded pretrained model from {path_model}")
        # else:
        #     print(f"⚠️ No pretrained model found at {path_model}, training from scratch.")

        # print(self.model)
        # exit(0)

        self.loss_depth, self.loss_segmentation = get_losses(config)
        print(f"[DEBUG] using {self.loss_depth.__class__.__name__} as depth loss, {self.loss_segmentation.__class__.__name__} as seg loss")
        print(self.loss_depth)
        self.optimizer_backbone, self.optimizer_scratch = get_optimizer(
            config, self.model
        )
        self.schedulers = get_schedulers(
            [self.optimizer_backbone, self.optimizer_scratch]
        )

    # 全黑version
    # def train(self, train_dataloader, val_dataloader):
    #     epochs = self.config["General"]["epochs"]
    #     if self.config["wandb"]["enable"]:
    #         wandb.init(project=self.config["wandb"]["project"], entity=self.config["wandb"]["entity"])
    #         wandb.config = {
    #             "learning_rate_backbone": self.config["General"]["lr_backbone"],
    #             "learning_rate_scratch": self.config["General"]["lr_scratch"],
    #             "epochs": epochs,
    #             "batch_size": self.config["General"]["batch_size"],
    #         }
    #     val_loss = Inf
    #     for epoch in range(epochs):  # loop over the dataset multiple times
    #         print("Epoch ", epoch + 1)
    #         running_loss = 0.0
    #         self.model.train()
    #         pbar = tqdm(train_dataloader)
    #         pbar.set_description("Training")
    #         for i, (X, Y_depths, Y_segmentations) in enumerate(pbar):
    #             # get the inputs; data is a list of [inputs, labels]
    #             # if i==0:
    #             #     print(f"===the size of sample image in training is {X.shape, Y_depths.shape, Y_segmentations.shapep}===")
    #                 # print(f"===the sample input image is {X}===")
    #                 # print(f"===the sample input depth is {Y_depths}===")
    #                 # assert torch.all(Y_depths == 0).item() is not True, "depth in dataset is all 0!"
    #             X, Y_depths, Y_segmentations = (
    #                 X.to(self.device),
    #                 Y_depths.to(self.device),
    #                 Y_segmentations.to(self.device),
    #             )
    #             if i<10:
    #                 print(f"==== the depth map mean is {Y_depths.mean()} ====")
    #                 print(f"==== the depth map max is {Y_depths.max()} ====")
    #             # zero the parameter gradients
    #             self.optimizer_backbone.zero_grad()
    #             self.optimizer_scratch.zero_grad()
    #             # forward + backward + optimizer
    #             output_depths, output_segmentations = self.model(X)
    #             assert torch.is_tensor(output_depths), "output_depths is not tensor"
    #             # assert  torch.is_tensor(output_segmentations) , "output_segmentations is not tensor"
    #             output_depths = (
    #                 output_depths.squeeze(1) if output_depths != None else None
    #             )

    #             Y_depths = Y_depths.squeeze(1)  # 1xHxW -> HxW
    #             Y_segmentations = Y_segmentations.squeeze(1)  # 1xHxW -> HxW
    #             # print(type(output_depths))
    #             # print(type(Y_depths))
    #             # get loss
    #             l1 = self.loss_depth(
    #                 output_depths, Y_depths
    #             )
    #             # print("[DEBUG] Got loss l1:", l1)
    #             l2 = self.loss_segmentation(output_segmentations, Y_segmentations)
    #             # loss = l1 + l2
    #             loss = l1
    #             assert torch.is_tensor(l1), f"loss 1 is {type(l1)}, {l1}"
    #             # assert torch.is_tensor(l2), f"loss 2 is {type(l2)}, {l2}"
    #             assert torch.is_tensor(loss) and loss.requires_grad, "Loss must be a tensor with gradient"
    #             # if not torch.is_tensor(loss) or loss.grad_fn is None:
    #             #     continue  
    #             loss.backward()
    #             # step optimizer
    #             self.optimizer_scratch.step()
    #             self.optimizer_backbone.step()

    #             running_loss += loss.item()
    #             if np.isnan(running_loss):
    #                 print(
    #                     "\n",
    #                     X.min().item(),
    #                     X.max().item(),
    #                     "\n",
    #                     Y_depths.min().item(),
    #                     Y_depths.max().item(),
    #                     "\n",
    #                     output_depths.min().item(),
    #                     output_depths.max().item(),
    #                     "\n",
    #                     loss.item(),
    #                 )
    #                 exit(0)

    #             if self.config["wandb"]["enable"] and (
    #                 (i % 50 == 0 and i > 0) or i == len(train_dataloader) - 1
    #             ):
    #                 wandb.log({"loss": running_loss / (i + 1)})
    #             pbar.set_postfix({"training_loss": running_loss / (i + 1)})

    #         new_val_loss = self.run_eval(val_dataloader)

    #         if new_val_loss < val_loss:
    #             self.save_model(epoch)
    #             val_loss = new_val_loss

    #         self.schedulers[0].step(new_val_loss)
    #         self.schedulers[1].step(new_val_loss)

    #     print("Finished Training")

    # new version, use unmasked raw depth
    # def train(self, train_dataloader, val_dataloader):
    #     epochs = self.config["General"]["epochs"]
    #     if self.config["wandb"]["enable"]:
    #         wandb.init(project=self.config["wandb"]["project"], entity=self.config["wandb"]["entity"])
    #         wandb.config = {
    #             "learning_rate_backbone": self.config["General"]["lr_backbone"],
    #             "learning_rate_scratch": self.config["General"]["lr_scratch"],
    #             "epochs": epochs,
    #             "batch_size": self.config["General"]["batch_size"],
    #         }
    #     val_loss = Inf
    #     for epoch in range(epochs):  # loop over the dataset multiple times
    #         print("Epoch ", epoch + 1)
    #         running_loss = 0.0
    #         self.model.train()
    #         pbar = tqdm(train_dataloader)
    #         pbar.set_description("Training")
    #         for i, (X, Y_depths, Y_masks) in enumerate(pbar):
    #             # get the inputs; data is a list of [inputs, labels]
    #             # if i==0:
    #             #     print(f"===the size of sample image in training is {X.shape, Y_depths.shape, Y_segmentations.shapep}===")
    #                 # print(f"===the sample input image is {X}===")
    #                 # print(f"===the sample input depth is {Y_depths}===")
    #                 # assert torch.all(Y_depths == 0).item() is not True, "depth in dataset is all 0!"
    #             X, Y_depths, Y_masks = (
    #                 X.to(self.device),
    #                 Y_depths.to(self.device),
    #                 Y_masks.to(self.device),
    #             )
    #             # if i<10:
    #             #     print(f"==== the depth map mean is {Y_depths.mean()} ====")
    #             #     print(f"==== the depth map max is {Y_depths.max()} ====")
    #             # zero the parameter gradients
    #             self.optimizer_backbone.zero_grad()
    #             self.optimizer_scratch.zero_grad()
    #             # forward + backward + optimizer
    #             output_depths, output_segmentations = self.model(X)
    #             assert torch.is_tensor(output_depths), "output_depths is not tensor"
    #             # assert  torch.is_tensor(output_segmentations) , "output_segmentations is not tensor"
    #             output_depths = (
    #                 output_depths.squeeze(1) if output_depths != None else None
    #             )

    #             Y_depths = Y_depths.squeeze(1)  # 1xHxW -> HxW
    #             Y_masks = Y_masks.squeeze(1)  # 1xHxW -> HxW
    #             # print(type(output_depths))
    #             # print(type(Y_depths))
    #             # get loss
    #             valid_mask = (Y_masks > 0).float()
    #             # l1 = self.loss_depth(
    #             #     output_depths, Y_depths
    #             # )
    #             l1 = ((output_depths - Y_depths) ** 2 * valid_mask).sum() / (valid_mask.sum() + 1e-8)
    #             # print("[DEBUG] Got loss l1:", l1)
    #             # l2 = self.loss_segmentation(output_segmentations, Y_segmentations)
    #             # loss = l1 + l2
    #             loss = l1
    #             assert torch.is_tensor(l1), f"loss 1 is {type(l1)}, {l1}"
    #             # assert torch.is_tensor(l2), f"loss 2 is {type(l2)}, {l2}"
    #             assert torch.is_tensor(loss) and loss.requires_grad, "Loss must be a tensor with gradient"
    #             # if not torch.is_tensor(loss) or loss.grad_fn is None:
    #             #     continue  
    #             loss.backward()
    #             # step optimizer
    #             self.optimizer_scratch.step()
    #             self.optimizer_backbone.step()

    #             running_loss += loss.item()
    #             if np.isnan(running_loss):
    #                 print(
    #                     "\n",
    #                     X.min().item(),
    #                     X.max().item(),
    #                     "\n",
    #                     Y_depths.min().item(),
    #                     Y_depths.max().item(),
    #                     "\n",
    #                     output_depths.min().item(),
    #                     output_depths.max().item(),
    #                     "\n",
    #                     loss.item(),
    #                 )
    #                 exit(0)

    #             if self.config["wandb"]["enable"] and (
    #                 (i % 50 == 0 and i > 0) or i == len(train_dataloader) - 1
    #             ):
    #                 wandb.log({"loss": running_loss / (i + 1)})
    #             pbar.set_postfix({"training_loss": running_loss / (i + 1)})

    #         new_val_loss = self.run_eval(val_dataloader)

    #         if new_val_loss < val_loss:
    #             self.save_model(epoch)
    #             val_loss = new_val_loss

    #         self.schedulers[0].step(new_val_loss)
    #         self.schedulers[1].step(new_val_loss)

    #     print("Finished Training")
    # def train(self, train_dataloader, val_dataloader):
    #     epochs = self.config["General"]["epochs"]
    #     if self.config["wandb"]["enable"]:
    #         wandb.init(project=self.config["wandb"]["project"], entity=self.config["wandb"]["entity"])
    #         wandb.config = {
    #             "learning_rate_backbone": self.config["General"]["lr_backbone"],
    #             "learning_rate_scratch": self.config["General"]["lr_scratch"],
    #             "epochs": epochs,
    #             "batch_size": self.config["General"]["batch_size"],
    #         }
    #     val_loss = Inf
    #     for epoch in range(epochs):  # loop over the dataset multiple times
    #         print("Epoch ", epoch + 1)
    #         running_loss = 0.0
    #         self.model.train()
    #         pbar = tqdm(train_dataloader)
    #         pbar.set_description("Training")
    #         for i, (X, Y_depths, Y_segmentations) in enumerate(pbar):
    #             # get the inputs; data is a list of [inputs, labels]
    #             X, Y_depths, Y_segmentations = (
    #                 X.to(self.device),
    #                 Y_depths.to(self.device),
    #                 Y_segmentations.to(self.device),
    #             )
    #             # zero the parameter gradients
    #             self.optimizer_backbone.zero_grad()
    #             self.optimizer_scratch.zero_grad()
    #             # forward + backward + optimizer
    #             output_depths, output_segmentations = self.model(X)
    #             output_depths = (
    #                 output_depths.squeeze(1) if output_depths != None else None
    #             )

    #             Y_depths = Y_depths.squeeze(1)  # 1xHxW -> HxW
    #             Y_segmentations = Y_segmentations.squeeze(1)  # 1xHxW -> HxW
    #             # get loss
    #             loss = self.loss_depth(
    #                 output_depths, Y_depths
    #             ) + self.loss_segmentation(output_segmentations, Y_segmentations)
    #             loss.backward()
    #             # step optimizer
    #             self.optimizer_scratch.step()
    #             self.optimizer_backbone.step()

    #             running_loss += loss.item()
    #             if np.isnan(running_loss):
    #                 print(
    #                     "\n",
    #                     X.min().item(),
    #                     X.max().item(),
    #                     "\n",
    #                     Y_depths.min().item(),
    #                     Y_depths.max().item(),
    #                     "\n",
    #                     output_depths.min().item(),
    #                     output_depths.max().item(),
    #                     "\n",
    #                     loss.item(),
    #                 )
    #                 exit(0)

    #             if self.config["wandb"]["enable"] and (
    #                 (i % 50 == 0 and i > 0) or i == len(train_dataloader) - 1
    #             ):
    #                 wandb.log({"loss": running_loss / (i + 1)})
    #             pbar.set_postfix({"training_loss": running_loss / (i + 1)})

    #         new_val_loss = self.run_eval(val_dataloader)

    #         if new_val_loss < val_loss:
    #             self.save_model(epoch)
    #             val_loss = new_val_loss

    #         self.schedulers[0].step(new_val_loss)
    #         self.schedulers[1].step(new_val_loss)

    #     print("Finished Training")
    def train(self, train_dataloader, val_dataloader):
        """
        Train the model for a given number of epochs, computing additional
        metrics (RMSE, MAE, IoU) on the fly and logging to wandb if enabled.
        """
        epochs = self.config["General"]["epochs"]
        use_wandb = self.config["wandb"]["enable"]

        # Initialize wandb run if requested
        if use_wandb:
            wandb.init(
                project=self.config["wandb"]["project"],
                entity=self.config["wandb"]["entity"],
                config={
                    "lr_backbone": self.config["General"]["lr_backbone"],
                    "lr_scratch": self.config["General"]["lr_scratch"],
                    "epochs": epochs,
                    "batch_size": self.config["General"]["batch_size"],
                }
            )

        best_val_loss = float('inf')

        # Loop over epochs
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            self.model.train()  # set model to training mode

            running_loss = 0.0
            # Initialize accumulators for additional metrics
            rmse_sum = 0.0
            mae_sum  = 0.0
            iou_sum  = 0.0
            batch_count = 0

            # Iterate over training batches with a progress bar
            pbar = tqdm(train_dataloader, desc="Training")
            for X, Y_depths, Y_segs in pbar:
                # Move data to the target device
                X, Y_depths, Y_segs = (
                    X.to(self.device),
                    Y_depths.to(self.device),
                    Y_segs.to(self.device),
                )

                # Zero gradients before backward pass
                self.optimizer_backbone.zero_grad()
                self.optimizer_scratch.zero_grad()

                # Forward pass: predict depths and segmentations
                pred_depths, pred_segs = self.model(X)
                pred_depths = pred_depths.squeeze(1)  # (B, H, W)
                Y_depths   = Y_depths.squeeze(1)       # (B, H, W)
                Y_segs      = Y_segs.squeeze(1)        # (B, H, W)

                # Compute multi-task loss
                loss = self.loss_depth(pred_depths, Y_depths)
                if pred_segs is not None:
                    loss += self.loss_segmentation(pred_segs, Y_segs)

                # Backpropagation
                loss.backward()
                self.optimizer_scratch.step()
                self.optimizer_backbone.step()

                # Accumulate running loss
                running_loss += loss.item()

                # --- Compute additional metrics --- #
                # 1. RMSE (Root Mean Squared Error) for depth
                mse = torch.mean((pred_depths - Y_depths) ** 2).item()
                rmse_sum += np.sqrt(mse)

                # 2. MAE (Mean Absolute Error) for depth
                mae_sum += torch.mean(torch.abs(pred_depths - Y_depths)).item()

                # 3. IoU (Intersection over Union) for segmentation
                if pred_segs is not None:
                    # Convert logits to class predictions
                    pred_labels = torch.argmax(pred_segs, dim=1)
                    # Compute IoU per sample
                    for i in range(pred_labels.size(0)):
                        p = pred_labels[i].detach().cpu().numpy()
                        g = Y_segs[i].detach().cpu().numpy()
                        inter = np.logical_and(p, g).sum()
                        uni   = np.logical_or(p, g).sum()
                        iou_sum += (inter / uni) if uni > 0 else 1.0

                batch_count += 1

                # Log intermediate training loss to wandb every 50 batches
                if use_wandb and (batch_count % 50 == 0 or batch_count == len(train_dataloader)):
                    wandb.log({"train_loss": running_loss / batch_count})

                # Update progress bar postfix with current averages
                avg_iou = (iou_sum / (batch_count * X.size(0))) if pred_segs is not None else 0.0
                pbar.set_postfix({
                    "loss": running_loss / batch_count,
                    "rmse": rmse_sum / batch_count,
                    "mae":  mae_sum / batch_count,
                    "iou":  avg_iou
                })

            # --- End of epoch: compute final averages --- #
            avg_loss = running_loss / batch_count
            avg_rmse = rmse_sum / batch_count
            avg_mae  = mae_sum / batch_count
            avg_iou  = iou_sum / (batch_count * train_dataloader.batch_size) \
                    if pred_segs is not None else 0.0

            print(
                f"Epoch {epoch+1} summary: "
                f"loss={avg_loss:.4f}, "
                f"RMSE={avg_rmse:.4f}, "
                f"MAE={avg_mae:.4f}, "
                f"IoU={avg_iou:.4f}"
            )

            # Log epoch metrics to wandb
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss_epoch": avg_loss,
                    "train_rmse": avg_rmse,
                    "train_mae": avg_mae,
                    "train_iou": avg_iou
                })

            # Perform validation and save best checkpoint
            val_loss = self.run_eval(val_dataloader)
            if val_loss < best_val_loss:
                self.save_model(epoch)
                best_val_loss = val_loss

            # Step learning rate schedulers with validation loss
            for sched in self.schedulers:
                sched.step(val_loss)

        print("Finished Training")
    # ljf old version
    # def train(self, train_loader, val_loader):
    #     epochs = self.config["General"]["epochs"]
    #     use_wandb = self.config["wandb"]["enable"]
    #     if use_wandb:
    #         wandb.init(
    #             project="DPTModel", 
    #             entity=self.config["wandb"]["username"]
    #         )
    #         wandb.config.update({
    #             "lr_backbone": self.config["General"]["lr_backbone"],
    #             "lr_scratch": self.config["General"]["lr_scratch"],
    #             "epochs": epochs,
    #             "batch_size": self.config["General"]["batch_size"],
    #         })

    #     best_val_loss = Inf
    #     for epoch in range(1, epochs + 1):
    #         print(f"Epoch {epoch}/{epochs}")
    #         self.model.train()
    #         running_loss = 0.0

    #         pbar = tqdm(train_loader, desc="  Train")
    #         for i, (X, Y_depths, Y_segs) in enumerate(pbar):
    #             X = X.to(self.device)
    #             Y_depths = Y_depths.to(self.device)
    #             Y_segs = Y_segs.to(self.device)

    #             self.optimizer_backbone.zero_grad()
    #             self.optimizer_scratch.zero_grad()

    #             # 前向
    #             output_depths, output_segs = self.model(X)

    #             # 计算 loss，初始化为 None
    #             loss = None

    #             # 深度分支
    #             if output_depths is not None:
    #                 od = output_depths.squeeze(1)  # (B, H, W)
    #                 ld = self.loss_depth(od, Y_depths)
    #                 loss = ld

    #             # 分割分支
    #             if output_segs is not None:
    #                 ls = self.loss_segmentation(output_segs, Y_segs.squeeze(1))
    #                 loss = ls if loss is None else loss + ls

    #             # 如果两路都没输出，跳过
    #             if loss is None:
    #                 continue

    #             # 确保 loss 为 Tensor
    #             if not torch.is_tensor(loss):
    #                 loss = torch.tensor(loss, dtype=torch.float32, device=self.device)
    #             if not torch.is_tensor(loss):
    #                 loss = torch.tensor(loss, dtype=torch.float32, device=self.device)

    #             # 如果 loss 没有 grad_fn（即不需要梯度），跳过这一批
    #             if loss.grad_fn is None:
    #                 pbar.set_postfix(warning="skipped no-grad loss")
    #                 continue

    #             # 反向 + 优化
    #             loss.backward()
    #             self.optimizer_scratch.step()
    #             self.optimizer_backbone.step()

    #             running_loss += loss.item()
    #             pbar.set_postfix(train_loss=running_loss / (i + 1))

    #             if use_wandb and (i % 50 == 0 or i == len(train_loader) - 1):
    #                 wandb.log({"train_loss": running_loss / (i + 1)})

    #         # 验证
    #         val_loss = self.run_eval(val_loader)
    #         print(f"  Val loss: {val_loss:.4f}")

    #         # 保存最佳模型
    #         if val_loss < best_val_loss:
    #             self.save_model(epoch)
    #             best_val_loss = val_loss

    #         # 调度器更新
    #         self.schedulers[0].step(val_loss)
    #         self.schedulers[1].step(val_loss)

    #     print("Finished Training")

    # def run_eval(self, val_dataloader):
    #     """
    #     Evaluate the model on the validation set and visualize some results
    #     on wandb
    #     :- val_dataloader -: torch dataloader
    #     """
    #     # return 0
    #     val_loss = 0.0
    #     self.model.eval()
    #     X_1 = None
    #     Y_depths_1 = None
    #     Y_segmentations_1 = None
    #     output_depths_1 = None
    #     output_segmentations_1 = None
    #     with torch.no_grad():
    #         pbar = tqdm(val_dataloader)
    #         pbar.set_description("Validation")
    #         for i, (X, Y_depths, Y_segmentations) in enumerate(pbar):
    #             X, Y_depths, Y_segmentations = (
    #                 X.to(self.device),
    #                 Y_depths.to(self.device),
    #                 Y_segmentations.to(self.device),
    #             )
    #             # if i==0:
    #                 # print(f"===the sample image shape is {X.shape}===")
    #                 # print(f"===the sample depth is {Y_depths}===")
    #             output_depths, output_segmentations = self.model(X)
    #             assert output_depths is not None, "output_depths is None"
    #             # assert output_segmentations is not None, "output_segmentations is None"
    #             output_depths = (
    #                 output_depths.squeeze(1) if output_depths != None else None
    #             )
    #             Y_depths = Y_depths.squeeze(1)
    #             Y_segmentations = Y_segmentations.squeeze(1)
    #             if i == 0:
    #                 X_1 = X
    #                 Y_depths_1 = Y_depths
    #                 Y_segmentations_1 = Y_segmentations
    #                 output_depths_1 = output_depths
    #                 output_segmentations_1 = output_segmentations
    #             # get loss
    #             loss = self.loss_depth(
    #                 output_depths, Y_depths
    #             ) + self.loss_segmentation(output_segmentations, Y_segmentations)
    #             assert isinstance(loss, (torch.Tensor, float, int)), f"Unexpected loss type: {type(loss)}"
    #             val_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
    #             pbar.set_postfix({"validation_loss": val_loss / (i + 1)})
    #         if self.config["wandb"]["enable"]:
    #             if wandb.run is None:
    #                 wandb.init(project=self.config["wandb"]["project"], entity=self.config["wandb"]["entity"])
    #             wandb.log({"val_loss": val_loss / (i + 1)})
    #             self.img_logger(
    #                 X_1,
    #                 Y_depths_1,
    #                 Y_segmentations_1,
    #                 output_depths_1,
    #                 output_segmentations_1,
                    
    #             )

    #     if 'i' not in locals():
    #         print("val loader not valid, set loss to 0.0!")
    #         return 0.0  # 没有正确的遍历valloader 

    #     else:
    #         return val_loss / (i + 1)

    def run_eval(self, val_dataloader):
        """
        Evaluate the model on the validation set, compute loss and additional metrics
        (RMSE, MAE, IoU), log them to wandb, and visualize one batch of results.
        """
        # Switch model to evaluation mode
        self.model.eval()

        total_loss = 0.0
        rmse_sum = 0.0
        mae_sum = 0.0
        iou_sum = 0.0
        sample_count = 0

        # Placeholders for one batch to visualize
        X_vis = None
        Y_depths_vis = None
        Y_segs_vis = None
        pred_depths_vis = None
        pred_segs_vis = None

        with torch.no_grad():
            pbar = tqdm(val_dataloader, desc="Validation")
            for batch_idx, (X, Y_depths, Y_segs) in enumerate(pbar):
                # Move data to device
                X = X.to(self.device)
                Y_depths = Y_depths.to(self.device).squeeze(1)   # shape (B, H, W)
                Y_segs   = Y_segs.to(self.device).squeeze(1)     # shape (B, H, W)

                # Forward pass
                pred_depths, pred_segs = self.model(X)
                pred_depths = pred_depths.squeeze(1)  # shape (B, H, W)

                # Store first batch for visualization
                if batch_idx == 0:
                    X_vis = X
                    Y_depths_vis = Y_depths
                    Y_segs_vis = Y_segs
                    pred_depths_vis = pred_depths
                    pred_segs_vis = pred_segs

                # Compute losses
                loss = self.loss_depth(pred_depths, Y_depths)
                if pred_segs is not None:
                    loss += self.loss_segmentation(pred_segs, Y_segs)
                total_loss += loss.item()

                # Compute RMSE for this batch
                mse = torch.mean((pred_depths - Y_depths) ** 2).item()
                rmse_sum += np.sqrt(mse)

                # Compute MAE for this batch
                mae_sum += torch.mean(torch.abs(pred_depths - Y_depths)).item()

                # Compute IoU for segmentation if available
                batch_size = X.size(0)
                sample_count += batch_size
                if pred_segs is not None:
                    # Convert logits to predicted classes
                    pred_labels = torch.argmax(pred_segs, dim=1)  # shape (B, H, W)
                    pred_np = pred_labels.detach().cpu().numpy()
                    gt_np   = Y_segs.detach().cpu().numpy()
                    for i in range(batch_size):
                        p = pred_np[i]
                        g = gt_np[i]
                        intersection = np.logical_and(p, g).sum()
                        union        = np.logical_or(p, g).sum()
                        # Avoid division by zero
                        iou_sum += (intersection / union) if union > 0 else 1.0

                # Update progress bar with running averages
                avg_loss = total_loss / (batch_idx + 1)
                avg_rmse = rmse_sum / (batch_idx + 1)
                avg_mae  = mae_sum / (batch_idx + 1)
                avg_iou  = (iou_sum / sample_count) if pred_segs is not None else 0.0
                pbar.set_postfix({
                    "val_loss": avg_loss,
                    "val_rmse": avg_rmse,
                    "val_mae":  avg_mae,
                    "val_iou":  avg_iou
                })

            # Log final metrics to wandb if enabled
            if self.config["wandb"]["enable"]:
                if wandb.run is None:
                    wandb.init(
                        project=self.config["wandb"]["project"],
                        entity=self.config["wandb"]["entity"]
                    )
                wandb.log({
                    "val_loss": total_loss / len(val_dataloader),
                    "val_rmse": rmse_sum / len(val_dataloader),
                    "val_mae":  mae_sum / len(val_dataloader),
                    "val_iou":  avg_iou
                })
                # Visualize one batch of images, depths, and segmentations
                self.img_logger(
                    X_vis,
                    Y_depths_vis,
                    Y_segs_vis,
                    pred_depths_vis,
                    pred_segs_vis
                )

        # Return average validation loss
        return total_loss / len(val_dataloader)
        # return val_loss / (i + 1)
    
    # def run_eval(self, val_dataloader):
    #     """
    #     Evaluate the model on the validation set and visualize some results
    #     on wandb
    #     """
    #     val_loss = 0.0
    #     self.model.eval()
    #     X_1 = None
    #     Y_depths_1 = None
    #     Y_segmentations_1 = None
    #     output_depths_1 = None
    #     output_segmentations_1 = None

    #     num_batches = 0

    #     with torch.no_grad():
    #         pbar = tqdm(val_dataloader)
    #         pbar.set_description("Validation")

    #         for i, (X, Y_depths, Y_segmentations) in enumerate(pbar):
    #             X, Y_depths, Y_segmentations = (
    #                 X.to(self.device),
    #                 Y_depths.to(self.device),
    #                 Y_segmentations.to(self.device),
    #             )

    #             output_depths, output_segmentations = self.model(X)
    #             output_depths = (
    #                 output_depths.squeeze(1) if output_depths is not None else None
    #             )
    #             Y_depths = Y_depths.squeeze(1)
    #             Y_segmentations = Y_segmentations.squeeze(1)

    #             if i == 0:
    #                 X_1 = X
    #                 Y_depths_1 = Y_depths
    #                 Y_segmentations_1 = Y_segmentations
    #                 output_depths_1 = output_depths
    #                 output_segmentations_1 = output_segmentations

    #             # compute loss
    #             loss = self.loss_depth(output_depths, Y_depths) + \
    #                 self.loss_segmentation(output_segmentations, Y_segmentations)

    #             assert isinstance(loss, (torch.Tensor, float, int)), f"Unexpected loss type: {type(loss)}"
    #             val_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
    #             num_batches += 1
    #             pbar.set_postfix({"validation_loss": val_loss / num_batches})

    #     # If at least one batch was processed, log to wandb
    #     if num_batches > 0:
    #         avg_val_loss = val_loss / num_batches
    #         if self.config.get("wandb", {}).get("enable", False):
    #             wandb.log({"val_loss": avg_val_loss})
    #             self.img_logger(
    #                 X_1,
    #                 Y_depths_1,
    #                 Y_segmentations_1,
    #                 output_depths_1,
    #                 output_segmentations_1,
    #             )
    #     else:
    #         print("[⚠️] Validation set is empty. Skipping WandB logging.")
    #         avg_val_loss = 0.0

    #     return avg_val_loss

    def save_model(self, epoch):
        path_model = os.path.join(
            self.config["General"]["path_model"], self.model.__class__.__name__
        )
        create_dir(path_model)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_backbone_state_dict": self.optimizer_backbone.state_dict(),
                "optimizer_scratch_state_dict": self.optimizer_scratch.state_dict(),
            },
            path_model + ".p",
        )
        print("Model saved at : {}".format(path_model))

    # def img_logger(self, X, Y_depths, Y_segmentations, output_depths, output_segmentations):
    #     nb_to_show = min(self.config["wandb"]["images_to_show"], len(X))
    #     output_dim = (
    #         int(self.config["wandb"]["im_w"]),
    #         int(self.config["wandb"]["im_h"]),
    #     )

    #     # Normalize and prepare input images
    #     imgs = X[:nb_to_show].detach().cpu().numpy()
    #     imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())
    #     imgs = imgs.transpose(0, 2, 3, 1)

    #     # ---- Log input images ----
    #     wandb.log({
    #         "img": [
    #             wandb.Image(cv2.resize(im, output_dim), caption=f"img_{i+1}")
    #             for i, im in enumerate(imgs)
    #         ]
    #     })

    #     if output_depths is not None:
    #         gt_depth = Y_depths[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
    #         pred_depth = output_depths[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
    #         gt_depth = np.repeat(gt_depth, 3, axis=1).transpose(0, 2, 3, 1)
    #         pred_depth = np.repeat(pred_depth, 3, axis=1).transpose(0, 2, 3, 1)

    #         wandb.log({
    #             "depth_truths": [
    #                 wandb.Image(cv2.resize(im, output_dim), caption=f"depth_truth_{i+1}")
    #                 for i, im in enumerate(gt_depth)
    #             ]
    #         })
    #         wandb.log({
    #             "depth_preds": [
    #                 wandb.Image(cv2.resize(im, output_dim), caption=f"depth_pred_{i+1}")
    #                 for i, im in enumerate(pred_depth)
    #             ]
    #         })

    #     if output_segmentations is not None:
    #         gt_seg = Y_segmentations[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
    #         pred_seg = torch.argmax(output_segmentations[:nb_to_show], dim=1)
    #         pred_seg = pred_seg.unsqueeze(1).detach().cpu().numpy()

    #         gt_seg_rgb = np.repeat(gt_seg, 3, axis=1).astype(np.float32).transpose(0, 2, 3, 1)
    #         pred_seg_rgb = np.repeat(pred_seg, 3, axis=1).astype(np.float32).transpose(0, 2, 3, 1)

    #         wandb.log({
    #             "seg_truths": [
    #                 wandb.Image(cv2.resize(im, output_dim), caption=f"seg_truth_{i+1}")
    #                 for i, im in enumerate(gt_seg_rgb)
    #             ]
    #         })
    #         wandb.log({
    #             "seg_preds": [
    #                 wandb.Image(cv2.resize(im, output_dim), caption=f"seg_pred_{i+1}")
    #                 for i, im in enumerate(pred_seg_rgb)
    #             ]
    #         })

    #         # Masked depth
    #         masked_gt = Y_depths[:nb_to_show] * Y_segmentations[:nb_to_show]
    #         masked_pred = output_depths[:nb_to_show] * pred_seg.squeeze(1)

    #         masked_gt = masked_gt.unsqueeze(1).detach().cpu().numpy()
    #         masked_pred = masked_pred.unsqueeze(1).detach().cpu().numpy()
    #         masked_gt = np.repeat(masked_gt, 3, axis=1).transpose(0, 2, 3, 1)
    #         masked_pred = np.repeat(masked_pred, 3, axis=1).transpose(0, 2, 3, 1)

    #         wandb.log({
    #             "masked_depth_truth": [
    #                 wandb.Image(cv2.resize(im, output_dim), caption=f"masked_gt_{i+1}")
    #                 for i, im in enumerate(masked_gt)
    #             ]
    #         })
    #         wandb.log({
    #             "masked_depth_pred": [
    #                 wandb.Image(cv2.resize(im, output_dim), caption=f"masked_pred_{i+1}")
    #                 for i, im in enumerate(masked_pred)
    #             ]
    #         })
    def img_logger(
        self, X, Y_depths, Y_segmentations, output_depths, output_segmentations
    ):
        nb_to_show = (
            self.config["wandb"]["images_to_show"]
            if self.config["wandb"]["images_to_show"] <= len(X)
            else len(X)
        )
        tmp = X[:nb_to_show].detach().cpu().numpy()
        imgs = (tmp - tmp.min()) / (tmp.max() - tmp.min())
        if output_depths != None:
            tmp = Y_depths[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
            depth_truths = np.repeat(tmp, 3, axis=1)
            tmp = output_depths[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
            tmp = np.repeat(tmp, 3, axis=1)
            depth_preds = tmp
        if output_segmentations != None:
            tmp = Y_segmentations[:nb_to_show].unsqueeze(1).detach().cpu().numpy()
            segmentation_truths = np.repeat(tmp, 3, axis=1).astype("float32")
            tmp = torch.argmax(output_segmentations[:nb_to_show], dim=1)
            tmp = tmp.unsqueeze(1).detach().cpu().numpy()
            tmp = np.repeat(tmp, 3, axis=1)
            segmentation_preds = tmp.astype("float32")
        # print("******************************************************")
        # print(imgs.shape, imgs.mean().item(), imgs.max().item(), imgs.min().item())
        # if output_depths != None:
        #     print(depth_truths.shape, depth_truths.mean().item(), depth_truths.max().item(), depth_truths.min().item())
        #     print(depth_preds.shape, depth_preds.mean().item(), depth_preds.max().item(), depth_preds.min().item())
        # if output_segmentations != None:
        #     print(segmentation_truths.shape, segmentation_truths.mean().item(), segmentation_truths.max().item(), segmentation_truths.min().item())
        #     print(segmentation_preds.shape, segmentation_preds.mean().item(), segmentation_preds.max().item(), segmentation_preds.min().item())
        # print("******************************************************")
        imgs = imgs.transpose(0, 2, 3, 1)
        if output_depths != None:
            depth_truths = depth_truths.transpose(0, 2, 3, 1)
            depth_preds = depth_preds.transpose(0, 2, 3, 1)
        if output_segmentations != None:
            segmentation_truths = segmentation_truths.transpose(0, 2, 3, 1)
            segmentation_preds = segmentation_preds.transpose(0, 2, 3, 1)
        output_dim = (
            int(self.config["wandb"]["im_w"]),
            int(self.config["wandb"]["im_h"]),
        )

        wandb.log(
            {
                "img": [
                    wandb.Image(
                        cv2.resize(im, output_dim), caption="img_{}".format(i + 1)
                    )
                    for i, im in enumerate(imgs)
                ]
            }
        )
        if output_depths != None:
            wandb.log(
                {
                    "depth_truths": [
                        wandb.Image(
                            cv2.resize(im, output_dim),
                            caption="depth_truths_{}".format(i + 1),
                        )
                        for i, im in enumerate(depth_truths)
                    ],
                    "depth_preds": [
                        wandb.Image(
                            cv2.resize(im, output_dim),
                            caption="depth_preds_{}".format(i + 1),
                        )
                        for i, im in enumerate(depth_preds)
                    ],
                }
            )
        if output_segmentations != None:
            wandb.log(
                {
                    "seg_truths": [
                        wandb.Image(
                            cv2.resize(im, output_dim),
                            caption="seg_truths_{}".format(i + 1),
                        )
                        for i, im in enumerate(segmentation_truths)
                    ],
                    "seg_preds": [
                        wandb.Image(
                            cv2.resize(im, output_dim),
                            caption="seg_preds_{}".format(i + 1),
                        )
                        for i, im in enumerate(segmentation_preds)
                    ],
                }
            )
