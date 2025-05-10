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
            model_type = config["General"]["model_type"],
        )

        self.model.to(self.device)

        path_model = os.path.join(config["General"]["path_model"], "DPTModel111.p")
        # path_model = os.path.join(config["General"]["path_model"], "dpt_sim.p")
        if os.path.exists(path_model):
            self.model.load_state_dict(torch.load(path_model, map_location=self.device)["model_state_dict"])
            print(f"✅ Loaded pretrained model from {path_model}")
        else:
            print(f"⚠️ No pretrained model found at {path_model}, training from scratch.")

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

    def train(self, train_dataloader, val_dataloader):
        epochs = self.config["General"]["epochs"]
        if self.config["wandb"]["enable"]:
            wandb.init(project="DPTModel", entity=self.config["wandb"]["username"])
            wandb.config = {
                "learning_rate_backbone": self.config["General"]["lr_backbone"],
                "learning_rate_scratch": self.config["General"]["lr_scratch"],
                "epochs": epochs,
                "batch_size": self.config["General"]["batch_size"],
            }
        val_loss = Inf
        for epoch in range(epochs):  # loop over the dataset multiple times
            print("Epoch ", epoch + 1)
            running_loss = 0.0
            self.model.train()
            pbar = tqdm(train_dataloader)
            pbar.set_description("Training")
            for i, (X, Y_depths, Y_segmentations) in enumerate(pbar):
                # get the inputs; data is a list of [inputs, labels]
                # if i==0:
                #     print(f"===the size of sample image in training is {X.shape, Y_depths.shape, Y_segmentations.shapep}===")
                    # print(f"===the sample input image is {X}===")
                    # print(f"===the sample input depth is {Y_depths}===")
                    # assert torch.all(Y_depths == 0).item() is not True, "depth in dataset is all 0!"
                X, Y_depths, Y_segmentations = (
                    X.to(self.device),
                    Y_depths.to(self.device),
                    Y_segmentations.to(self.device),
                )
                # zero the parameter gradients
                self.optimizer_backbone.zero_grad()
                self.optimizer_scratch.zero_grad()
                # forward + backward + optimizer
                output_depths, output_segmentations = self.model(X)
                assert torch.is_tensor(output_depths), "output_depths is not tensor"
                # assert  torch.is_tensor(output_segmentations) , "output_segmentations is not tensor"
                output_depths = (
                    output_depths.squeeze(1) if output_depths != None else None
                )

                Y_depths = Y_depths.squeeze(1)  # 1xHxW -> HxW
                Y_segmentations = Y_segmentations.squeeze(1)  # 1xHxW -> HxW
                # print(type(output_depths))
                # print(type(Y_depths))
                # get loss
                l1 = self.loss_depth(
                    output_depths, Y_depths
                )
                # print("[DEBUG] Got loss l1:", l1)
                l2 = self.loss_segmentation(output_segmentations, Y_segmentations)
                loss = l1 + l2
                assert torch.is_tensor(l1), f"loss 1 is {type(l1)}, {l1}"
                # assert torch.is_tensor(l2), f"loss 2 is {type(l2)}, {l2}"
                # if not torch.is_tensor(loss) or loss.grad_fn is None:
                #     continue  
                assert torch.is_tensor(loss) and loss.requires_grad, "Loss must be a tensor with gradient"
                loss.backward()
                # step optimizer
                self.optimizer_scratch.step()
                self.optimizer_backbone.step()

                running_loss += loss.item()
                if np.isnan(running_loss):
                    print(
                        "\n",
                        X.min().item(),
                        X.max().item(),
                        "\n",
                        Y_depths.min().item(),
                        Y_depths.max().item(),
                        "\n",
                        output_depths.min().item(),
                        output_depths.max().item(),
                        "\n",
                        loss.item(),
                    )
                    exit(0)

                if self.config["wandb"]["enable"] and (
                    (i % 50 == 0 and i > 0) or i == len(train_dataloader) - 1
                ):
                    wandb.log({"loss": running_loss / (i + 1)})
                pbar.set_postfix({"training_loss": running_loss / (i + 1)})

            new_val_loss = self.run_eval(val_dataloader)

            if new_val_loss < val_loss:
                self.save_model(epoch)
                val_loss = new_val_loss

            self.schedulers[0].step(new_val_loss)
            self.schedulers[1].step(new_val_loss)

        print("Finished Training")
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

    def run_eval(self, val_dataloader):
        """
        Evaluate the model on the validation set and visualize some results
        on wandb
        :- val_dataloader -: torch dataloader
        """
        val_loss = 0.0
        self.model.eval()
        X_1 = None
        Y_depths_1 = None
        Y_segmentations_1 = None
        output_depths_1 = None
        output_segmentations_1 = None
        with torch.no_grad():
            pbar = tqdm(val_dataloader)
            pbar.set_description("Validation")
            for i, (X, Y_depths, Y_segmentations) in enumerate(pbar):
                X, Y_depths, Y_segmentations = (
                    X.to(self.device),
                    Y_depths.to(self.device),
                    Y_segmentations.to(self.device),
                )
                output_depths, output_segmentations = self.model(X)
                assert output_depths is not None, "output_depths is None"
                # assert output_segmentations is not None, "output_segmentations is None"
                # if i==0:
                #     print(f"===the sample image is {X}===")
                #     print(f"===the sample depth is {Y_depths}===")
                output_depths = (
                    output_depths.squeeze(1) if output_depths != None else None
                )
                Y_depths = Y_depths.squeeze(1)
                Y_segmentations = Y_segmentations.squeeze(1)
                if i == 0:
                    X_1 = X
                    Y_depths_1 = Y_depths
                    Y_segmentations_1 = Y_segmentations
                    output_depths_1 = output_depths
                    output_segmentations_1 = output_segmentations
                # get loss
                loss = self.loss_depth(
                    output_depths, Y_depths
                ) + self.loss_segmentation(output_segmentations, Y_segmentations)
                assert isinstance(loss, (torch.Tensor, float, int)), f"Unexpected loss type: {type(loss)}"
                val_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
                pbar.set_postfix({"validation_loss": val_loss / (i + 1)})
            if self.config["wandb"]["enable"]:
                wandb.log({"val_loss": val_loss / (i + 1)})
                self.img_logger(
                    X_1,
                    Y_depths_1,
                    Y_segmentations_1,
                    output_depths_1,
                    output_segmentations_1,
                )

        if 'i' not in locals():
            print("val loader not valid, set loss to 0.0!")
            return 0.0  # 没有正确的遍历valloader 

        else:
            return val_loss / (i + 1)

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
