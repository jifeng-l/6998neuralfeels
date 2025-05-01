# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/antocad/FocusOnDepth and https://github.com/isl-org/DPT

# Tactile transformer model

import numpy as np
import timm
import torch.nn as nn
import torch 
from neuralfeels.contrib.tactile_transformer.fusion import Fusion
from neuralfeels.contrib.tactile_transformer.head import HeadDepth, HeadSeg
from neuralfeels.contrib.tactile_transformer.reassemble import Reassemble
from neuralfeels.contrib.tactile_transformer.mamba_backbone import MambaBackbone


# class DPTModel(nn.Module):
#     def __init__(
#         self,
#         image_size=(3, 384, 384),
#         patch_size=16,
#         emb_dim=1024,
#         resample_dim=256,
#         read="projection",
#         num_layers_encoder=24,
#         hooks=[5, 11, 17, 23],
#         reassemble_s=[4, 8, 16, 32],
#         transformer_dropout=0,
#         nclasses=2,
#         type="full",
#         model_timm="vit_large_patch16_384",
#         pretrained=False,
#         # "encoder="mamba","vit"
#         model_type="vit",#"mamba"
#     ):
#         """
#         type : {"full", "depth", "segmentation"}
#         image_size : (c, h, w)
#         patch_size : *a square*
#         emb_dim <=> D (in the paper)
#         resample_dim <=> ^D (in the paper)
#         read : {"ignore", "add", "projection"}
#         """
#         super().__init__()
#         # parameterize for mamba encoder
#         if model_type == "mamba":
#             self.transformer_encoders = MambaBackbone(
#                 image_size=image_size,
#                 patch_size=patch_size,
#                 emb_dim=emb_dim,
#                 num_layers=num_layers_encoder
#             )
#         else:
#             self.transformer_encoders = timm.create_model(model_timm, pretrained=pretrained)
#         self.type_ = type

#         # Register hooks
#         self.activation = {}
#         self.hooks = hooks
#         self._get_layers_from_hooks(self.hooks)

#         # Reassembles Fusion
#         self.reassembles = []
#         self.fusions = []
#         for s in reassemble_s:
#             self.reassembles.append(
#                 Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim)
#             )
#             self.fusions.append(Fusion(resample_dim))
#         self.reassembles = nn.ModuleList(self.reassembles)
#         self.fusions = nn.ModuleList(self.fusions)

#         # Head
#         if type == "full":
#             self.head_depth = HeadDepth(resample_dim)
#             self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)
#         elif type == "depth":
#             self.head_depth = HeadDepth(resample_dim)
#             self.head_segmentation = None
#         else:
#             self.head_depth = None
#             self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)

#     def forward(self, img):

#         t = self.transformer_encoders(img)
#         previous_stage = None
#         for i in np.arange(len(self.fusions) - 1, -1, -1, dtype=int):
#             hook_to_take = "t" + str(self.hooks[int(i)])
#             activation_result = self.activation[hook_to_take]
#             reassemble_result = self.reassembles[i](activation_result)
#             fusion_result = self.fusions[i](reassemble_result, previous_stage)
#             previous_stage = fusion_result
#         out_depth = None
#         out_segmentation = None
#         if self.head_depth != None:
#             out_depth = self.head_depth(previous_stage)
#         if self.head_segmentation != None:
#             out_segmentation = self.head_segmentation(previous_stage)
#         return out_depth, out_segmentation

#     def _get_layers_from_hooks(self, hooks: list):
#         def get_activation(name):
#             def hook(model, input, output):
#                 self.activation[name] = output

#             return hook

#         for h in hooks:
#             self.transformer_encoders.blocks[h].register_forward_hook(
#                 get_activation("t" + str(h))
#             )

class DPTModel(nn.Module):
    def __init__(
        self,
        image_size=(3, 384, 384),
        patch_size: int = 16,
        emb_dim: int = 1024,
        resample_dim: int = 256,
        read: str = "projection",
        num_layers_encoder: int = 24,
        hooks: list = [5, 11, 17, 23],
        reassemble_s: list = [4, 8, 16, 32],
        nclasses: int = 2,
        type="full",                        # controls which heads to build
        model_timm: str = "vit_large_patch16_384",
        pretrained: bool = False,
        model_type: str = "mamba",          # "mamba" or any timm ViT
    ):
        """
        Args:
          image_size:    (C, H, W) for reference when using non-mamba ViT.
          type:          one of {"full","depth","segmentation"}, or alias "dpt"→"full".
          model_type:    "mamba" to use MambaBackbone, otherwise a timm ViT.
        """
        super().__init__()

        # Save basic config
        self.image_size = image_size
        self.patch_size = patch_size
        self.hooks = hooks
        self.model_type = model_type

        # --- Normalize and validate the `type` argument ---
        type_str = str(type).lower()
        if type_str in ("full", "dpt", "dptmodel"):
            self.type_ = "full"
        elif type_str == "depth":
            self.type_ = "depth"
        elif type_str in ("segmentation", "seg"):
            self.type_ = "segmentation"
        else:
            valid = "'full','depth','segmentation' (or 'dpt')"
            raise ValueError(f"Unknown model type: {type!r}. Valid options are {valid}.")

        # --- 1) Build the encoder ---
        if model_type == "mamba":
            self.transformer_encoders = MambaBackbone(
                image_size=image_size,
                patch_size=patch_size,
                emb_dim=emb_dim,
                num_layers=num_layers_encoder,
            )
        else:
            self.transformer_encoders = timm.create_model(model_timm, pretrained=pretrained)

        # --- 2) Register forward hooks to capture intermediate activations ---
        self.activation = {}
        self._register_hooks()

        # --- 3) Build multi-scale Reassemble + Fusion stacks ---
        self.reassembles = nn.ModuleList()
        self.fusions = nn.ModuleList()
        for s in reassemble_s:
            self.reassembles.append(
                Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim)
            )
            self.fusions.append(Fusion(resample_dim))

        # --- 4) Build the final prediction heads based on self.type_ ---
        if self.type_ == "full":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)
        elif self.type_ == "depth":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = None
        else:  # "segmentation"
            self.head_depth = None
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)

    def _register_hooks(self):
        """Attach a forward hook on each transformer block to store its output in self.activation."""
        def get_hook(name):
            def hook(module, inp, out):
                self.activation[name] = out
            return hook

        # Locate the transformer blocks for hooking
        if self.model_type == "mamba":
            layers = self.transformer_encoders.blocks
        else:
            # timm ViT sometimes stores them in .blocks or .transformer.encoder.layers
            layers = getattr(self.transformer_encoders, "blocks", None) \
                  or getattr(self.transformer_encoders, "transformer").encoder.layers

        for h in self.hooks:
            layers[h].register_forward_hook(get_hook(f"t{h}"))

    def forward(self, img: torch.Tensor):
        # 1) Clear stale activations
        self.activation.clear()

        # 2) Run the encoder (MambaBackbone sets .current_hw; timm ViT is unaffected)
        _ = self.transformer_encoders(img)

        # 3) Compute the patch-grid height & width
        if self.model_type == "mamba":
            H, W = self.transformer_encoders.current_hw
        else:
            _, H_img, W_img = self.image_size
            H, W = H_img // self.patch_size, W_img // self.patch_size

        # 4) Fuse multi-scale features from highest→lowest resolution
        prev = None
        for i in range(len(self.fusions) - 1, -1, -1):
            key = f"t{self.hooks[i]}"
            if key not in self.activation:
                raise KeyError(f"Missing activation {key}; got {list(self.activation.keys())}")
            act = self.activation[key]                   # shape [B, N, D]
            x = self.reassembles[i](act, hw=(H, W))      # reshape → [B, C, H, W]
            prev = self.fusions[i](x, prev)

        # 5) Apply final heads
        out_depth = self.head_depth(prev) if self.head_depth else None
        out_segmentation = self.head_segmentation(prev) if self.head_segmentation else None

        return out_depth, out_segmentation