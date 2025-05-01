# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/antocad/FocusOnDepth and https://github.com/isl-org/DPT

# Resampling code for tactile transformer

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class Read_ignore(nn.Module):
    def __init__(self, start_index=1):
        super(Read_ignore, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        return x[:, self.start_index :]


class Read_add(nn.Module):
    def __init__(self, start_index=1):
        super(Read_add, self).__init__()
        self.start_index = start_index

    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index :] + readout.unsqueeze(1)


class Read_projection(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(Read_projection, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        features = torch.cat((x[:, self.start_index :], readout), -1)
        return self.project(features)


class MyConvTranspose2d(nn.Module):
    def __init__(self, conv, output_size):
        super(MyConvTranspose2d, self).__init__()
        self.output_size = output_size
        self.conv = conv

    def forward(self, x):
        x = self.conv(x, output_size=self.output_size)
        return x


class Resample(nn.Module):
    def __init__(self, p, s, h, emb_dim, resample_dim):
        super(Resample, self).__init__()
        assert s in [4, 8, 16, 32], "s must be in [0.5, 4, 8, 16, 32]"
        self.conv1 = nn.Conv2d(
            emb_dim, resample_dim, kernel_size=1, stride=1, padding=0
        )
        if s == 4:
            self.conv2 = nn.ConvTranspose2d(
                resample_dim,
                resample_dim,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            )
        elif s == 8:
            self.conv2 = nn.ConvTranspose2d(
                resample_dim,
                resample_dim,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            )
        elif s == 16:
            self.conv2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(
                resample_dim,
                resample_dim,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Reassemble(nn.Module):
    def __init__(self, image_size, read, p, s, emb_dim, resample_dim):
        """
        Args:
          image_size:   (C, H_img, W_img) for the *reference* resolution
                        used to initialize a static grid fallback.
          read:         one of "ignore", "add", "projection" (or "project").
          p:            patch size in pixels.
          s:            resample coefficient.
          emb_dim:      D, the transformer embedding dim.
          resample_dim: output channels after Resample.
        """
        super().__init__()
        C, H_img, W_img = image_size

        # static patch-grid dims for fallback
        self.h0 = H_img // p
        self.w0 = W_img // p

        # remember patch size (only used for fallback above)
        self.patch_size = p

        # normalize the read mode
        mode = read.lower()
        if mode == "ignore":
            self.read = Read_ignore()
        elif mode == "add":
            self.read = Read_add()
        elif mode in ("projection", "project"):
            self.read = Read_projection(emb_dim)
        else:
            raise ValueError(f"Unknown read mode {read!r}; expected 'ignore','add','projection'.")

        # final projection / up/downsampling block
        self.resample = Resample(p, s, H_img, emb_dim, resample_dim)

    def forward(self, x: torch.Tensor, hw: tuple = None) -> torch.Tensor:
        """
        Args:
          x:  [B, N, D] transformer tokens, where N == h*w (no class token).
          hw: (h, w) patch-grid dims. If provided, we use it directly
              instead of the static self.h0/self.w0.
        Returns:
          [B, D_resample, h, w] feature map.
        """
        # 1) apply read-mode (drops or fuses class token),
        #    result is [B, N, D]
        x = self.read(x)

        B, N, D = x.shape

        # 2) determine the patch grid
        if hw is None:
            h, w = self.h0, self.w0
        else:
            # hw is already in patch units!
            h, w = hw

        # 3) sanity check
        assert N == h * w, f"Token count {N} != grid {h}×{w}"

        # 4) reshape → [B, D, h, w]
        x = x.transpose(1, 2).reshape(B, D, h, w)

        # 5) run through resample block
        return self.resample(x)
# self design
# class Reassemble(nn.Module):
#     def __init__(self, image_size, read, p, s, emb_dim, resample_dim):
#         """
#         ...
#         read : {"ignore", "add", "projection"}  # we now also accept "project"
#         ...
#         """
#         super().__init__()
#         C, H_img, W_img = image_size

#         # store static grid dims
#         self.h0 = H_img // p
#         self.w0 = W_img // p
#         self.patch_size = p

#         # normalize and alias the read mode
#         mode = read.lower()
#         if mode == "ignore":
#             self.read = Read_ignore()
#         elif mode == "add":
#             self.read = Read_add()
#         elif mode in ("projection", "project"):    # <-- accept both
#             self.read = Read_projection(emb_dim)
#         else:
#             raise ValueError(f"Unknown read mode: {read!r}. "
#                              f"Expected one of 'ignore', 'add', 'projection' or 'project'.")

#         # projection + resample block
#         self.resample = Resample(p, s, H_img, emb_dim, resample_dim)

#     def forward(self, x: torch.Tensor, hw: tuple = None) -> torch.Tensor:
#         """
#         Args:
#             x:  Transformer output tokens, shape [B, N, D].
#                 N = num_patches (h*w) or num_patches+1 if a class token is included.
#             hw: Optional (H_img, W_img) of the *actual* input image.
#                 If given, we recompute the patch‐grid from it:
#                     h = H_img // patch_size, w = W_img // patch_size

#         Returns:
#             A feature map of shape [B, D_resample, h, w].
#         """
#         # 1) apply the chosen readout (this will drop or fuse the class token)
#         #    result is [B, num_patches, D]
#         x = self.read(x)

#         B, N, D = x.shape

#         # 2) determine the patch grid
#         if hw is not None:
#             H_img, W_img = hw
#             h = H_img // self.patch_size
#             w = W_img // self.patch_size
#         else:
#             h, w = self.h0, self.w0

#         # sanity check
#         assert N == h * w, f"Token count {N} != grid {h}×{w}"

#         # 3) reshape tokens → [B, D, h, w]
#         x = x.transpose(1, 2).reshape(B, D, h, w)

#         # 4) run through resample (projection + up/downsampling)
#         x = self.resample(x)
#         return x
    
# class Reassemble(nn.Module):
#     def __init__(self, image_size, read, p, s, emb_dim, resample_dim):
#         """
#         p = patch size
#         s = coefficient resample
#         emb_dim <=> D (in the paper)
#         resample_dim <=> ^D (in the paper)
#         read : {"ignore", "add", "projection"}
#         """
#         super(Reassemble, self).__init__()
#         channels, image_height, image_width = image_size

#         # Read
#         self.read = Read_ignore()
#         if read == "add":
#             self.read = Read_add()
#         elif read == "projection":
#             self.read = Read_projection(emb_dim)

#         # Concat after read
#         self.concat = Rearrange(
#             "b (h w) c -> b c h w",
#             c=emb_dim,
#             h=(image_height // p),
#             w=(image_width // p),
#         )

#         # Projection + Resample
#         self.resample = Resample(p, s, image_height, emb_dim, resample_dim)

#     def forward(self, x):
#         x = self.read(x)
#         x = self.concat(x)
#         x = self.resample(x)
#         return x
