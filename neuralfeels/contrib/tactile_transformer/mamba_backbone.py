# neuralfeels/models/mamba_backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm import Mamba
import timm  # 如果你之前没 import，请加上

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=1024, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size

    def forward(self, x):
        # x: [B, 3, H, W] → [B, D, H', W']
        x = self.proj(x)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # → [B, N, D], N = H*W
        return x, (H, W)


class MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class MambaBackbone(nn.Module):
    def __init__(self, image_size=(3, 384, 384), patch_size=16, emb_dim=1024, num_layers=24):
        super().__init__()
        C, H_img, W_img = image_size
        self.patch_embed = PatchEmbed(in_chans=C, embed_dim=emb_dim, patch_size=patch_size)
        self.patch_size = patch_size

        # 原始 patch 网格尺寸（h0, w0）
        self.h0 = H_img // patch_size
        self.w0 = W_img // patch_size
        self.emb_dim = emb_dim

        # 学习一个静态的 base pos_embed，然后在 forward 时插值
        self.pos_embed = nn.Parameter(torch.zeros(1, self.h0 * self.w0, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # mamba blocks
        self.blocks = nn.ModuleList([MambaBlock(emb_dim) for _ in range(num_layers)])

        # 用来在 forward 里记录实际通过 Conv2D 后的 (H, W)
        self.current_hw = None

    def forward(self, x):
        # x: [B, 3, H, W]
        x, (H, W) = self.patch_embed(x)           # [B, N, D]
        self.current_hw = (H, W)

        # 将静态 pos_embed 从 (1, h0*w0, D) → (1, D, h0, w0)
        p = self.pos_embed.view(1, self.h0, self.w0, self.emb_dim).permute(0, 3, 1, 2)
        # 插值到实际的 (H, W)
        p = F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False)
        # 再 flatten 回 (1, H*W, D)
        p = p.permute(0, 2, 3, 1).reshape(1, H * W, self.emb_dim)

        # 加上动态插值后的位置编码
        x = x + p

        # 串联所有 MambaBlock
        for blk in self.blocks:
            x = blk(x)

        # 最后返回 [B, N, D]，DPTModel 会用 hook 拿各层激活
        return x
