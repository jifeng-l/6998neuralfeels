# neuralfeels/models/robust_mamba_backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: tuple[int, int]):
    """
    grid_size = (H, W)
    returns pos_embed: [H*W, embed_dim], sin-cos 2D embeddings.
    """
    H, W = grid_size
    # 每个轴用 embed_dim//2 维：h_emb + w_emb = embed_dim
    assert embed_dim % 2 == 0, "embed_dim must be even for 2D sincos"
    d_half = embed_dim // 2

    # 1D sincos for height
    y_pos = torch.arange(H, dtype=torch.float32).unsqueeze(1)  # [H,1]
    dim = torch.arange(d_half // 2, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (2 * dim / d_half))
    sin_inp_y = y_pos * inv_freq.unsqueeze(0)  # [H, d_half//2]
    emb_y = torch.cat([sin_inp_y.sin(), sin_inp_y.cos()], dim=1)  # [H, d_half]

    # 1D sincos for width
    x_pos = torch.arange(W, dtype=torch.float32).unsqueeze(1)  # [W,1]
    sin_inp_x = x_pos * inv_freq.unsqueeze(0)  # reuse same inv_freq
    emb_x = torch.cat([sin_inp_x.sin(), sin_inp_x.cos()], dim=1)  # [W, d_half]

    # outer combine: 对称地把 y-emb 和 x-emb 交叉拼接
    # 最终获得 [H*W, embed_dim]，坐标 (i,j) 对应位置 idx = i*W + j
    grid_emb = torch.zeros(H * W, embed_dim)
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            grid_emb[idx, :d_half] = emb_y[i]
            grid_emb[idx, d_half:] = emb_x[j]

    return grid_emb  # [H*W, D]


class RobustMambaBackbone(nn.Module):
    """
    基于 Mamba 的 SSM “变种 Transformer”骨干，使用 2D sin-cos 相对位置编码，
    patch_size 默认 8，可动态适配任意分辨率。
    """
    def __init__(
        self,
        image_size=(3, 384, 384),
        patch_size=8,
        emb_dim=1024,
        num_layers=24,
    ):
        super().__init__()
        C, H_ref, W_ref = image_size

        # 1) Patch embedding (stride=patch_size)
        self.patch_embed = nn.Conv2d(
            in_channels=C,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.patch_size = patch_size

        # 2) cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 3) Mamba blocks
        self.blocks = nn.ModuleList([  
            nn.Sequential(
                nn.LayerNorm(emb_dim),
                Mamba(d_model=emb_dim),
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, emb_dim * 4),
                nn.GELU(),
                nn.Linear(emb_dim * 4, emb_dim),
            )  
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        """
        x: [B, C, H, W]
        return: tokens: [B, 1+H'*W', D]
        """
        B = x.size(0)

        # 1) patchify via conv → [B, D, H', W']
        x = self.patch_embed(x)
        D, H, W = x.shape[1:]
        N = H * W

        # 2) flatten → [B, N, D]
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # 3) prepend cls token → [B, 1+N, D]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)

        # 4) 2D Sin-Cos 相对位置编码
        #    生成 [N, D]，然后 repeat B 次、在最前面插一个 zeros 吻合 cls
        device = tokens.device
        pe_grid = get_2d_sincos_pos_embed(self.cls_token.shape[-1], (H, W))  # [N,D]
        pe_grid = pe_grid.to(device)
        # 在最前面给 cls_token 一个零偏置
        pe = torch.cat([torch.zeros(1, pe_grid.size(1), device=device), pe_grid], dim=0)
        tokens = tokens + pe.unsqueeze(0)  # shape (1+N,D) → broadcast to (B,1+N,D)

        # 5) through Mamba+MLP blocks
        for blk in self.blocks:
            # each blk is sequential: LN→SSM→LN→MLP
            tokens = tokens + blk(tokens)  

        return tokens