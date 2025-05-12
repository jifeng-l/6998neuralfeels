# neuralfeels/models/robust_mamba_backbone.py

import torch
import torch.nn as nn
from mamba_ssm import Mamba

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: tuple[int, int]):
    """
    grid_size = (H, W)
    returns pos_embed: [H*W, embed_dim], sin-cos 2D embeddings.
    """
    H, W = grid_size
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
    sin_inp_x = x_pos * inv_freq.unsqueeze(0)
    emb_x = torch.cat([sin_inp_x.sin(), sin_inp_x.cos()], dim=1)  # [W, d_half]

    # Combine
    grid_emb = torch.zeros(H * W, embed_dim)
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            grid_emb[idx, :d_half] = emb_y[i]
            grid_emb[idx, d_half:] = emb_x[j]
    return grid_emb  # [H*W, D]


class RobustMambaBackbone(nn.Module):
    """
    renew Mamba Transformer backbone,
    Supports placing hooks on several layers and reassembling the hidden states of these layers to output to the upper layer Reassemble.
    """
    def __init__(
        self,
        image_size=(3, 224, 224),
        patch_size=16,
        emb_dim=96,
        num_layers=24,
        hooks: list[int] = None,
    ):
        """
        hooks: list of layer-indices (0-based) to extract hidden-state from.
        """
        super().__init__()
        C, H_ref, W_ref = image_size
        self.emb_dim = emb_dim
        self.hooks = hooks or []

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=C,
            out_channels=emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.patch_size = patch_size

        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Mamba + MLP blocks
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

        # Pre-computed reference patch-grid size
        self.h0 = H_ref // patch_size
        self.w0 = W_ref // patch_size

    def forward(self, x: torch.Tensor):
        """
        Input x: [B, C, H_img, W_img]
        Return:
          features: [B, N, emb_dim * len(hooks)] , N = h*w
          hw: (h, w) Current patch-grid size
        """
        B = x.shape[0]
        # 1) patchify → [B, D, h, w]
        x = self.patch_embed(x)
        D, h, w = x.shape[1:]
        N = h * w

        # 2) flatten + prepend cls → [B, 1+N, D]
        tokens = x.flatten(2).transpose(1, 2)   # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
        tokens = torch.cat([cls, tokens], dim=1)  # (B,1+N,D)

        # 3) 2D sin-cos Position encoding
        pe = get_2d_sincos_pos_embed(self.emb_dim, (h, w)).to(x.device)  # (N,D)
        pe = torch.cat([torch.zeros(1, self.emb_dim, device=x.device), pe], dim=0)  # (1+N,D)
        tokens = tokens + pe.unsqueeze(0)  # broadcast → (B,1+N,D)

        # 4) Run layer by layer and hook
        hooked: list[torch.Tensor] = []
        for idx, blk in enumerate(self.blocks):
            # residual SSM + MLP
            tokens = tokens + blk(tokens)  # (B,1+N,D)
            if idx in self.hooks:
                # Collect the output of this layer
                hooked.append(tokens)

        # If the hook list does not contain the last layer, ensure at least the last layer is added
        if len(self.hooks)==0 or (len(self.hooks)>0 and self.hooks[-1] != len(self.blocks)-1):
            hooked.append(tokens)

        # 5) Drop cls token & concatenate feature dimensions
        #    Each hooked[i] is (B,1+N,D) → slice to (B,N,D)
        feats = torch.cat([h[:, 1:, :] for h in hooked], dim=-1)  # (B, N, D * len(hooked))

        return feats, (h, w)