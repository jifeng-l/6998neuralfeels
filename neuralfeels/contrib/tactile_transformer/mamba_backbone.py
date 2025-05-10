# neuralfeels/models/mamba_backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class PatchEmbed(nn.Module):
    """
    Turn an image into patch embeddings via a Conv2d.
    """
    def __init__(self, in_chans=3, embed_dim=1024, patch_size=16):
        super().__init__()
        # conv with stride=kernel_size slices into non-overlapping patches
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.patch_size = patch_size

    def forward(self, x):
        # x: [B, C, H, W] → [B, D, H', W']
        x = self.proj(x)
        B, D, H, W = x.shape
        # Flatten patches → [B, N, D]
        tokens = x.flatten(2).transpose(1, 2)
        return tokens, (H, W)


class MambaBlock(nn.Module):
    """
    A single Mamba SSM block with LayerNorm + residual.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class MambaBackbone(nn.Module):
    """
    Mamba-based “transformer” backbone with a ViT-style cls_token and
    dynamic positional embeddings that interpolate to any input size.
    """
    def __init__(self,
                 image_size=(3, 384, 384),
                 patch_size=16,
                 emb_dim=1024,
                 num_layers=24):
        super().__init__()
        C, H_ref, W_ref = image_size

        # 1) patch embeddings
        self.patch_embed = PatchEmbed(in_chans=C,
                                      embed_dim=emb_dim,
                                      patch_size=patch_size)
        self.patch_size = patch_size

        # 2) class token
        #    one learned vector per embedding dimension
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 3) positional embeddings for (cls + patch_grid)
        self.h0 = H_ref // patch_size
        self.w0 = W_ref // patch_size
        tot = 1 + self.h0 * self.w0    # +1 for cls_token
        self.pos_embed = nn.Parameter(torch.zeros(1, tot, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 4) the stack of MambaBlocks
        self.blocks = nn.ModuleList([MambaBlock(emb_dim)
                                     for _ in range(num_layers)])

        # will hold actual patch-grid dims at runtime
        self.current_hw = None

    def forward(self, x):
        """
        Args:
          x: [B, C, H, W]
        Returns:
          tokens: [B, 1+H'*W', D]
        """
        # 1) patchify + record H',W'
        tokens, (H, W) = self.patch_embed(x)
        self.current_hw = (H, W)

        B = tokens.size(0)

        # 2) prepend cls_token
        cls = self.cls_token.expand(B, -1, -1)  # [B,1,D]
        tokens = torch.cat((cls, tokens), dim=1)  # [B, 1+N, D]

        # 3) split pos_embed into cls + patch parts
        cls_pos   = self.pos_embed[:, :1, :]                   # [1,1,D]
        patch_pos = self.pos_embed[:, 1:, :]                   # [1, h0·w0, D]

        # 4) interpolate patch_pos from (h0,w0) → (H,W)
        p = patch_pos.view(1, self.h0, self.w0, -1)            # [1,h0,w0,D]
        p = p.permute(0, 3, 1, 2)                              # [1,D,h0,w0]
        p = F.interpolate(p, size=(H, W),
                         mode="bilinear",
                         align_corners=False)                # [1,D,H,W]
        p = p.permute(0, 2, 3, 1).reshape(1, H * W, -1)        # [1,H·W,D]

        # 5) re-combine cls_pos + interpolated patch_pos
        pos = torch.cat((cls_pos, p), dim=1)  # [1, 1+H·W, D]

        # 6) add positional embeddings
        tokens = tokens + pos

        # 7) run through MambaBlocks
        for blk in self.blocks:
            tokens = blk(tokens)

        return tokens