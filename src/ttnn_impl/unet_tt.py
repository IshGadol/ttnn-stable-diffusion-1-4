from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    TTNNDownBlock,
    TTNNDownBlockConfig,
    TTNNMidBlock,
    TTNNMidBlockConfig,
    TTNNUpBlock,
    TTNNUpBlockConfig,
)


@dataclass
class TTNNUNetConfig:
    """
    Configuration for the TTNN UNet scaffolding.

    This mirrors the key aspects of diffusers' UNet2DConditionModel for SD 1.4.
    """

    height: int = 512
    width: int = 512
    in_channels: int = 4
    out_channels: int = 4

    # Channel layout by stage: [64x64, 32x32, 16x16, 8x8]
    block_out_channels: Sequence[int] = field(
        default_factory=lambda: (320, 640, 1280, 1280)
    )

    # Attention + conditioning
    cross_attention_dim: int = 768
    layers_per_block: int = 2
    num_heads: int = 8

    # Time embedding dimensionality
    time_embed_dim: int = 1280


class TTNNUNet(nn.Module):
    """
    Torch-based UNet for TTNN scaffolding.

    This implementation:
        - Uses TTNNDownBlock / TTNNMidBlock / TTNNUpBlock
        - Follows the SD 1.4 channel and block layout
        - Exposes a forward signature compatible with the TTNN pipeline:

            forward(latents, timesteps, context) -> noise_pred

    Shape contract:
        latents:   (B, 4, 64, 64)
        timesteps: (B,) or scalar
        context:   (B, L, 768)
        out:       (B, 4, 64, 64)
    """

    def __init__(
        self,
        config: TTNNUNetConfig,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.config = config
        self.device_str = device or "ttnn"
        self.dtype = dtype

        block_out = list(config.block_out_channels)
        assert len(block_out) == 4, "Expected 4 UNet stages for SD 1.4"

        # ------------------------------------------------------------------
        # Time embedding: simple learned MLP on scalar timesteps
        # ------------------------------------------------------------------
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.time_embed_dim),
            nn.SiLU(),
            nn.Linear(config.time_embed_dim, config.time_embed_dim),
        )

        # ------------------------------------------------------------------
        # Input convolution (4 -> 320)
        # ------------------------------------------------------------------
        self.conv_in = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=block_out[0],
            kernel_size=3,
            padding=1,
        )

        # ------------------------------------------------------------------
        # Down path
        # ------------------------------------------------------------------
        self.down_blocks = nn.ModuleList()

        # Stage 0: 64x64 -> 32x32, CrossAttnDownBlock2D-like
        self.down_blocks.append(
            TTNNDownBlock(
                TTNNDownBlockConfig(
                    in_channels=block_out[0],
                    out_channels=block_out[0],
                    num_layers=config.layers_per_block,
                    time_embed_dim=config.time_embed_dim,
                    add_attention=True,
                    cross_attention_dim=config.cross_attention_dim,
                    num_heads=config.num_heads,
                    add_downsample=True,
                )
            )
        )

        # Stage 1: 32x32 -> 16x16, CrossAttnDownBlock2D-like
        self.down_blocks.append(
            TTNNDownBlock(
                TTNNDownBlockConfig(
                    in_channels=block_out[0],
                    out_channels=block_out[1],
                    num_layers=config.layers_per_block,
                    time_embed_dim=config.time_embed_dim,
                    add_attention=True,
                    cross_attention_dim=config.cross_attention_dim,
                    num_heads=config.num_heads,
                    add_downsample=True,
                )
            )
        )

        # Stage 2: 16x16 -> 8x8, CrossAttnDownBlock2D-like
        self.down_blocks.append(
            TTNNDownBlock(
                TTNNDownBlockConfig(
                    in_channels=block_out[1],
                    out_channels=block_out[2],
                    num_layers=config.layers_per_block,
                    time_embed_dim=config.time_embed_dim,
                    add_attention=True,
                    cross_attention_dim=config.cross_attention_dim,
                    num_heads=config.num_heads,
                    add_downsample=True,
                )
            )
        )

        # Stage 3: 8x8 -> 8x8, DownBlock2D-like (no attention, no further downsample)
        self.down_blocks.append(
            TTNNDownBlock(
                TTNNDownBlockConfig(
                    in_channels=block_out[2],
                    out_channels=block_out[3],
                    num_layers=config.layers_per_block,
                    time_embed_dim=config.time_embed_dim,
                    add_attention=False,
                    cross_attention_dim=config.cross_attention_dim,
                    num_heads=config.num_heads,
                    add_downsample=False,
                )
            )
        )

        # ------------------------------------------------------------------
        # Mid block (UNetMidBlock2DCrossAttn-like)
        # ------------------------------------------------------------------
        self.mid_block = TTNNMidBlock(
            TTNNMidBlockConfig(
                channels=block_out[3],
                time_embed_dim=config.time_embed_dim,
                cross_attention_dim=config.cross_attention_dim,
                num_heads=config.num_heads,
            )
        )

        # ------------------------------------------------------------------
        # Up path
        # ------------------------------------------------------------------
        self.up_blocks = nn.ModuleList()

        # Up stage 0 (lowest resolution, pair with skip3), UpBlock2D-like (no attention)
        # Does not upsample; stays at 8x8.
        self.up_blocks.append(
            TTNNUpBlock(
                TTNNUpBlockConfig(
                    in_channels=block_out[3],
                    out_channels=block_out[3],
                    num_layers=config.layers_per_block,
                    time_embed_dim=config.time_embed_dim,
                    add_attention=False,
                    cross_attention_dim=config.cross_attention_dim,
                    num_heads=config.num_heads,
                    add_upsample=False,
                )
            )
        )

        # Up stage 1, CrossAttnUpBlock2D-like: 8x8 -> 16x16
        self.up_blocks.append(
            TTNNUpBlock(
                TTNNUpBlockConfig(
                    in_channels=block_out[3],
                    out_channels=block_out[2],
                    num_layers=config.layers_per_block,
                    time_embed_dim=config.time_embed_dim,
                    add_attention=True,
                    cross_attention_dim=config.cross_attention_dim,
                    num_heads=config.num_heads,
                    add_upsample=True,
                )
            )
        )

        # Up stage 2, CrossAttnUpBlock2D-like: 16x16 -> 32x32
        self.up_blocks.append(
            TTNNUpBlock(
                TTNNUpBlockConfig(
                    in_channels=block_out[2],
                    out_channels=block_out[1],
                    num_layers=config.layers_per_block,
                    time_embed_dim=config.time_embed_dim,
                    add_attention=True,
                    cross_attention_dim=config.cross_attention_dim,
                    num_heads=config.num_heads,
                    add_upsample=True,
                )
            )
        )

        # Up stage 3, CrossAttnUpBlock2D-like: 32x32 -> 64x64
        self.up_blocks.append(
            TTNNUpBlock(
                TTNNUpBlockConfig(
                    in_channels=block_out[1],
                    out_channels=block_out[0],
                    num_layers=config.layers_per_block,
                    time_embed_dim=config.time_embed_dim,
                    add_attention=True,
                    cross_attention_dim=config.cross_attention_dim,
                    num_heads=config.num_heads,
                    add_upsample=True,
                )
            )
        )

        # ------------------------------------------------------------------
        # Output head
        # ------------------------------------------------------------------
        self.out_norm = nn.GroupNorm(num_groups=32, num_channels=block_out[0], eps=1e-5)
        self.out_conv = nn.Conv2d(
            in_channels=block_out[0],
            out_channels=config.out_channels,
            kernel_size=3,
            padding=1,
        )

    def _compute_time_emb(self, timesteps: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Simple learned embedding for scalar timesteps.

        Args:
            timesteps: shape (B,) or scalar

        Returns:
            (B, time_embed_dim)
        """
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)

        if timesteps.shape[0] == 1 and batch_size > 1:
            timesteps = timesteps.expand(batch_size)

        t = timesteps.float().view(-1, 1)  # (B, 1)
        t_emb = self.time_embed(t)         # (B, time_embed_dim)
        return t_emb

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latents:   (B, 4, 64, 64)
            timesteps: (B,) or scalar
            context:   (B, L, 768)

        Returns:
            noise_pred: (B, 4, 64, 64)
        """
        B, C, H, W = latents.shape
        assert C == self.config.in_channels, f"Expected {self.config.in_channels} latent channels, got {C}"

        # Compute time embedding
        t_emb = self._compute_time_emb(timesteps, batch_size=B)

        # Input conv
        x = self.conv_in(latents)  # (B, 320, 64, 64)

        # Down path with skip collection
        skips = []
        for down in self.down_blocks:
            x, x_skip = down(x, t_emb, context)
            skips.append(x_skip)

        # Mid block
        x = self.mid_block(x, t_emb, context)

        # Up path: consume skips in reverse order
        skips = skips[::-1]
        for up, x_skip in zip(self.up_blocks, skips):
            x = up(x, x_skip, t_emb, context)

        # Output head
        x = self.out_norm(x)
        x = F.silu(x)
        x = self.out_conv(x)
        return x
