from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TTNNResBlockConfig:
    in_channels: int
    out_channels: int
    time_embed_dim: int
    groups: int = 32


class TTNNResBlock(nn.Module):
    """
    Torch-based ResNet block used as a scaffolding target for TTNN.

    This is conceptually aligned with diffusers' ResnetBlock2D:

        - GroupNorm -> SiLU -> Conv
        - Add projected time embedding
        - SiLU -> Conv
        - Optional 1x1 conv for channel-matching skip

    Shape contract:
        x:    (B, C_in, H, W)
        t_emb:(B, time_embed_dim)
        out:  (B, C_out, H, W)
    """

    def __init__(self, config: TTNNResBlockConfig) -> None:
        super().__init__()

        self.config = config
        C_in = config.in_channels
        C_out = config.out_channels
        groups = config.groups
        t_dim = config.time_embed_dim

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=C_in, eps=1e-5)
        self.conv1 = nn.Conv2d(C_in, C_out, kernel_size=3, padding=1)

        # Time embedding projection to channels
        self.time_proj = nn.Linear(t_dim, C_out)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=C_out, eps=1e-5)
        self.conv2 = nn.Conv2d(C_out, C_out, kernel_size=3, padding=1)

        if C_in != C_out:
            self.skip_conv = nn.Conv2d(C_in, C_out, kernel_size=1)
        else:
            self.skip_conv = None

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C_in, H, W)
            t_emb: (B, time_embed_dim)

        Returns:
            (B, C_out, H, W)
        """
        residual = x

        # First norm + activation + conv
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        # Time embedding: project and add as bias
        # t_emb: (B, time_embed_dim) -> (B, C_out, 1, 1)
        t_out = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        x = x + t_out

        # Second norm + activation + conv
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        # Skip connection
        if self.skip_conv is not None:
            residual = self.skip_conv(residual)

        return x + residual


@dataclass
class TTNNCrossAttentionConfig:
    """
    Configuration for TTNN-style cross-attention.

    channels:            feature channels (C) of the UNet stage
    cross_attention_dim: dimension of the context embeddings (e.g., 768 for CLIP)
    num_heads:           number of attention heads
    """
    channels: int
    cross_attention_dim: int
    num_heads: int


class TTNNCrossAttentionBlock(nn.Module):
    """
    Torch-based cross-attention block between spatial features and text context.

    This is a scaffolding target for a future TTNN implementation.

    Inputs:
        x:       (B, C, H, W)
        context: (B, L, D_ctx)  (e.g. D_ctx = 768)

    Behavior:
        - flatten spatial to tokens (HW)
        - project x and context into a shared C-dim space
        - compute multi-head attention: Q from x, K/V from context
        - project back to (B, C, H, W) and add residual
    """

    def __init__(self, config: TTNNCrossAttentionConfig) -> None:
        super().__init__()

        self.config = config
        C = config.channels
        D_ctx = config.cross_attention_dim
        num_heads = config.num_heads

        if C % num_heads != 0:
            raise ValueError(f"channels={C} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = C // num_heads

        # Project spatial features into attention space
        self.in_proj = nn.Linear(C, C)

        # Project context (e.g. CLIP embeddings) into same C-dim space
        self.context_proj = nn.Linear(D_ctx, C)

        # Q/K/V projections
        self.to_q = nn.Linear(C, C)
        self.to_k = nn.Linear(C, C)
        self.to_v = nn.Linear(C, C)

        # Output projection back to C channels
        self.to_out = nn.Linear(C, C)

        self.norm = nn.LayerNorm(C)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, N, C) -> (B, num_heads, N, head_dim)
        """
        B, N, C = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, num_heads, N, head_dim) -> (B, N, C)
        """
        B, H, N, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * Dh)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, C, H, W)
            context: (B, L, D_ctx)

        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        _, L, D_ctx = context.shape

        # Flatten spatial dimensions to tokens
        x_tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, HW, C)

        # Project context into same C-dim space
        ctx_proj = self.context_proj(context)  # (B, L, C)

        # LayerNorm on spatial tokens
        x_norm = self.norm(x_tokens)

        # Linear projection to Q / K / V
        x_proj = self.in_proj(x_norm)         # (B, HW, C)
        q = self.to_q(x_proj)                 # (B, HW, C)
        k = self.to_k(ctx_proj)               # (B, L, C)
        v = self.to_v(ctx_proj)               # (B, L, C)

        # Reshape for multi-head attention
        q = self._reshape_heads(q)  # (B, heads, HW, head_dim)
        k = self._reshape_heads(k)  # (B, heads, L,  head_dim)
        v = self._reshape_heads(v)  # (B, heads, L,  head_dim)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, heads, HW, L)
        attn_weights = torch.softmax(attn_scores, dim=-1)           # (B, heads, HW, L)

        attn_out = torch.matmul(attn_weights, v)  # (B, heads, HW, head_dim)

        # Merge heads and project back
        attn_out = self._merge_heads(attn_out)    # (B, HW, C)
        attn_out = self.to_out(attn_out)          # (B, HW, C)

        # Residual connection
        x_tokens_out = x_tokens + attn_out        # (B, HW, C)

        # Reshape back to (B, C, H, W)
        x_out = x_tokens_out.view(B, H, W, C).permute(0, 3, 1, 2)
        return x_out


@dataclass
class TTNNDownBlockConfig:
    in_channels: int
    out_channels: int
    num_layers: int
    time_embed_dim: int
    add_attention: bool
    cross_attention_dim: int = 768
    num_heads: int = 8
    add_downsample: bool = True


class TTNNDownBlock(nn.Module):
    """
    Torch-based down block for TTNN UNet scaffolding.

    Roughly mirrors diffusers' CrossAttnDownBlock2D / DownBlock2D:

        [ResBlock (+ CrossAttn)] x num_layers
        optional downsample (stride-2 conv)

    Returns:
        x_down:  (B, out_channels, H_out, W_out)
        x_skip:  (B, out_channels, H,     W)      # before downsample
    """

    def __init__(self, config: TTNNDownBlockConfig) -> None:
        super().__init__()

        self.config = config
        self.resblocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList() if config.add_attention else None

        C_in = config.in_channels
        C_out = config.out_channels

        # Build ResBlocks (with possibly changing channel sizes)
        in_c = C_in
        for i in range(config.num_layers):
            out_c = C_out
            rb_cfg = TTNNResBlockConfig(
                in_channels=in_c,
                out_channels=out_c,
                time_embed_dim=config.time_embed_dim,
                groups=32,
            )
            self.resblocks.append(TTNNResBlock(rb_cfg))
            in_c = out_c

            if config.add_attention:
                att_cfg = TTNNCrossAttentionConfig(
                    channels=out_c,
                    cross_attention_dim=config.cross_attention_dim,
                    num_heads=config.num_heads,
                )
                self.attn_blocks.append(TTNNCrossAttentionBlock(att_cfg))

        # Downsample conv (stride-2) if enabled
        if config.add_downsample:
            self.downsample = nn.Conv2d(C_out, C_out, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:       (B, C_in, H, W)
            t_emb:   (B, time_embed_dim)
            context: (B, L, D_ctx)

        Returns:
            x_down: (B, C_out, H_out, W_out)
            x_skip: (B, C_out, H,     W)
        """
        # ResBlocks (+ optional attention)
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, t_emb)
            if self.attn_blocks is not None:
                attn_block = self.attn_blocks[i]
                x = attn_block(x, context)

        x_skip = x

        # Downsample if configured
        if self.downsample is not None:
            x = self.downsample(x)

        return x, x_skip


@dataclass
class TTNNMidBlockConfig:
    channels: int
    time_embed_dim: int
    cross_attention_dim: int = 768
    num_heads: int = 8


class TTNNMidBlock(nn.Module):
    """
    Torch-based mid-block at the bottleneck resolution.

    Pattern:
        ResBlock -> CrossAttn -> ResBlock
    """

    def __init__(self, config: TTNNMidBlockConfig) -> None:
        super().__init__()

        self.config = config
        C = config.channels

        self.res1 = TTNNResBlock(
            TTNNResBlockConfig(
                in_channels=C,
                out_channels=C,
                time_embed_dim=config.time_embed_dim,
                groups=32,
            )
        )
        self.attn = TTNNCrossAttentionBlock(
            TTNNCrossAttentionConfig(
                channels=C,
                cross_attention_dim=config.cross_attention_dim,
                num_heads=config.num_heads,
            )
        )
        self.res2 = TTNNResBlock(
            TTNNResBlockConfig(
                in_channels=C,
                out_channels=C,
                time_embed_dim=config.time_embed_dim,
                groups=32,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, C, H, W)
            t_emb:   (B, time_embed_dim)
            context: (B, L, D_ctx)

        Returns:
            (B, C, H, W)
        """
        x = self.res1(x, t_emb)
        x = self.attn(x, context)
        x = self.res2(x, t_emb)
        return x


@dataclass
class TTNNUpBlockConfig:
    in_channels: int
    out_channels: int
    num_layers: int
    time_embed_dim: int
    add_attention: bool
    cross_attention_dim: int = 768
    num_heads: int = 8
    add_upsample: bool = True


class TTNNUpBlock(nn.Module):
    """
    Torch-based up block for TTNN UNet scaffolding.

    Roughly mirrors diffusers' UpBlock2D / CrossAttnUpBlock2D:

        - optional upsample
        - concat skip
        - [ResBlock (+ CrossAttn)] x num_layers

    Args expect:
        x:      (B, C_in,  H_in,  W_in)
        x_skip: (B, C_skip,H_out, W_out)  # same spatial size as upsampled x

    Returns:
        x_out:  (B, C_out, H_out, W_out)
    """

    def __init__(self, config: TTNNUpBlockConfig) -> None:
        super().__init__()

        self.config = config
        self.resblocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList() if config.add_attention else None

        C_in = config.in_channels
        C_out = config.out_channels

        # The first ResBlock sees concatenated channels: C_in + C_skip
        # We'll handle C_skip at runtime by adjusting the in_channels.
        # Here we track "current" channels through the block.
        self.first_in_channels = None  # set at runtime

        # For simplicity, we assume all ResBlocks have out_channels = C_out
        for _ in range(config.num_layers):
            # placeholder; actual in_channels will be set dynamically on first build
            rb_cfg = TTNNResBlockConfig(
                in_channels=C_out,   # will be overwritten dynamically
                out_channels=C_out,
                time_embed_dim=config.time_embed_dim,
                groups=32,
            )
            self.resblocks.append(TTNNResBlock(rb_cfg))

            if config.add_attention:
                att_cfg = TTNNCrossAttentionConfig(
                    channels=C_out,
                    cross_attention_dim=config.cross_attention_dim,
                    num_heads=config.num_heads,
                )
                self.attn_blocks.append(TTNNCrossAttentionBlock(att_cfg))

        self.upsample = config.add_upsample

    def forward(
        self,
        x: torch.Tensor,
        x_skip: torch.Tensor,
        t_emb: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:      (B, C_in,  H_in,  W_in)
            x_skip: (B, C_skip,H_out, W_out)
            t_emb:  (B, time_embed_dim)
            context:(B, L, D_ctx)

        Returns:
            (B, C_out, H_out, W_out)
        """
        B, C_in, H_in, W_in = x.shape
        B2, C_skip, H_skip, W_skip = x_skip.shape
        assert B == B2, "Batch size mismatch between x and x_skip"

        # Optional upsample
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        # Now x should match spatial dims of skip
        _, _, H, W = x.shape
        if (H, W) != (H_skip, W_skip):
            # In practice you may want stricter enforcement or more complex resizing,
            # but nearest-neighbor is fine for scaffolding.
            x = F.interpolate(x, size=(H_skip, W_skip), mode="nearest")

        # Concatenate skip features
        x = torch.cat([x, x_skip], dim=1)  # (B, C_in + C_skip, H, W)

        # Build ResBlocks with correct "first" in_channels if not set
        if self.first_in_channels is None:
            total_in = C_in + C_skip
            # Rebuild the first ResBlock to match input channels
            rb0_cfg = TTNNResBlockConfig(
                in_channels=total_in,
                out_channels=self.resblocks[0].config.out_channels,
                time_embed_dim=self.resblocks[0].config.time_embed_dim,
                groups=self.resblocks[0].config.groups,
            )
            self.resblocks[0] = TTNNResBlock(rb0_cfg)
            self.first_in_channels = total_in

        # Apply ResBlocks (+ optional attention)
        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, t_emb)
            if self.attn_blocks is not None:
                attn_block = self.attn_blocks[i]
                x = attn_block(x, context)

        return x
