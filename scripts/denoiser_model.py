"""Compact 3D U-Net used by the latent-only/semantic A/B experiment."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def timestep_embedding(timestep: torch.Tensor, width: int) -> torch.Tensor:
    """Standard sinusoidal embedding for continuous flow time in [0, 1]."""
    half = width // 2
    scales = torch.exp(
        -math.log(10_000) * torch.arange(half, device=timestep.device) / max(half - 1, 1)
    )
    angles = timestep.float()[:, None] * scales[None] * 1_000
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if width % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_width: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_width, 2 * out_channels)
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1)
        )

    def forward(self, features: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(features)))
        scale, shift = self.time(F.silu(time)).chunk(2, dim=-1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1 + scale[:, :, None, None, None])
        hidden = hidden + shift[:, :, None, None, None]
        return self.skip(features) + self.conv2(F.silu(hidden))


class SemanticCrossAttention(nn.Module):
    """Cross-attend a 4x6x6 latent grid to a 4x4x4 semantic grid."""

    def __init__(self, channels: int = 256, semantic_width: int = 1024, heads: int = 8):
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.semantic_norm = nn.LayerNorm(semantic_width, elementwise_affine=False)
        self.semantic_projection = nn.Linear(semantic_width, channels, bias=False)
        self.query_position = nn.Parameter(torch.randn(1, 4 * 6 * 6, channels) * 0.01)
        self.semantic_position = nn.Parameter(torch.randn(1, 4 * 4 * 4, channels) * 0.01)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )

    def forward(self, features: torch.Tensor, semantics: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = features.shape
        if (frames, height, width) != (4, 6, 6):
            raise ValueError(f"Expected bottleneck [B,C,4,6,6], got {tuple(features.shape)}")
        if semantics.shape[1:] != (64, 1024):
            raise ValueError(f"Expected semantics [B,64,1024], got {tuple(semantics.shape)}")

        query = features.flatten(2).transpose(1, 2)
        query_input = self.query_norm(query) + self.query_position

        # L2 normalization makes the semantic scale identical in both experiment arms.
        # A zero tensor stays exactly zero through normalization and the bias-free projection.
        normalized = F.normalize(semantics.float(), dim=-1).to(features.dtype)
        context = self.semantic_projection(self.semantic_norm(normalized))
        active = semantics.detach().abs().sum((1, 2), keepdim=True).gt(0).to(context.dtype)
        context = context + active * self.semantic_position

        attended, _ = self.attention(query_input, context, context, need_weights=False)
        query = query + attended
        query = query + self.ff(self.ff_norm(query))
        return query.transpose(1, 2).reshape(batch, channels, frames, height, width)


class Denoiser(nn.Module):
    """Predict the B-chunk flow velocity from history, noisy B, time, and semantics.

    Shapes:
        history/noisy: [B, 16, 4, 24, 24]
        timestep:       [B]
        semantics:      [B, 64, 1024], all zeros for the latent-only arm
        output:         [B, 16, 4, 24, 24]
    """

    def __init__(self, time_width: int = 256):
        super().__init__()
        self.time_width = time_width
        self.time_mlp = nn.Sequential(
            nn.Linear(time_width, time_width), nn.SiLU(), nn.Linear(time_width, time_width)
        )
        self.input = nn.Conv3d(32, 64, 3, padding=1)
        self.encoder_24 = ResBlock3D(64, 64, time_width)
        self.down_12 = nn.Conv3d(64, 128, 3, stride=(1, 2, 2), padding=1)
        self.encoder_12 = ResBlock3D(128, 128, time_width)
        self.down_6 = nn.Conv3d(128, 256, 3, stride=(1, 2, 2), padding=1)
        self.bottleneck_in = ResBlock3D(256, 256, time_width)
        self.semantic_attention = SemanticCrossAttention(256, 1024, 8)
        self.bottleneck_out = ResBlock3D(256, 256, time_width)
        self.up_12 = nn.ConvTranspose3d(
            256, 128, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.decoder_12 = ResBlock3D(256, 128, time_width)
        self.up_24 = nn.ConvTranspose3d(
            128, 64, (1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1)
        )
        self.decoder_24 = ResBlock3D(128, 64, time_width)
        self.output_norm = nn.GroupNorm(32, 64)
        self.output = nn.Conv3d(64, 16, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        history: torch.Tensor,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        semantics: torch.Tensor,
    ) -> torch.Tensor:
        expected = (16, 4, 24, 24)
        if history.shape[1:] != expected or noisy.shape != history.shape:
            raise ValueError(
                f"Expected matching history/noisy [B,16,4,24,24], got "
                f"{tuple(history.shape)} and {tuple(noisy.shape)}"
            )
        time = self.time_mlp(timestep_embedding(timestep, self.time_width))
        level_24 = self.encoder_24(self.input(torch.cat((history, noisy), dim=1)), time)
        level_12 = self.encoder_12(self.down_12(level_24), time)
        bottleneck = self.bottleneck_in(self.down_6(level_12), time)
        bottleneck = self.semantic_attention(bottleneck, semantics)
        bottleneck = self.bottleneck_out(bottleneck, time)
        level_12_out = self.decoder_12(
            torch.cat((self.up_12(bottleneck), level_12), dim=1), time
        )
        level_24_out = self.decoder_24(
            torch.cat((self.up_24(level_12_out), level_24), dim=1), time
        )
        return self.output(F.silu(self.output_norm(level_24_out)))


if __name__ == "__main__":
    model = Denoiser()
    batch = 2
    result = model(
        torch.randn(batch, 16, 4, 24, 24),
        torch.randn(batch, 16, 4, 24, 24),
        torch.rand(batch),
        torch.randn(batch, 64, 1024),
    )
    print({"parameters": sum(p.numel() for p in model.parameters()), "shape": result.shape})
