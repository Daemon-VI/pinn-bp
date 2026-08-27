"""1-D convolutional encoder for PPG windows.

Shared by every neural model in the project -- the PINN and the data-only baseline use the
*same* encoder class with the same width and depth. That is the point: when the two are
compared, the encoder is held fixed and only the head and the loss differ, so a difference
in results cannot be attributed to one of them having more capacity.

Design notes:

*Small on purpose.* About 190k parameters. Training runs on a 6-thread CPU with no GPU, and
the synthetic cohort is tens of thousands of 1000-sample windows; a ResNet-scale model would
train for hours and overfit a 200-subject cohort besides.

*Squeeze-excitation over attention.* An SE block is a channel-wise gate costing two small
linear layers. Self-attention over a 1000-sample sequence would cost far more and, at this
data scale, reliably overfits.

*Both average and max pooling.* Average pooling captures sustained morphology; max pooling
captures the sharpest event in the window, which is what the systolic upstroke is. Windows
that contain a motion burst also produce a large max, and keeping both lets the network
represent "there was a spike here" rather than having it averaged into invisibility.

*GroupNorm, not BatchNorm.* Batch statistics are a poor idea here for a specific reason:
the robustness sweep evaluates on batches whose corruption level is homogeneous, so batch
statistics shift systematically between evaluation conditions and BatchNorm would make the
measured degradation partly an artifact of normalisation. GroupNorm is per-sample and has no
such coupling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["PPGEncoder", "SEBlock", "ResidualBlock"]


class SEBlock(nn.Module):
    """Squeeze-and-excitation channel gate."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        w = x.mean(dim=-1)
        w = self.fc2(self.act(self.fc1(w)))
        return x * torch.sigmoid(w).unsqueeze(-1)


class ResidualBlock(nn.Module):
    """Two dilated convolutions, GroupNorm, SE gate, residual connection.

    Dilation widens the receptive field without extra parameters or further downsampling.
    That matters here: the network must relate features separated by a full cardiac cycle
    (a systolic peak and the following beat's foot are ~125 samples apart at 125 Hz and
    70 bpm), and reaching that span by stacking stride-2 layers alone would throw away the
    temporal resolution the derivative channels are carrying.
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1, dilation: int = 1, groups: int = 8):
        super().__init__()
        pad = dilation * (5 - 1) // 2
        self.conv1 = nn.Conv1d(c_in, c_out, kernel_size=5, stride=stride, padding=pad,
                               dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(min(groups, c_out), c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, kernel_size=5, stride=1, padding=pad,
                               dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(min(groups, c_out), c_out)
        self.act = nn.GELU()
        self.se = SEBlock(c_out)

        self.skip: nn.Module
        if stride != 1 or c_in != c_out:
            self.skip = nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(groups, c_out), c_out),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        h = self.se(h)
        # Lengths can differ by one when stride and padding interact with an odd input
        # length; trim rather than pad so no fabricated samples enter the residual sum.
        n = min(h.shape[-1], identity.shape[-1])
        return self.act(h[..., :n] + identity[..., :n])


class PPGEncoder(nn.Module):
    """Encode a multi-channel PPG window into a latent vector.

    Args:
        in_channels: 3 by default -- PPG, VPG, APG.
        widths: channel count per stage.
        latent_dim: size of the output embedding.
        dropout: applied to the pooled embedding only. Dropout inside a convolutional stack
            fights GroupNorm and, at this data scale, mostly slows convergence.

    Shape:
        Input ``(B, C, L)``; output ``(B, latent_dim)``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        widths: tuple[int, ...] = (32, 48, 64, 96),
        latent_dim: int = 128,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, widths[0], kernel_size=9, stride=2, padding=4, bias=False),
            nn.GroupNorm(min(8, widths[0]), widths[0]),
            nn.GELU(),
        )

        blocks: list[nn.Module] = []
        c_prev = widths[0]
        # Dilation doubles with depth so the receptive field grows geometrically and covers
        # more than one full cardiac cycle by the last stage.
        for i, c in enumerate(widths):
            blocks.append(ResidualBlock(c_prev, c, stride=2 if i > 0 else 1, dilation=2**i))
            c_prev = c
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.Linear(2 * c_prev, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        return self.head(pooled)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
