import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Temporal Upsampler ----------
class TemporalUpsampler(nn.Module):
    """Temporal 1D dilated conv + transposed conv to go from 8h -> 48h."""
    def __init__(self, in_channels, hidden=64):
        super().__init__()
        self.head = nn.Conv1d(in_channels, hidden, 3, padding=1)
        self.block1 = nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2)
        self.block2 = nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4)
        self.upsample = nn.ConvTranspose1d(hidden, hidden, kernel_size=6, stride=6)
        self.tail = nn.Conv1d(hidden, in_channels, 1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 3, 4, 1, 2).reshape(-1, C, T)  # [B*H*W, C, T]
        x = F.relu(self.head(x))
        x = F.relu(self.block1(x))
        x = F.relu(self.block2(x))
        x = self.upsample(x)
        x = self.tail(x)
        x = x.reshape(B, H, W, C, -1).permute(0, 3, 4, 1, 2)
        return x  # [B, C, 48, H, W]


# ---------- Spatial SR Module ----------
class ResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return x + out * self.res_scale


class SpatialSR(nn.Module):
    """Frame-wise EDSR-like upsampler."""
    def __init__(self, in_channels, scale=6, num_blocks=20):
        super().__init__()
        self.head = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.resblocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(64, in_channels, 3, padding=1),
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.reshape(B*T, C, H, W)
        x = self.head(x)
        x = self.resblocks(x)
        x = self.upsample(x)
        x = x.reshape(B, T, C, x.shape[-2], x.shape[-1]).permute(0, 2, 1, 3, 4)
        return x


# ---------- Combined Model ----------
class FactorizedSR(nn.Module):
    def __init__(self, in_channels=5):
        super().__init__()
        self.temporal = TemporalUpsampler(in_channels)
        self.spatial = SpatialSR(in_channels)

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        return x
