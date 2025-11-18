"""Simple image tensor helpers used by datasets."""
import torch, torchvision.transforms.functional as TF
from typing import Tuple, Optional

def to_float01(x: torch.Tensor) -> torch.Tensor:
    """Convert integer images to float in [0,1]; keep float tensors as-is."""
    if x.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
        return x.float() / 255.0
    return x.float()

def resize_hw(x: torch.Tensor, size_hw: Tuple[int,int]) -> torch.Tensor:
    """Resize [C,H,W] or [T,C,H,W] to size_hw using antialias=True."""
    if x.ndim == 3:
        return TF.resize(x, size_hw, antialias=True)
    elif x.ndim == 4:
        return torch.stack([TF.resize(x[t], size_hw, antialias=True) for t in range(x.shape[0])], dim=0)
    raise ValueError("Expected [C,H,W] or [T,C,H,W]")

def standardize_per_band(x: torch.Tensor, mean: Optional[torch.Tensor], std: Optional[torch.Tensor]) -> torch.Tensor:
    """Standardize per-channel using provided mean/std (both 1D of length C)."""
    if mean is None or std is None: return x
    if x.ndim == 4:
        return (x - mean.view(1,-1,1,1)) / (std.view(1,-1,1,1) + 1e-6)
    elif x.ndim == 3:
        return (x - mean.view(-1,1,1)) / (std.view(-1,1,1) + 1e-6)
    return x
