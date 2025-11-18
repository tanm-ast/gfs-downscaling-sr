"""
Generic "sequence-of-frames" dataset:

Expected layout:
  root/
    train/
      sample_0001/
        frames/
          t000.npy (or .pt/.png)  # each file is multi-band [C,H,W] or [H,W,C]
          ...
          t00T.npy  # target frame (t+1)
        weather.json              # optional {"temp":..,"rh":..,"wind":..,"press":..}
    val/
      sample_0002/...

This dataset returns:
  x_seq:   [T, C, H, W]   past frames
  weather: [W]            station features (optional; zeros if missing)
  y_next:  [1, H, W]      next frame (uses first channel of target)
"""
import json, re
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch, imageio.v2 as imageio
from torch.utils.data import Dataset
from .common_transforms import to_float01, resize_hw, standardize_per_band

def _natsort_key(name: str):
    '''
    Used for implement natural sorting (human sorting) pf files/frame objects:
    instead of sorting ["file1", "file10", "file2"] lexicographically (which would give file1, file10, file2), 
    split into ["file", 1], ["file", 10], ["file", 2] and sort by numeric value of digits.
    '''
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]

def _sorted_paths(frames_dir: Path, pattern: str) -> List[Path]:
    '''
    Sort all the pathnames matching pattern using natural sorting (human sorting)
    '''
    return sorted(frames_dir.glob(pattern), key=lambda p: _natsort_key(p.name))

def _load_frame(path: Path) -> torch.Tensor:
    """
    Load a single "frame" file:
      - .npy: [C,H,W] or [H,W,C]
      - .pt/.pth: saved torch tensor [C,H,W] or [H,W,C]
      - .png/.jpg: grayscale or RGB -> convert to [C,H,W]
    """
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(path)
        t = torch.from_numpy(arr)
        if t.ndim == 3 and t.shape[0] not in (1,3,4,6,8,10,12) and t.shape[-1] in (1,3,4,6,8,10,12):
            t = t.permute(2,0,1)  # [H,W,C] -> [C,H,W]
        return t.float()
    if suf in (".pt",".pth"):
        t = torch.load(path)
        if t.ndim == 3 and t.shape[0] not in (1,3,4,6,8,10,12) and t.shape[-1] in (1,3,4,6,8,10,12):
            t = t.permute(2,0,1)
        return t.float()
    # images
    img = imageio.imread(path)
    t = torch.from_numpy(img)
    if t.ndim == 2: t = t.unsqueeze(0)            # [H,W] -> [1,H,W]
    else:           t = t.permute(2,0,1)          # [H,W,C] -> [C,H,W]
    return t.float()

class SequenceFolderDataset(Dataset):
    def __init__(self, root: str, split: str, T: int, H: int, W: int,
                 weather_keys: Optional[list]=None, frame_glob: str="*.*",
                 per_band_mean: Optional[list]=None, per_band_std: Optional[list]=None):
        super().__init__()
        self.root = Path(root); self.split=split; self.T=T; self.size=(H,W)
        self.weather_keys = weather_keys or []
        self.frame_glob = frame_glob
        self.mean = torch.tensor(per_band_mean) if per_band_mean is not None else None
        self.std  = torch.tensor(per_band_std)  if per_band_std  is not None else None

        self.samples = [p for p in (self.root/self.split).iterdir() if p.is_dir()]
        if not self.samples:
            raise FileNotFoundError(f"No samples in {self.root/self.split}. Expected sample_xxx/frames/…")

    def __len__(self): return len(self.samples)

    def _load_weather(self, folder: Path) -> torch.Tensor:
        '''
        Load weather station specific static variables from a json file
        '''
        wfile = folder / "weather.json"
        if not wfile.exists():
            return torch.zeros(len(self.weather_keys), dtype=torch.float32)
        d = json.loads(wfile.read_text())
        return torch.tensor([float(d.get(k, 0.0)) for k in self.weather_keys], dtype=torch.float32)

    def __getitem__(self, idx):
        folder = self.samples[idx]
        frames_dir = folder / "frames"
        fps = _sorted_paths(frames_dir, self.frame_glob)
        assert len(fps) >= self.T+1, f"{folder}: need at least {self.T+1} frames"

        past, target = fps[:self.T], fps[self.T]
        # Load past frames into [T,C,H,W]
        xs = [to_float01(_load_frame(p)) for p in past]  # each [C,H,W], float in [0,1]
        x = torch.stack(xs, dim=0)                       # [T,C,H,W]
        # Load target, keep first channel as regression target [1,H,W]
        y_full = to_float01(_load_frame(target))
        if y_full.ndim == 2: y_full = y_full.unsqueeze(0)
        y = y_full[:1]                                   # [1,H,W]

        # Resize + standardize (optional)
        x = resize_hw(x, self.size)
        y = resize_hw(y, self.size)
        x = standardize_per_band(x, self.mean, self.std)

        w = self._load_weather(folder)
        return x, w, y
