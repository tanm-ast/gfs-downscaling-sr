'''
GFS-like regional downscaling preprocessing & Dataset utilities.

- Expects **one NetCDF per day** containing 5 variables:
    t2m, d2m, r2, wspd, sp
  with 1-hour temporal resolution and exactly 48 forecast hours.

- HR grid: 0.25° lat-lon within
    lat: 40N .. 5N (descending or ascending is OK)
    lon: 65E .. 100E
  => ~141 x 141 grid points.

- LR (Low Resolution) input is created from HR (High Resolution) data by:
    - Temporal downsample: 1h -> 6h (48 -> 8)
    - Spatial downsample: 0.25° -> 1.5° (=×6) via low-pass (gaussian) + decimation

- Targets are the **original HR** fields (hourly, 0.25°).

- Provides:
    1) Split utilities (train/val/test by date)
    2) Stats computation (mean/std per variable from TRAIN only)
    3) Preprocessing functions (build LR from HR)
    4) A PyTorch Dataset that yields (LR_seq, HR_seq) patch pairs
       suited to both Factorized (Option A) & Diffusion models.

Author: Tanmay Asthana
'''

from __future__ import annotations
import os
import re
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Sequence, Callable

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import Dataset


# ------------------------------
# Configuration / constants
# ------------------------------

DEFAULT_VARS = ('t2m', 'd2m', 'r2', 'wspd', 'sp')

@dataclass
class DomainConfig:
    '''
    Spatial/temporal downscaling factors and patch sizes.
    Created as a separate class to make the code adaptable to changing factors/patch sizes.
    '''
    temporal_factor: int = 6           # 48h -> 8 steps
    spatial_factor: int  = 6           # 0.25° -> 1.5°
    hr_patch: int        = 96          # HR patch size (divisible by spatial_factor)
    
    # Derived LR patch size (hr_patch / spatial_factor)
    @property
    def lr_patch(self) -> int:
        assert self.hr_patch % self.spatial_factor == 0, "hr_patch must be divisible by spatial_factor"
        return self.hr_patch // self.spatial_factor


@dataclass
class VarStats:
    '''Per-variable normalization stats computed on TRAIN only.'''
    mean: Dict[str, float]
    std: Dict[str, float]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"mean": self.mean, "std": self.std}, f, indent=2)

    @staticmethod
    def load(path: str) -> "VarStats":
        with open(path, "r") as f:
            d = json.load(f)
        return VarStats(mean=d["mean"], std=d["std"])


# ------------------------------
# Utility: filename -> date
# ------------------------------

_DATE_RE = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")

def extract_yyyymmdd_from_name(path: str) -> Optional[str]:
    '''
    Parse YYYYMMDD from filename (or any digits in the path).
    Accepts '20210301', '2021-03-01', '..._20210301_...' etc.
    Returns 'YYYYMMDD' string or None if not found.
    '''
    m = _DATE_RE.search(os.path.basename(path))
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{y}{mo}{d}"


# ------------------------------
# Mask helper
# ------------------------------

def _to_float_with_nan(da: xr.DataArray) -> np.ndarray:
    '''
    Convert DataArray (which may hold a masked array) to a float array with NaNs for invalids.
    '''
    # Prefer xarray's mask if present
    arr = da.to_masked_array(copy=False)  # np.ma.MaskedArray or ndarray
    if isinstance(arr, np.ma.MaskedArray):
        out = arr.filled(np.nan).astype(np.float32, copy=False)
    else:
        out = np.asarray(arr, dtype=np.float32)
    return out

# ------------------------------
# xarray helpers
# ------------------------------
def _ensure_time_sorted(ds: xr.Dataset) -> xr.Dataset:
    '''Sort by ascending time steps to avoid surprises.'''
    if 'step' in ds.dims or 'step' in ds.coords:
        return ds.sortby('step')
    raise ValueError("Dataset must have a 'step' dimension.")


def _subset_vars(ds: xr.Dataset, variables: Sequence[str]) -> xr.Dataset:
    '''Keep only the variables of interest; raises if any are missing.'''
    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        raise KeyError(f"Variables missing in dataset: {missing}")
    return ds[list(variables)]

def _stack_vars_thwc(ds: xr.Dataset, variables) -> np.ndarray:
    '''
    Return complete dataset with shapes [T, H, W, C] in the form of a numpy array,
    with channel ordering same as that in the 'variables' list.
    valid_mask is True where values are valid (not NaN).
    '''
    ds = ds.transpose('step', 'latitude', 'longitude')
    vals = []
    
    for v in variables:
        # a = _to_float_with_nan(ds[v])     # [T,H,W] float32 with NaNs
        a = np.asarray(ds[v].values, dtype=np.float32)  # [T,H,W], no NaNs assumed
        vals.append(a)
        
    vals_stack = np.stack(vals, axis=-1)           # [T,H,W,C]
    
    return vals_stack

# ------------------------------
# Masked numpy array helpers
# ------------------------------

def _gaussian_downsample2d(arr_thwc: np.ndarray, factor: int, sigma: float = 2.0,mode: str = "nearest",
    nan_safe: bool = False) -> np.ndarray:
    '''
    Spatial low-pass + decimate by 'factor' on the dims (H,W).
    Implementation: convolve data*weights and weights separately, divide.
    
    If nan_safe=False (default): assume no NaNs -> single Gaussian filter pass on 'arr' (fast).
    If nan_safe=True: normalized convolution (data*mask / mask) for robustness (slower).

    
    Expects arr[T, H, W, C]. 
    Returns shape [T, H//factor, W//factor, C].
    '''
    if not nan_safe:
        # Blur only H,W axes: sigma tuple = (T=0, H=sigma, W=sigma, C=0)
        lp = gaussian_filter(arr_thwc, sigma=(0, sigma, sigma, 0), mode=mode)
        return lp[:, ::factor, ::factor, :].astype(np.float32, copy=False)

    # NaN-aware version
    w = (~np.isnan(arr_thwc)).astype(np.float32)
    a = np.nan_to_num(arr_thwc, copy=False, nan=0.0).astype(np.float32,copy=False)

    num = gaussian_filter(a * w, sigma=(0, sigma, sigma, 0), mode="nearest")
    den = gaussian_filter(w,     sigma=(0, sigma, sigma, 0), mode="nearest")
       
    out = num / np.maximum(den, 1e-8)

    # Where den==0 (all NaN neighborhood), keep NaN
    out[den < 1e-8] = np.nan

    return out[:, ::factor, ::factor, :]

    
def _downsample_time(arr_thwc: np.ndarray, factor: int, method: str = "nearest") -> np.ndarray:
    '''
    Temporal downsample with NaN handling from T -> T//factor.
    - 'nearest': pick frames 0, f, 2f, ... (preserves NaNs)
    - 'mean': block-average ignoring NaNs
    '''
    T = arr_thwc.shape[0]
    T2 = T // factor
    if method == "nearest":
        idx = np.arange(0, T2 * factor, factor)
        return arr_thwc[idx]
    elif method == "mean":
        trimmed = arr_thwc[:T2 * factor]
        blk = trimmed.reshape(T2, factor, *trimmed.shape[1:])  # [T2, f, H, W, C]
        # with np.errstate(invalid="ignore"):
        #     # nanmean over block axis=1
        #     return np.nanmean(blk, axis=1)
        return blk.mean(axis=1)   # [T,H,W,C], no NaNs assumed
    else:
        raise ValueError("method must be 'nearest' or 'mean'.")


# ------------------------------
# Core: build LR (Low Resolution) inputs from HR (High Resolution) ds
# ------------------------------

def build_lr_from_hr(
    ds_hr: xr.Dataset,
    variables: Sequence[str] = DEFAULT_VARS,
    temporal_factor: int = 6,
    spatial_factor: int = 6,
    temporal_method: str = "nearest",  # or "mean"
    spatial_sigma: float = 2.5,
    nan_safe_spatial: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Create LR inputs and HR targets from a single-day HR Dataset (1h, 0.25°).
    with mask propagation.

    Returns:
      X_lr:   [T_lr, H_lr, W_lr, C]  float32 with NaNs for invalids
      Y_hr:   [T_hr, H_hr, W_hr, C]  float32 with NaNs for invalids
    '''
    # 1) keep only variables and sort time
    ds_hr = _subset_vars(_ensure_time_sorted(ds_hr), variables)

    # 2) convert to numpy arrays [T,H,W,C]
    # HR numpy arrays with all variables stacked in channel dim
    y_hr = _stack_vars_thwc(ds_hr, variables).astype(np.float32, copy=False)   # [T_hr, H_hr, W_hr, C]

    # 3) temporal downsample
    x_tmp = _downsample_time(y_hr, factor=temporal_factor, method=temporal_method)  # [T_lr,H_hr,W_hr,C]

    # 4) spatial downsample (low-pass + decimate)
    x_lr = _gaussian_downsample2d(x_tmp, factor=spatial_factor, sigma=spatial_sigma, mode="nearest", 
                                  nan_safe=nan_safe_spatial)
    

    return x_lr, y_hr


# ------------------------------
# Splits
# ------------------------------

def split_by_date(
    nc_files: List[str],
    train_start: str, train_end: str,
    val_start: str,   val_end: str,
    test_start: str,  test_end: str,
) -> Tuple[List[str], List[str], List[str]]:
    '''
    Split files by inclusive date ranges 'YYYYMMDD'.

    Each file path must contain a date string (YYYYMMDD or YYYY-MM-DD).
    '''
    def in_range(yyyymmdd: str, lo: str, hi: str) -> bool:
        return (yyyymmdd >= lo) and (yyyymmdd <= hi)

    train_files, val_files, test_files = [], [], []
    for p in nc_files:
        d = extract_yyyymmdd_from_name(p)
        if d is None:
            continue
        if in_range(d, train_start, train_end):
            train_files.append(p)
        elif in_range(d, val_start, val_end):
            val_files.append(p)
        elif in_range(d, test_start, test_end):
            test_files.append(p)
    return sorted(train_files), sorted(val_files), sorted(test_files)


def split_by_ratio(
    files: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split list of files by ratios. (train, val, test)
    Deterministic with a fixed seed.
    """
    assert 0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    return train_files, val_files, test_files



# ------------------------------
# Stats (TRAIN only)
# ------------------------------

def compute_var_stats_train(
    train_files: List[str],
    variables: Sequence[str] = DEFAULT_VARS,
    chunk_limit: Optional[int] = None,
) -> VarStats:
    '''
    Compute per-variable mean/std **from HR targets** over TRAIN set.
    Optionally limit to first N files for speed (chunk_limit).
    '''
    sums = {v: 0.0 for v in variables}
    sums2 = {v: 0.0 for v in variables}
    counts = {v: 0 for v in variables}

    if chunk_limit is None:
        nfiles = len(train_files)
    else:
        nfiles = min(chunk_limit, len(train_files))

    for p in train_files[:nfiles]:
        ds = xr.open_dataset(p)
        ds = _subset_vars(_ensure_time_sorted(ds), variables).transpose('step', 'latitude', 'longitude')
        for v in variables:
            a = ds[v].values  # [T,H,W]
            sums[v]  += float(a.sum())
            sums2[v] += float((a*a).sum())
            counts[v] += a.size
        ds.close()

    mean = {v: (sums[v] / counts[v]) for v in variables}
    std  = {v: float(np.sqrt(max(sums2[v] / counts[v] - mean[v]**2, 1e-12))) for v in variables}
    return VarStats(mean=mean, std=std)


def apply_standardize(arr_thwc: np.ndarray, stats: VarStats,variables: Sequence[str] = DEFAULT_VARS) -> np.ndarray:
    '''
    Standardize a [T,H,W,C] array **in place** using VarStats.
    Returns a new standardized array (does not modify input).
    '''
    arr = arr_thwc.astype(np.float32, copy=True)
    for ci, v in enumerate(variables):
        m, s = stats.mean[v], stats.std[v]
        arr[..., ci] = (arr[..., ci] - m) / (s + 1e-6)
    return arr


# ------------------------------
# Dataset
# ------------------------------

class DownscalingDataset(Dataset):
    '''
    PyTorch Dataset for spatio-temporal SR training.

    Returns a tuple:
    X_lr:  [C, T_lr, H_lr, W_lr]  (standardized)
    Y_hr:  [C, T_hr, H_hr, W_hr]  (standardized)
    meta:  dict with bookkeeping info (date, indices)

    Two modes:
    - training/validation (random HR patches; matching LR patches)
    - testing/inference (full field tiles without randomness)

    
    - hr_patch should be divisible by spatial_factor; lr_patch = hr_patch/6.
    '''

    def __init__(
        self,
        files: List[str],
        stats: VarStats,
        variables: Sequence[str] = DEFAULT_VARS,
        dom: DomainConfig = DomainConfig(),
        mode: str = "train",            # "train" | "val" | "test"
        temporal_method: str = "nearest",
        spatial_sigma: float = 2.5,
        samples_per_day: int = 16,      # number of random patches per day (train/val)
        seed: int = 42,
        cache_fn: Optional[Callable[[str], Optional[Tuple[np.ndarray,np.ndarray]]]] = None,
        nan_safe_spatial: bool = False
    ):
        self.files = files
        self.variables = list(variables)
        self.stats = stats
        self.dom = dom
        self.mode = mode
        self.temporal_method = temporal_method
        self.spatial_sigma = spatial_sigma
        self.samples_per_day = samples_per_day if mode in ("train","val") else 1
        self.rng = np.random.default_rng(seed)
        self.cache_fn = cache_fn  # optional: provide a function to load/save precomputed (X_lr, Y_hr)
        self.nan_safe_spatial = nan_safe_spatial

        '''
        Build an index: each dataset index corresponds to a (file_idx, sample_k) pair.     
        Use this index to get samples_per_day patches from each file.
        From this tuple index, generate a global index to be used by __getitem__ function.
        E.g., with 1593 days in the dataset and samples_per_day=16, we get
        1,593 x 16 samples per epoch.
        For "test" mode, samples_per_day=1 and we get the full field (no randomness).
        '''
        self._index = []
        for fi, p in enumerate(self.files):
            for k in range(self.samples_per_day):
                self._index.append((fi, k))

    def __len__(self) -> int:
        return len(self._index)

    def _load_day_arrays(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Load the day (HR), build LR. Optionally use a cache function for speed.

        Returns:
          X_lr: [T_lr, H_lr, W_lr, C] standardized
          Y_hr: [T_hr, H_hr, W_hr, C] standardized
        '''
        # Optional cache hook (e.g., read/write .npz per day)
        if self.cache_fn is not None:
            cached = self.cache_fn(path)
            if cached is not None:
                X_lr, Y_hr = cached
                # assume cache stored standardized arrays already
                return X_lr, Y_hr

        ds = xr.open_dataset(path)
        X_lr, Y_hr = build_lr_from_hr(
            ds, variables=self.variables,
            temporal_factor=self.dom.temporal_factor,
            spatial_factor=self.dom.spatial_factor,
            temporal_method=self.temporal_method,
            spatial_sigma=self.spatial_sigma
        )
        ds.close()

        # Standardize using TRAIN stats (consistent across splits)
        X_lr = apply_standardize(X_lr, self.stats, self.variables)  # [Tlr,Hlr,Wlr,C]
        Y_hr = apply_standardize(Y_hr, self.stats, self.variables)  # [Thr,Hhr,Whr,C]

        return X_lr, Y_hr

    def _sample_patch_indices(self, H_hr: int, W_hr: int) -> Tuple[int, int]:
        '''Random HR top-left corner; ensure fits within HR dims.'''
        if self.mode == "test":
            '''
            deterministic: center crop if hr_patch is not too large, else top-left
            For center crop:
            2*y0 + hr_patch <= H_hr
            2*x0 + hr_patch <= W_hr
            (Visual presentation in one dimension:
            |<--y0--><--hr_patch--><--y0-->| <= |<------- H_hr ------->|)
            '''
            y0 = max(0, (H_hr - self.dom.hr_patch)//2)
            x0 = max(0, (W_hr - self.dom.hr_patch)//2)
            return y0, x0
        # random for train/val
        y0 = self.rng.integers(0, max(1, H_hr - self.dom.hr_patch + 1))
        x0 = self.rng.integers(0, max(1, W_hr - self.dom.hr_patch + 1))
        return int(y0), int(x0)

    def __getitem__(self, idx: int):
        file_idx, k = self._index[idx]
        path = self.files[file_idx]
        date_str = extract_yyyymmdd_from_name(path) or f"day{file_idx:05d}"

        X_lr, Y_hr = self._load_day_arrays(path)
        # shapes
        T_lr, H_lr, W_lr, C = X_lr.shape
        T_hr, H_hr, W_hr, C2 = Y_hr.shape
        assert C2 == C, "LR/HR channels mismatch"

        # Sample a random HR patch (same over all timesteps)
        y0, x0 = self._sample_patch_indices(H_hr, W_hr)
        y1, x1 = y0 + self.dom.hr_patch, x0 + self.dom.hr_patch
        patch_hr = Y_hr[:, y0:y1, x0:x1, :]            # [T_hr, ph, pw, C]

        # Compute corresponding LR coords (integer mapping with factor=6)
        fy = self.dom.spatial_factor
        fx = self.dom.spatial_factor
        ly0, lx0 = y0 // fy, x0 // fx
        ly1, lx1 = ly0 + self.dom.lr_patch, lx0 + self.dom.lr_patch
        patch_lr = X_lr[:, ly0:ly1, lx0:lx1, :]        # [T_lr, pl, pl, C]

        # Reorder to PyTorch-friendly [C,T,H,W]
        x = np.transpose(patch_lr, (3, 0, 1, 2)).astype(np.float32)
        y = np.transpose(patch_hr, (3, 0, 1, 2)).astype(np.float32)

        # Convert to tensors
        x_t = torch.from_numpy(x)   # [C, T_lr, H_lr_patch, W_lr_patch]
        y_t = torch.from_numpy(y)   # [C, T_hr, H_hr_patch, W_hr_patch]

        meta = {
            "date": date_str,
            "file": path,
            "y0x0": (int(y0), int(x0)),
            "patch_hr": (self.dom.hr_patch, self.dom.hr_patch),
            "patch_lr": (self.dom.lr_patch, self.dom.lr_patch),
        }

        return x_t, y_t, meta


# ------------------------------
# Optional caching helper
# ------------------------------

def make_npz_cache(cache_dir: Optional[str], variables: Sequence[str], stats: VarStats):
    '''
    Returns a function suitable for DownscalingDataset(cache_fn=...).

    The cache function:
      - If NPZ for the day exists, loads (X_lr, Y_hr) from it and returns.
      - Else returns None (caller will build on-the-fly).
    We DO NOT write here to keep the function read-only; if you want to write,
    you can extend it (see commented section).
    '''
    if cache_dir is None:
        return None

    os.makedirs(cache_dir, exist_ok=True)

    def _cache_fn(day_path: str) -> Optional[Tuple[np.ndarray,np.ndarray]]:
        day = extract_yyyymmdd_from_name(day_path) or os.path.splitext(os.path.basename(day_path))[0]
        npz_path = os.path.join(cache_dir, f"{day}.npz")
        if os.path.exists(npz_path):
            d = np.load(npz_path)
            X_lr = d["X_lr"]; Y_hr = d["Y_hr"]
            return X_lr, Y_hr
        # Example code to write cache (disabled by default):
        # ds = xr.open_dataset(day_path)
        # X_lr, Y_hr = build_lr_from_hr(ds, variables=variables, temporal_factor=6, spatial_factor=6)
        # ds.close()
        # X_lr = apply_standardize(X_lr, stats, variables)
        # Y_hr = apply_standardize(Y_hr, stats, variables)
        # np.savez_compressed(npz_path, X_lr=X_lr, Y_hr=Y_hr)
        # return X_lr, Y_hr
        return None

    return _cache_fn
