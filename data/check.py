from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Sequence, Callable

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import pdb


# ------------------------------
# Configuration / constants
# ------------------------------

DEFAULT_VARS = ("t2m", "d2m", "r2", "wspd", "sp")

@dataclass
class DomainConfig:
    """Spatial/temporal downscaling factors and patch sizes."""
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
    """Per-variable normalization stats computed on TRAIN only."""
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
    """
    Try to parse YYYYMMDD from filename (or any digits in the path).
    Accepts '20210301', '2021-03-01', '..._20210301_...' etc.
    Returns 'YYYYMMDD' string or None if not found.
    """
    m = _DATE_RE.search(os.path.basename(path))
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{y}{mo}{d}"


# ------------------------------
# Mask helper
# ------------------------------

def _to_float_with_nan(da: xr.DataArray) -> np.ndarray:
    """
    Convert DataArray (which may hold a masked array) to a float array with NaNs for invalids.
    """
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
    """Sort by ascending time steps to avoid surprises."""
    if "step" in ds.dims or "step" in ds.coords:
        return ds.sortby("step")
    raise ValueError("Dataset must have a 'step' dimension.")


def _subset_vars(ds: xr.Dataset, variables: Sequence[str]) -> xr.Dataset:
    """Keep only the variables of interest; raises if any are missing."""
    missing = [v for v in variables if v not in ds.data_vars]
    if missing:
        raise KeyError(f"Variables missing in dataset: {missing}")
    return ds[list(variables)]

def _stack_vars_thwc(ds: xr.Dataset, variables) -> np.ndarray:
    """
    Return complete dataset with shapes [T, H, W, C] in the form of a numpy array,
    with channel ordering same as that in the 'variables' list.
    valid_mask is True where values are valid (not NaN).
    """
    ds = ds.transpose("step", "latitude", "longitude")
    vals = []
    
    for v in variables:
        a = _to_float_with_nan(ds[v])     # [T,H,W] float32 with NaNs
        vals.append(a)
        
    vals_stack = np.stack(vals, axis=-1)           # [T,H,W,C]
    
    return vals_stack

# ------------------------------
def _gaussian_downsample2d(arr_thwc: np.ndarray, factor: int, sigma: float = 2.5) -> np.ndarray:
    """
    NaN-aware low-pass + decimate by 'factor' on the dims (H,W).
    Implementation: convolve data*weights and weights separately, divide.
    Expects arr[T, H, W, C]. Returns shape [T, H//factor, W//factor, C].
    """
    w = (~np.isnan(arr_thwc)).astype(np.float32)
    a = np.nan_to_num(arr_thwc, copy=False, nan=0.0).astype(np.float32,copy=False)

    num = gaussian_filter(a * w, sigma=(0, sigma, sigma, 0), mode="nearest")
    den = gaussian_filter(w,     sigma=(0, sigma, sigma, 0), mode="nearest")
       
    out = num / np.maximum(den, 1e-8)

    # Where den==0 (all NaN neighborhood), keep NaN
    out[den < 1e-8] = np.nan

    return out[:, ::factor, ::factor, :]

    
def _downsample_time(arr_thwc: np.ndarray, factor: int, method: str = "nearest") -> np.ndarray:
    """
    Temporal downsample with NaN handling from T -> T//factor.
    - 'nearest': pick frames 0, f, 2f, ... (preserves NaNs)
    - 'mean': block-average ignoring NaNs
    """
    T = arr_thwc.shape[0]
    T2 = T // factor
    if method == "nearest":
        idx = np.arange(0, T2 * factor, factor)
        return arr_thwc[idx]
    elif method == "mean":
        trimmed = arr_thwc[:T2 * factor]
        blk = trimmed.reshape(T2, factor, *trimmed.shape[1:])  # [T2, f, H, W, C]
        with np.errstate(invalid="ignore"):
            # nanmean over block axis=1
            return np.nanmean(blk, axis=1)
    else:
        raise ValueError("method must be 'nearest' or 'mean'.")
    


if __name__ == '__main__':
    temporal_factor = 6
    temporal_method = "nearest"
    variables = DEFAULT_VARS
    sigma = 1.5
    dir = r"F:\Work\cloud-nowcasting\datasets\GFS"
    files = sorted(os.listdir(dir))
    save_dir = r"F:\Work\cloud-nowcasting\Data_Exploration"
    
    for i,f in enumerate(files):
        
        print(i)
        file = os.path.join(dir, f)
        ds_hr = xr.open_dataset(file)
        ds_hr = _subset_vars(_ensure_time_sorted(ds_hr), DEFAULT_VARS)

        # for var in variables:
        #     g = ds_hr[var][:].values  # [T,H,W]
        #     if np.isnan(g).any() or np.isinf(g).any():
        #         print(f)
        #         pdb.set_trace()

        y_hr = _stack_vars_thwc(ds_hr, variables)
        
        x_tmp = _downsample_time(y_hr, factor=temporal_factor, method=temporal_method)  # [T_lr,H_hr,W_hr,C]
        y_tmp = _gaussian_downsample2d(x_tmp, factor=6, sigma=sigma)  # [T_lr,H_lr,W_lr,C]

        for j,var in enumerate(variables):
            g = y_tmp[:,:,:,j]
            if np.isnan(g).any() or np.isinf(g).any():
                print(f)
                print(var)
                pdb.set_trace()
        
        for j,var in enumerate(variables):
            g_hr = y_hr[:8,:,:,j]
            g_lr = y_tmp[:,:,:,j]

            # plt.figure(figsize=(12,6))
            # for k in range(8):
            #     plt.subplot(2,4,k+1)
            #     plt.imshow(g_hr[k], vmin=float(np.nanpercentile(g_hr, 1)), vmax=float(np.nanpercentile(g_hr, 99)), cmap='jet')
            #     plt.colorbar()
            #     plt.title(f"HR {var} t={k}")
            # plt.suptitle(f"File {i}: {f} HR")
            # plt.tight_layout()
            # plt.show()
            
            plt.figure(figsize=(12,6))
            for k in range(8):
                plt.subplot(2,4,k+1)
                plt.imshow(g_lr[k], vmin=float(np.nanpercentile(g_lr, 1)), vmax=float(np.nanpercentile(g_lr, 99)), cmap='jet')
                plt.colorbar()
                plt.title(f"LR {var} t={k}")
            plt.suptitle(f"File {i}: {f} LR")
            plt.tight_layout()

            if var == "t2m":
                plt.savefig(os.path.join(save_dir, f"Air_temp_LR_samples_sigma{sigma}.png"))
            elif var == "d2m":
                plt.savefig(os.path.join(save_dir, f"Dew_Pt_temp_LR_samples_sigma{sigma}.png"))  
            elif var == "r2":
                plt.savefig(os.path.join(save_dir, f"Rel_Hum_LR_samples_sigma{sigma}.png")) 
            elif var == "wspd":
                plt.savefig(os.path.join(save_dir, f"Surf_WSPD_LR_samples_sigma{sigma}.png"))
            elif var == "sp":
                plt.savefig(os.path.join(save_dir, f"Surface_Pressure_LR_samples_sigma{sigma}.png"))

            
            plt.close('all')

        pdb.set_trace()
            # plt.show()
    
