#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a Factorized Spatio-Temporal Super-Resolution model (TemporalUpsampler + EDSR-like SpatialSR)
with mask-aware losses and a temporal consistency regularizer.

Key features:
- Robust file discovery and split into train/val/test (by ratios or date cutoffs)
- Mask-aware Huber/L1 loss (only valid pixels contribute)
- Optional temporal consistency loss on Δt to reduce frame-to-frame "flutter"
- W&B logging via the correct Lightning import (WandbLogger)
- Mixed precision and gradient clipping for RTX 3060 stability
- Works with cache_fn from data/gfs_downscaling.py if you have precomputed NPZs

Dependencies:
    pip install pytorch-lightning wandb netcdf4 xarray numpy scipy torchmetrics
"""

import os
import re
import sched
import sys
import json
import glob
import math
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence

import numpy as np
import socket
import torch

# Accelerate training. Use "medium" if you see numerical issues
torch.set_float32_matmul_precision("high") # for Ampere+ GPUs; can be "medium" or "high"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import OptimizerLRScheduler

# use typed, callback-wired trainer builder
from utils.trainer_factory import build_trainer
import pdb


# --------- Project modules (make sure these paths exist) ----------
# Expecting these files in your repo:
#   data/gfs_downscaling.py
#   models/factorized_sr.py
from data.gfs_downscaling import (
    DownscalingDataset,
    compute_var_stats_train,
    DEFAULT_VARS,
    split_by_ratio,
    split_by_date,
    DomainConfig,
    make_npz_cache,
)

from models.factorized_sr import FactorizedSR


host = socket.gethostname()
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
# =============================
# File discovery and splitting
# =============================

def find_files(data_root: str, ext: str = ".nc") -> List[str]:
    """
    Recursively find all files with given extension under data_root.
    Returns a sorted list for reproducibility.
    """
    patt = os.path.join(data_root, "**", f"*{ext}")
    files = glob.glob(patt, recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    if len(files) == 0:
        raise FileNotFoundError(f"No '{ext}' files found under: {data_root}")
    return files


# =============================
# Loss functions (mask-aware)
# =============================
def sobel_gradients_2d(x: torch.Tensor) -> torch.Tensor:
    sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=x.device, dtype=x.dtype).view(1,1,3,3)
    C = x.size(1)
    kx, ky = sx.expand(C,1,3,3), sy.expand(C,1,3,3)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    
    # return torch.sqrt(gx*gx + gy*gy + 1e-12)
    mag2 = gx*gx + gy*gy
    mag2 = torch.clamp(mag2, max=1e12)  # optional cap
    return torch.sqrt(mag2 + 1e-12)


# PSD (radial)
def psd_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    PSD loss with radial averaging (batch+channel averaged).
    pred/target: [B, C, H, W]
    Returns scalar loss = MSE between normalized radial PSDs.
    """
    assert pred.shape == target.shape
    B, C, H, W = pred.shape

    def _psd(x):
        # # zero-mean to avoid DC dominance
        # x32 = (x - x.mean(dim=(-2, -1), keepdim=True)).float()   # zero-mean, cast to fp32
        # X = torch.fft.rfft2(x32, norm="ortho")
        # P = (X.real**2 + X.imag**2).contiguous().to(x.dtype)  # power
        # return P
    
        # Force FFT in fp32 to avoid cuFFT fp16 constraints & overflow quirks
        with torch.cuda.amp.autocast(enabled=False):
            x32 = (x - x.mean(dim=(-2, -1), keepdim=True)).float()      # zero-mean, fp32
            X = torch.fft.rfft2(x32, norm="ortho")                      # complex64
            P = (X.real**2 + X.imag**2)                                 # power spectrum (fp32)
            # sanitize to keep it finite and non-negative
            P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
            P = torch.clamp(P, min=0.0)
        return P

    Pp = _psd(pred)
    Pt = _psd(target)

    # Build radial bins
    yy = torch.fft.fftfreq(H, d=1.0).to(pred.device)  # [-0.5..0.5)
    xx = torch.fft.rfftfreq(W, d=1.0).to(pred.device) # [0..0.5]
    fy = yy.view(H, 1).expand(H, xx.numel())
    fx = xx.view(1, xx.numel()).expand(H, xx.numel())
    fr = torch.sqrt(fx*fx + fy*fy)  # [H, W//2+1]
    # Bin edges in frequency radius
    nbins = min(64, H//2)
    bin_edges = torch.linspace(0.0, fr.max()+1e-6, nbins+1, device=pred.device)
    # Digitize
    # shape [H, W//2+1] -> bin index 0..nbins-1
    inds = torch.bucketize(fr, bin_edges) - 1
    inds = inds.clamp(0, nbins-1)

    # Accumulate per bin
    # reshape to [B*C, H, W//2+1]
    BC = B * C
    Pp_flat = Pp.reshape(BC, H, -1)
    Pt_flat = Pt.reshape(BC, H, -1)
    inds_flat = inds.reshape(-1)

    # For each bin, average power
    psd_p = []
    psd_t = []
    for k in range(nbins):
        mask = (inds_flat == k).reshape(1, H, -1)  # broadcast over batch*channel later
        # mask sum size
        denom = mask.sum().clamp_min(1)
        pp_k = (Pp_flat * mask).sum(dim=(1,2)) / denom  # [BC]
        pt_k = (Pt_flat * mask).sum(dim=(1,2)) / denom  # [BC]
        psd_p.append(pp_k)
        psd_t.append(pt_k)
    psd_p = torch.stack(psd_p, dim=-1)  # [BC, nbins]
    psd_t = torch.stack(psd_t, dim=-1)  # [BC, nbins]

    # Normalize each PSD curve to unit area to focus on shape/scale distribution
    psd_p = psd_p / (psd_p.sum(dim=-1, keepdim=True) + eps)
    psd_t = psd_t / (psd_t.sum(dim=-1, keepdim=True) + eps)

    return F.mse_loss(psd_p, psd_t)


def masked_l1(pred, target, mask=None, eps: float = 1e-6):
    """
    Mean L1 over valid pixels.
    pred/target: [B, C, T, H, W]
    mask:        [B, C, T, H, W] bool (True=valid). If None -> all valid.
    """
    if mask is None:
        return F.l1_loss(pred, target)
    w = mask.float()
    return (torch.abs(pred - target) * w).sum() / (w.sum() + eps)


def masked_huber(pred, target, mask=None, delta: float = 1.0, eps: float = 1e-6):
    """
    Masked SmoothL1 / Huber loss.
    """
    if mask is None:
        return F.smooth_l1_loss(pred, target, beta=delta)
    w = mask.float()
    diff = pred - target
    abs_diff = torch.abs(diff)
    # SmoothL1 (Huber): piecewise quadratic/linear
    quad = torch.minimum(abs_diff, torch.tensor(delta, device=abs_diff.device))
    lin  = abs_diff - quad
    loss = 0.5 * (quad ** 2) / delta + lin
    return (loss * w).sum() / (w.sum() + eps)


def temporal_consistency_loss(y_hat, y, mask=None, eps: float = 1e-6):
    """
    L1 on temporal differences,time-delta, between prediction and target.
    Inputs: [B, C, T, H, W]; compares frames t and t+1 (so T-1 diffs).
    If mask is provided, intersect masks at t and t+1.
    """
    d_pred = y_hat[:, :, 1:] - y_hat[:, :, :-1]   # [B,C,T-1,H,W]
    d_true = y[:,   :, 1:] - y[:,   :, :-1]       # [B,C,T-1,H,W]
    if mask is not None:
        m = mask.float()
        m_pair = (m[:, :, 1:] * m[:, :, :-1]).bool()
    else:
        m_pair = None
    return masked_l1(d_pred, d_true, m_pair, eps=eps)

# =============================
# LightningModule
# =============================

class LitFactorizedSR(pl.LightningModule):
    """
    Lightning wrapper for FactorizedSR model with:
      - main pixel loss (Huber/L1)
      - optional temporal consistency loss on time-delta
      - mask-aware training
    Logs:
      - train/loss
      - val/monitor (used for checkpointing)
      - val/loss, val/rmse, val/ssim (optional, if torchmetrics available)
    """
    def __init__(
        self,
        stats,                       # VarStats from compute_var_stats_train
        in_channels: int,
        lr: float = 2e-4,
        min_lr: float = 1e-5,
        lambda_dt: float = 0.05,
        use_huber: bool = True,
        huber_delta: float = 1.0,
        lambda_grad: float = 0.03,
        lambda_psd: float  = 0.015,
        weight_decay: float = 1e-4,

        temporal_factor: int = 6,
        spatial_factor: int = 6,
        # for logging only:
        monitor_metric: str = "val/monitor",
        lr_schedule: str = "epoch-cosine",     # "epoch-cosine" | "warmup-cosine"
        warmup_steps: int = 800,               # used only for warmup-cosine
        total_steps: Optional[int] = None,     # optional override for warmup-cosine

    ):
        super().__init__()
        self.save_hyperparameters(ignore=["stats"])
        self.model = FactorizedSR(in_channels=in_channels)  # EDSR-like spatial + temporal module
        self.stats = stats
        self.lr = lr
        self.min_lr = min_lr
        self.lambda_dt = lambda_dt
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.lambda_dt = lambda_dt
        self.lambda_grad = lambda_grad
        self.lambda_psd = lambda_psd
        self.weight_decay = weight_decay
        self.monitor_metric = monitor_metric
        # NEW: keep the scheduler knobs
        self.lr_schedule = lr_schedule
        self.warmup_steps = int(warmup_steps)
        self.total_steps: Optional[int] = total_steps

        # Optional metrics (installed via torchmetrics)
        try:
            from torchmetrics import MeanSquaredError, StructuralSimilarityIndexMeasure
            self.metric_rmse = MeanSquaredError(squared=False)
            # data_range ~ after standardization; 4 standard deviations is typical safe band
            self.metric_ssim = StructuralSimilarityIndexMeasure(data_range=4.0)
        except Exception:
            self.metric_rmse = None
            self.metric_ssim = None

    def forward(self, x):
        return self.model(x)

    def _pixel_loss(self, y_hat, y, mask):
        if self.use_huber:
            return masked_huber(y_hat, y, mask, delta=self.huber_delta)
        return masked_l1(y_hat, y, mask)

    def _compute_losses(self, y_hat, y, mask=None):

        B, C, T, H, W = y.shape
        yhat_2d = y_hat.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        y_2d    = y.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        Gyhat = sobel_gradients_2d(yhat_2d)
        Gy    = sobel_gradients_2d(y_2d)

        L_grad = F.l1_loss(Gyhat, Gy)
        L_pix = self._pixel_loss(y_hat, y, mask)
        L_psd = psd_loss(yhat_2d, y_2d)
        L_dt  = temporal_consistency_loss(y_hat, y, mask)
        # total = L_pix + self.lambda_dt * L_dt
        total = L_pix + self.lambda_dt * L_dt + self.lambda_grad * L_grad + self.lambda_psd * L_psd
        return total, {"loss/pixel": L_pix, "loss/dt": L_dt, "loss/grad": L_grad, "loss/psd": L_psd}

    def training_step(self, batch, batch_idx):
        # Batch may be (x, y, meta) or (x, y, meta, mx, my) if mask-aware dataset is used
        if len(batch) == 3:
            x, y, _ = batch
            mask = None
        else:
            x, y, _, mx, my = batch
            mask = my  
            
        y_hat = self(x)

        # (Optional) ensure invalid targets don't leak into gradients
        if mask is not None:
            y_hat = torch.where(mask, y_hat, torch.zeros_like(y_hat))
            y     = torch.where(mask, y,     torch.zeros_like(y))

        total, parts = self._compute_losses(y_hat, y, mask)
        self.log("train/loss", total, prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict({f"train/{k}": v for k, v in parts.items()}, prog_bar=False)
        return total

    def validation_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, _ = batch
            mask = None
        else:
            x, y, _, mx, my = batch
            mask = my

        y_hat = self(x)

        if mask is not None:
            y_hat = torch.where(mask, y_hat, torch.zeros_like(y_hat))
            y     = torch.where(mask, y,     torch.zeros_like(y))

        total, parts = self._compute_losses(y_hat, y, mask)

        # Primary monitor metric for callbacks/checkpoints
        self.log("val/monitor", total, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/loss", total, prog_bar=False, on_step=False, on_epoch=True)

        # Optional metrics if torchmetrics available
        if self.metric_rmse is not None:
            rmse = self.metric_rmse(y_hat.contiguous(), y.contiguous())
            self.log("val/rmse", rmse, prog_bar=False, on_epoch=True)
        if self.metric_ssim is not None:
            # SSIM expects [N, C, H, W]; we can flatten time into batch
            B, C, T, H, W = y.shape
            y_ = y.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
            yh_ = y_hat.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
            ssim = self.metric_ssim(yh_, y_)
            self.log("val/ssim", ssim, prog_bar=False, on_epoch=True)

        return total

    def configure_optimizers(self) -> OptimizerLRScheduler:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.99))

        if self.lr_schedule == "epoch-cosine":
            # Exactly like your old behavior: cosine per EPOCH, simple & robust
            tmax = int(self.trainer.max_epochs or 1)
            sched = CosineAnnealingLR(opt, T_max=tmax, eta_min=0.0)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

        # warmup-cosine (per STEP): short LinearLR warmup then CosineAnnealingLR
        # resolve total_steps
        total_steps = self.total_steps
        if total_steps is None:
            est = getattr(self.trainer, "estimated_stepping_batches", None)
            total_steps = int(est) if est is not None else (int(self.trainer.max_epochs or 1) * 1000)

        warm = max(int(self.warmup_steps), 0)
        cosine_steps = max(int(total_steps) - warm, 1)

        if warm > 0:
            warmup = LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warm)
            cosine = CosineAnnealingLR(opt, T_max=cosine_steps, eta_min=0.0)
            sched = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warm])
        else:
            sched = CosineAnnealingLR(opt, T_max=int(total_steps), eta_min=0.0)

        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}

    

# =============================
# DataModule-like build helpers
# =============================

def build_dataloaders(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    stats,
    variables: List[str],
    dom: DomainConfig,
    batch_size: int,
    num_workers: int,
    cache_dir: Optional[str] = None,
    temporal_method: str = "nearest",
    spatial_sigma: float = 2.5,
    return_masks: bool = False,
):
    """
    Build train/val/test DataLoaders.
    """
    cache_fn = make_npz_cache(cache_dir, variables, stats) if cache_dir else None

    train_ds = DownscalingDataset(
        train_files, stats, variables=variables, mode="train",
        dom=dom, cache_fn=cache_fn,
        temporal_method=temporal_method, spatial_sigma=spatial_sigma
    )
    val_ds = DownscalingDataset(
        val_files, stats, variables=variables, mode="val",
        dom=dom, cache_fn=cache_fn,
        temporal_method=temporal_method, spatial_sigma=spatial_sigma
    )
    test_ds = DownscalingDataset(
        test_files, stats, variables=variables, mode="test",
        dom=dom, cache_fn=cache_fn,
        temporal_method=temporal_method, spatial_sigma=spatial_sigma
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,         shuffle=False, num_workers=max(1, num_workers//2), pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,         shuffle=False, num_workers=max(1, num_workers//2), pin_memory=True)

    return train_loader, val_loader, test_loader


# =============================
# Main
# =============================

def parse_args():
    p = argparse.ArgumentParser(description="Train Factorized SR model (temporal + spatial) for GFS downscaling.")
    # Data
    p.add_argument("--data_root", type=str, required=True, help="Root folder containing daily NetCDF files.")
    p.add_argument("--ext", type=str, default=".nc", help="File extension to search for.")
    p.add_argument("--cache_dir", type=str, default=None, help="Optional NPZ cache directory (read-through).")
    p.add_argument("--variables", type=str, nargs="+", default=DEFAULT_VARS, help="Variables to use, e.g., t2m d2m r2 wspd sp")

    # Splits
    p.add_argument("--split", type=str, choices=["ratio", "date"], default="date", help="Split strategy.")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--train_start_date", type=str, default='20210301', help='For split="date": e.g., 2023-12-31')
    p.add_argument("--train_end_date", type=str, default='20240826', help='For split="date": e.g., 2023-12-31')
    p.add_argument("--val_start_date", type=str, default='20240827', help='For split="date": e.g., 2024-06-30')
    p.add_argument("--val_end_date", type=str, default='20250201', help='For split="date": e.g., 2024-06-30')
    p.add_argument("--test_start_date", type=str, default='20250202', help='For split="date": e.g., 2024-07-01')
    p.add_argument("--test_end_date", type=str, default='20250723', help='For split="date": e.g., 2024-07-01')
    p.add_argument("--seed", type=int, default=42)

    # Model / Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--lambda_dt", type=float, default=0.05, help="Weight for temporal time-delta loss.")
    p.add_argument("--use_huber", action="store_true", help="Use Huber (SmoothL1) instead of L1.")
    p.add_argument("--huber_delta", type=float, default=1.0)
    p.add_argument("--lambda_grad", type=float, default=0.03, help="Sobel gradient loss.")
    p.add_argument("--lambda_psd", type=float, default=0.0, help="PSD (spectral) loss.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=16)

    # --- Sharpness/PSD preset & knobs ---
    p.add_argument("--preset", type=str, default='sharp_psd',
                   choices=[None, "sharp_psd"], help="Quick config for sharper but stable training.")

    # Warm startup choices
    p.add_argument("--lr_schedule", type=str, choices=["epoch-cosine", "warmup-cosine"],
               default="epoch-cosine",help="Use classic epoch cosine or per-step warmup+cosine.")
    p.add_argument("--warmup_steps", type=int, default=800, help="Only used if --lr_schedule=warmup-cosine.")

    p.add_argument("--resume_wandb_id", type=str, default=None,
               help="Append logs to an existing W&B run (the short run id from the URL).")

    # Resume from last checkpoint
    p.add_argument("--resume_ckpt", type=str, default=None,
               help="Optional checkpoint path to resume training from (e.g., outputs/factorized/last.ckpt)")


    # Domain / Preproc
    p.add_argument("--temporal_factor", type=int, default=6)
    p.add_argument("--spatial_factor", type=int, default=6)
    p.add_argument("--hr_patch", type=int, default=96)
    p.add_argument("--temporal_method", type=str, choices=["nearest", "mean"], default="nearest")
    p.add_argument("--spatial_sigma", type=float, default=3.0)

    # Logging / Output
    p.add_argument("--project", type=str, default="gfs-new")
    p.add_argument("--run_name", type=str, default="factorized-new-losses")
    p.add_argument("--out_dir", type=str, default="outputs/factorized")
    p.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    return p.parse_args()



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    if args.preset == "sharp_psd":
        args.lr = 8e-4
        args.warmup_steps = 1500
        args.min_lr = 1e-5
        args.weight_decay = 1e-4
        args.grad_clip_norm = 1.0
        # Keep your current lambda_dt unless you saw flicker:
        args.lambda_dt = 0.05
        args.lambda_grad = 0.03
        args.lambda_psd  = 0.0

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Discover files
    all_files = find_files(args.data_root, ext=args.ext)

    # 2) Split
    if args.split == "ratio":
        train_files, val_files, test_files = split_by_ratio(
            all_files, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
        )
    else:
        if (not args.train_start_date or not args.val_start_date or 
            not args.test_start_date or not args.train_end_date or 
            not args.val_end_date or not args.test_end_date):
            raise ValueError('For split="date", please provide '
            '--train_start_date, --train_end_date, --val_start_date, '
            '--val_end_date, --test_start_date and --test_end_date.')
        train_files, val_files, test_files = split_by_date(
            all_files, args.train_start_date, args.train_end_date, 
            args.val_start_date, args.val_end_date,
            args.test_start_date, args.test_end_date
        )
    
    print(f"[Split] Train: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}")

    # 3) Compute normalization stats on TRAIN only (mask/NaN-aware in your gfs_downscaling)
    stats = compute_var_stats_train(train_files, variables=args.variables)
    with open(os.path.join(args.out_dir, "var_stats.json"), "w") as f:
        json.dump({"mean": stats.mean, "std": stats.std}, f, indent=2)

    # 4) Domain config (ensure hr_patch divisible by spatial_factor)
    if args.hr_patch % args.spatial_factor != 0:
        raise ValueError(f"hr_patch ({args.hr_patch}) must be divisible by spatial_factor ({args.spatial_factor}).")
    dom = DomainConfig(
        temporal_factor=args.temporal_factor,
        spatial_factor=args.spatial_factor,
        hr_patch=args.hr_patch,
    )

    # 5) Build data loaders
    train_loader, val_loader, test_loader = build_dataloaders(
        train_files=train_files,
        val_files=val_files,
        test_files=test_files,
        stats=stats,
        variables=args.variables,
        dom=dom,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
        temporal_method=args.temporal_method,
        spatial_sigma=args.spatial_sigma,
        return_masks=True,
    )

    # compute total_steps if you're using warmup-cosine
    steps_per_epoch = len(train_loader)
    total_steps = None
    if args.lr_schedule == "warmup-cosine":
        total_steps = steps_per_epoch * args.max_epochs

    # 6) Build Lightning model  
    in_channels = len(args.variables)
    
    # weight-only load (fresh optimizer/scheduler)
    if args.resume_ckpt:
        lit_model = LitFactorizedSR.load_from_checkpoint(
            args.resume_ckpt,
            stats=stats,
            in_channels=in_channels,
            lr=args.lr,
            min_lr=args.min_lr,
            lambda_dt=args.lambda_dt,
            use_huber=args.use_huber,
            huber_delta=args.huber_delta,
            lambda_grad=args.lambda_grad,
            lambda_psd=args.lambda_psd,
            weight_decay=args.weight_decay,
            temporal_factor=args.temporal_factor,
            spatial_factor=args.spatial_factor,
            monitor_metric="val/monitor",
            lr_schedule=args.lr_schedule,      # <- keep your chosen schedule
            warmup_steps=args.warmup_steps,    # <- warmup you want
            total_steps=total_steps,           # <- step horizon for cosine
        )
    else:
        lit_model = LitFactorizedSR(
            stats=stats,
            in_channels=in_channels,
            lr=args.lr,
            min_lr=args.min_lr,
            lambda_dt=args.lambda_dt,
            use_huber=args.use_huber,
            huber_delta=args.huber_delta,
            lambda_grad=args.lambda_grad,
            lambda_psd=args.lambda_psd,
            weight_decay=args.weight_decay,
            temporal_factor=args.temporal_factor,
            spatial_factor=args.spatial_factor,
            monitor_metric="val/monitor",
            lr_schedule=args.lr_schedule,          
            warmup_steps=args.warmup_steps,
            total_steps=total_steps
        )

    # If the user passed a run id, tell W&B to resume that exact run
    if args.resume_wandb_id:
        os.environ["WANDB_RESUME"] = "must"          # force attach to the same run
        os.environ["WANDB_RUN_ID"] = args.resume_wandb_id

    # 7) Build a typed, callback-rich Trainer via factory
    trainer, ckpt_cb, th_cb, wandb_logger = build_trainer(
        out_dir=args.out_dir,
        max_epochs=args.max_epochs,
        precision=args.precision,          # factory normalizes: "16-mixed", "32-true", etc.
        grad_clip=args.grad_clip,
        accumulate_grad_batches=1,
        log_every_n_steps=25,
        throughput_every_n_steps=50,       # logs perf/throughput every 50 steps
        project=args.project,
        run_name=f"{args.run_name}-{host}",   # Append hostname for clear machine-wise tracking at W&B website
        wandb_mode=args.wandb_mode,        # "online" | "offline" | "disabled"
        failfast_patience=2,               # early fail if val metric explodes/NaNs or stalls
        accelerator="gpu",
        devices=1, 
    )

    # Persist the run id so you can reuse it later
    try:
        run_id = wandb_logger.experiment.id
        print(f"[W&B] run_id: {run_id}")
        with open(os.path.join(args.out_dir, "wandb_run_id.txt"), "w") as f:
            f.write(run_id + "\n")
    except Exception:
        pass

    # Optionally resume training from a saved checkpoint
    ckpt_path = args.resume_ckpt if args.resume_ckpt else None


    # 9) Train
    same_schedule = (args.lr_schedule == "epoch-cosine")   # adjust if needed to detect match
    if args.resume_ckpt and same_schedule:
        trainer.fit(lit_model, train_loader, val_loader, ckpt_path=args.resume_ckpt)
    else:
        # weights-only path shown above
        trainer.fit(lit_model, train_loader, val_loader)
    # trainer.fit(lit_model, train_loader, val_loader)

    from utils.estimate_runtime import estimate_runtime
    if th_cb.last_tps is not None:
        est = estimate_runtime(train_loader, observed_tps=th_cb.last_tps)
        print(f"[Observed] ~{est['secs_per_epoch']/60:.1f} min/epoch at {th_cb.last_tps:.0f} samp/s")


    # 10) (Optional) Test on test set using best ckpt
    if ckpt_cb.best_model_path and os.path.exists(ckpt_cb.best_model_path):
        print(f"[Eval] Loading best checkpoint: {ckpt_cb.best_model_path}")
        best = LitFactorizedSR.load_from_checkpoint(
            ckpt_cb.best_model_path,
            stats=stats,
            in_channels=in_channels,
            lr=args.lr,
            lambda_dt=args.lambda_dt,
            use_huber=args.use_huber,
            huber_delta=args.huber_delta,
            weight_decay=args.weight_decay,
            temporal_factor=args.temporal_factor,
            spatial_factor=args.spatial_factor,
            lr_schedule=args.lr_schedule,
            warmup_steps=args.warmup_steps,
            total_steps=total_steps
        )
        trainer.test(best, dataloaders=test_loader)
    else:
        print("[Eval] No best checkpoint found; skipping test.")

    print("[Done]")

if __name__ == "__main__":
    main()  

