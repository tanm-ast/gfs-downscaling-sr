#!/usr/bin/env python3
"""
Full-Field Super-Resolution Test & Baseline Evaluation (GFS Downscaling)

Adds --scales {standard,destandard,both} to compute metrics and save figures on:
- standardized tensors (z-scores)
- de-standardized tensors (physical units)
- or both

Images are saved with suffixes: *_std.png or *_destd.png
CSV contains separate rows per scale, e.g., model_RMSE_std, model_RMSE_destd.
"""

import argparse
import os
import math
import csv
from typing import List, Tuple, Dict, Optional, Callable, Type, cast

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from typing import Optional, Callable
ssim_skimage: Optional[Callable] = None
try:
        ssim_skimage = _ssim
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

CubicSpline: Optional[Callable] = None
try:
    from scipy.interpolate import CubicSpline as _CubicSpline
    CubicSpline = _CubicSpline
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from data.gfs_downscaling import (
    DEFAULT_VARS,
    VarStats,
    DomainConfig,
    build_lr_from_hr,
    apply_standardize,
    compute_var_stats_train,
    split_by_date,
    split_by_ratio,
    extract_yyyymmdd_from_name,
)

from models.factorized_sr import FactorizedSR
from train_factorized import find_files



# ---- Pure-PyTorch SSIM (no skimage) ----
import torch
import torch.nn.functional as F

def _gaussian_kernel2d(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d

def ssim_torch(pred: torch.Tensor,
               target: torch.Tensor,
               data_range: float = 1.0,
               win_size: int = 11,
               sigma: float = 1.5,
               reduction: str = "mean") -> float:
    """
    SSIM for images in PyTorch.
    pred, target: [N, C, H, W] tensors on same device/dtype.
    Returns float if reduction='mean'.
    """
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
    N, C, H, W = pred.shape
    device, dtype = pred.device, pred.dtype

    k = _gaussian_kernel2d(win_size, sigma, device, dtype)
    k = k.view(1, 1, win_size, win_size).repeat(C, 1, 1, 1)  # [C,1,K,K]
    pad = win_size // 2

    mu_x = F.conv2d(pred, k, padding=pad, groups=C)
    mu_y = F.conv2d(target, k, padding=pad, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(pred * pred, k, padding=pad, groups=C) - mu_x2
    sigma_y = F.conv2d(target * target, k, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(pred * target, k, padding=pad, groups=C) - mu_xy

    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    ssim_map = ((2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2))

    if reduction == "none":
        return ssim_map  # type: ignore[return-value]

    return float(ssim_map.mean().item())
# ---------------- Metrics ----------------
def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).item())

def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).item())

def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(max(data_range, 1e-6)) - 10.0 * math.log10(mse)

def ssim_img(pred_hw: np.ndarray, target_hw: np.ndarray, data_range: float) -> float:
    if _HAS_SKIMAGE and ssim_skimage is not None:
        return float(ssim_skimage(target_hw, pred_hw, data_range=float(max(data_range, 1e-6)),
                                  gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
    p = torch.from_numpy(pred_hw).unsqueeze(0).unsqueeze(0)
    t = torch.from_numpy(target_hw).unsqueeze(0).unsqueeze(0)
    mu_x = p.mean(); mu_y = t.mean()
    sig_x = p.var(unbiased=False); sig_y = t.var(unbiased=False)
    sig_xy = ((p - mu_x) * (t - mu_y)).mean()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_val = ((2*mu_x*mu_y + c1) * (2*sig_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sig_x + sig_y + c2))
    return float(ssim_val.item())

def dynamic_data_range(t: torch.Tensor, mode: str = "percentile") -> float:
    """Estimate an appropriate data_range for PSNR/SSIM on physical-scaled data."""
    if mode == "percentile":
        arr = cast(np.ndarray, to_numpy(t))
        lo = np.percentile(arr, 1.0)
        hi = np.percentile(arr, 99.0)
        return float(max(hi - lo, 1e-6))
    return float((t.max() - t.min()).item())


# ------------- (De-)Standardization -------------
def destandardize_tensor(x: torch.Tensor, stats: VarStats, var_names: List[str]) -> torch.Tensor:
    """
    x: [B, C, T, H, W] standardized -> de-standardized (physical units)
    """
    y = x.clone()
    for ci, v in enumerate(var_names):
        m = float(stats.mean[v]); s = float(stats.std[v])
        y[:, ci] = y[:, ci] * (s + 1e-6) + m
    return y


# ------------- Temporal / Spatial baselines -------------
def temporal_interp(x: torch.Tensor, factor: int, mode: str = "linear") -> torch.Tensor:
    if factor <= 1 or x.size(2) == 1:
        return x
    if mode == "cubic" and _HAS_SCIPY and CubicSpline is not None:
        B, C, T, H, W = x.shape
        t_in = np.arange(T, dtype=np.float32)
        t_out = np.linspace(0, T - 1, T * factor, dtype=np.float32)
        x_np = to_numpy(x)
        y_np = np.empty((B, C, T * factor, H, W), dtype=np.float32)
        for b in range(B):
            for c in range(C):
                series = x_np[b, c].reshape(T, -1)
                cs = CubicSpline(t_in, series, axis=0, bc_type="natural")
                yi = cs(t_out)
                y_np[b, c] = yi.reshape(T * factor, H, W)
        return torch.from_numpy(y_np).to(x.device)
    return F.interpolate(x, size=(x.size(2)*factor, x.size(3), x.size(4)), mode="trilinear", align_corners=False)

def spatial_bicubic(x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    if x.dim() == 5:
        B, C, T, H, W = x.shape
        y = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        y2 = F.interpolate(y, size=out_hw, mode="bicubic", align_corners=False)
        return y2.reshape(B, T, C, out_hw[0], out_hw[1]).permute(0,2,1,3,4)
    else:
        return F.interpolate(x, size=out_hw, mode="bicubic", align_corners=False)


# ------------- Plotting (full maps) -------------
def plot_full_maps(out_dir: str, day_str: str, var_name: str, hour_idx: int,
                   hr: torch.Tensor, pred: torch.Tensor,
                   bic: Optional[torch.Tensor] = None,
                   temp: Optional[torch.Tensor] = None,
                   stc: Optional[torch.Tensor] = None,
                   vmin: Optional[float] = None, vmax: Optional[float] = None,
                   suffix: str = "std"):
    """
    hr/pred/baselines: [B=1, C, T, H, W] (already in chosen scale: std or destd)
    """
    os.makedirs(out_dir, exist_ok=True)
    B, C, T, H, W = hr.shape
    assert B == 1

    def _get(img: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if img is None: return None
        return to_numpy(img[0, :, hour_idx])  # [C,H,W]

    hr_c = _get(hr); md_c = _get(pred); bi_c = _get(bic); tp_c = _get(temp); sc_c = _get(stc)
    if hr_c is None:
        return
    hr_img = hr_c[0] if hr_c.ndim == 3 else hr_c
    hr_img = np.asarray(hr_img)
    md_img = md_c[0] if (md_c is not None and getattr(md_c, 'ndim', 0) == 3) else (md_c if md_c is not None else None)
    bi_img = bi_c[0] if (bi_c is not None and getattr(bi_c, 'ndim', 0) == 3) else (bi_c if bi_c is not None else None)
    tp_img = tp_c[0] if (tp_c is not None and getattr(tp_c, 'ndim', 0) == 3) else (tp_c if tp_c is not None else None)
    sc_img = sc_c[0] if (sc_c is not None and getattr(sc_c, 'ndim', 0) == 3) else (sc_c if sc_c is not None else None)

    vmin = vmin if vmin is not None else float(np.percentile(np.asarray(hr_img), 2))
    vmax = vmax if vmax is not None else float(np.percentile(np.asarray(hr_img), 98))

    panels = [("HR", hr_img), ("Model", md_img), ("Bicubic", bi_img), ("TempInterp", tp_img), ("ST-Combined", sc_img)]
    panels = [(t, im) for (t, im) in panels if im is not None]

    fig, axs = plt.subplots(1, len(panels), figsize=(5*len(panels), 4), constrained_layout=True)
    if len(panels) == 1:
        axs = [axs]
    for ax, (title, im) in zip(axs, panels):
        ax.imshow(im, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{day_str}  var={var_name}  hour={hour_idx}  [{suffix}]")
    out_path = os.path.join(out_dir, f"{day_str}_{var_name}_h{hour_idx:02d}_{suffix}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ext", type=str, default=".nc")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--stats_json", type=str, default=None)
    ap.add_argument("--variables", type=str, nargs="+", default=list(DEFAULT_VARS))
    ap.add_argument("--split", type=str, choices=["ratio","date"], default="date")
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--train_start_date", type=str, default="20210301")
    ap.add_argument("--train_end_date", type=str, default="20240826")
    ap.add_argument("--val_start_date", type=str, default="20240827")
    ap.add_argument("--val_end_date", type=str, default="20250201")
    ap.add_argument("--test_start_date", type=str, default="20250202")
    ap.add_argument("--test_end_date", type=str, default="20250723")

    ap.add_argument("--temporal_factor", type=int, default=6)
    ap.add_argument("--spatial_factor", type=int, default=6)
    ap.add_argument("--temporal_mode", type=str, choices=["linear","cubic"], default="cubic")

    ap.add_argument("--batch_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=str, default="eval_out")
    ap.add_argument("--results_csv", type=str, default="metrics_test.csv")
    ap.add_argument("--num_examples", type=int, default=4)
    ap.add_argument("--example_var", type=str, default=None)
    ap.add_argument("--example_hours", type=int, nargs="+", default=[0, 6, 12])

    ap.add_argument("--scales", type=str, choices=["standard","destandard","both"], default="both",
                    help="Compute metrics/save images on standardized data, destandardized data, or both.")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Discover files
    all_files = find_files(args.data_root, ext=args.ext)
    if args.split == "ratio":
        train_files, val_files, test_files = split_by_ratio(all_files, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=42)
    else:
        train_files, val_files, test_files = split_by_date(
            all_files,
            args.train_start_date, args.train_end_date,
            args.val_start_date, args.val_end_date,
            args.test_start_date, args.test_end_date
        )
    if len(test_files) == 0:
        raise RuntimeError("No TEST files matched split.")

    # Stats
    if args.stats_json and os.path.exists(args.stats_json):
        import json
        with open(args.stats_json, "r") as f:
            d = json.load(f)
        stats = VarStats(mean=d["mean"], std=d["std"])
    else:
        print("[Info] stats_json not provided; recomputing TRAIN stats.")
        stats = compute_var_stats_train(train_files, variables=args.variables)

    # Load model
    in_channels = len(args.variables)
    device = torch.device(args.batch_device)
    model = FactorizedSR(in_channels=in_channels).to(device)
    loaded = False
    try:
        from train_factorized import LitFactorizedSR
        lit = LitFactorizedSR.load_from_checkpoint(
            args.ckpt,
            stats=stats,
            in_channels=in_channels,
            lr=1e-4, lambda_dt=0.05, use_huber=True, huber_delta=1.0, weight_decay=1e-4,
            temporal_factor=args.temporal_factor, spatial_factor=args.spatial_factor,
            lr_schedule="epoch-cosine", warmup_steps=0, total_steps=None
        )
        model.load_state_dict(lit.model.state_dict(), strict=False)
        loaded = True
    except Exception as e:
        print(f"[Warn] Lightning checkpoint load failed: {e}. Trying plain state_dict...")
        try:
            sd = torch.load(args.ckpt, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
                new_sd = {}
                for k, v in sd.items():
                    if k.startswith("model."):
                        new_sd[k[len("model."):]] = v
                    else:
                        new_sd[k] = v
                sd = new_sd
            model.load_state_dict(sd, strict=False)
            loaded = True
        except Exception as e2:
            print(f"[Error] state_dict load failed: {e2}")
    if not loaded:
        raise RuntimeError("Failed to load model weights.")

    model.eval()

    # Aggregators per scale
    def empty_metrics():
        return {
            "model_RMSE":0.0,"model_MAE":0.0,"model_PSNR":0.0,"model_SSIM":0.0,
            "bicubic_RMSE":0.0,"bicubic_MAE":0.0,"bicubic_PSNR":0.0,"bicubic_SSIM":0.0,
            "tempinterp_RMSE":0.0,"tempinterp_MAE":0.0,"tempinterp_PSNR":0.0,"tempinterp_SSIM":0.0,
            "st_combined_RMSE":0.0,"st_combined_MAE":0.0,"st_combined_PSNR":0.0,"st_combined_SSIM":0.0,
        }

    metrics_sum_std = empty_metrics() if args.scales in ("standard","both") else None
    metrics_sum_dst = empty_metrics() if args.scales in ("destandard","both") else None

    var_names = list(args.variables)
    plot_var = args.example_var if args.example_var else var_names[0]
    if plot_var not in var_names:
        raise ValueError(f"--example_var '{plot_var}' not in variables {var_names}")
    ch_idx_plot = var_names.index(plot_var)

    n_days = 0
    for di, p in enumerate(test_files):
        import xarray as xr
        ds = xr.open_dataset(p)
        X_lr, Y_hr = build_lr_from_hr(
            ds, variables=var_names,
            temporal_factor=args.temporal_factor,
            spatial_factor=args.spatial_factor,
            temporal_method="nearest",
            spatial_sigma=3.0,
            nan_safe_spatial=False
        )
        ds.close()

        X_lr_std = apply_standardize(X_lr, stats, variables=var_names)
        Y_hr_std = apply_standardize(Y_hr, stats, variables=var_names)

        x = torch.from_numpy(np.transpose(X_lr_std, (3,0,1,2))).unsqueeze(0).float().to(device)
        y = torch.from_numpy(np.transpose(Y_hr_std, (3,0,1,2))).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred = model(x)

        # Baselines on standardized LR
        bic = spatial_bicubic(x, (y.shape[-2], y.shape[-1]))
        temp = temporal_interp(x, factor=args.temporal_factor, mode=args.temporal_mode)
        if (temp.shape[-2], temp.shape[-1]) != (y.shape[-2], y.shape[-1]):
            temp = spatial_bicubic(temp, (y.shape[-2], y.shape[-1]))
        stc = temp

        # Align time lengths
        def _align_T(a: torch.Tensor, T: int) -> torch.Tensor:
            if a.size(2) == T: return a
            if a.size(2) > T:  return a[:, :, :T]
            pad = T - a.size(2)
            pad_tail = torch.zeros((a.size(0), a.size(1), pad, a.size(3), a.size(4)), device=a.device, dtype=a.dtype)
            return torch.cat([a, pad_tail], dim=2)

        pred = _align_T(pred, y.size(2))
        bic  = _align_T(bic,  y.size(2))
        temp = _align_T(temp, y.size(2))
        stc  = _align_T(stc,  y.size(2))

        # --- STANDARDIZED metrics/images ---
        if metrics_sum_std is not None:
            # Clamp to a sane standardized band
            pred_std = pred.clamp(-10.0, 10.0); y_std = y
            bic_std  = bic.clamp(-10.0, 10.0);  temp_std = temp.clamp(-10.0, 10.0); stc_std = stc.clamp(-10.0, 10.0)

            B, C, T, H, W = y_std.shape
            def bt(t): return t.permute(0,2,1,3,4).reshape(B*T, C, H, W)

            y_bt = bt(y_std); pr_bt = bt(pred_std); bi_bt = bt(bic_std); tp_bt = bt(temp_std); sc_bt = bt(stc_std)

            dr_std = 4.0  # standardized dynamic range for PSNR/SSIM
            metrics_sum_std["model_RMSE"] += rmse(pr_bt, y_bt)
            metrics_sum_std["model_MAE"]  += mae(pr_bt, y_bt)
            metrics_sum_std["model_PSNR"] += psnr(pr_bt, y_bt, data_range=dr_std)
            metrics_sum_std["model_SSIM"] += ssim_torch(pr_bt, y_bt, data_range=dr_std)

            metrics_sum_std["bicubic_RMSE"] += rmse(bi_bt, y_bt)
            metrics_sum_std["bicubic_MAE"]  += mae(bi_bt, y_bt)
            metrics_sum_std["bicubic_PSNR"] += psnr(bi_bt, y_bt, data_range=dr_std)
            metrics_sum_std["bicubic_SSIM"] += ssim_torch(bi_bt, y_bt, data_range=dr_std)

            metrics_sum_std["tempinterp_RMSE"] += rmse(tp_bt, y_bt)
            metrics_sum_std["tempinterp_MAE"]  += mae(tp_bt, y_bt)
            metrics_sum_std["tempinterp_PSNR"] += psnr(tp_bt, y_bt, data_range=dr_std)
            metrics_sum_std["tempinterp_SSIM"] += ssim_torch(tp_bt, y_bt, data_range=dr_std)

            metrics_sum_std["st_combined_RMSE"] += rmse(sc_bt, y_bt)
            metrics_sum_std["st_combined_MAE"]  += mae(sc_bt, y_bt)
            metrics_sum_std["st_combined_PSNR"] += psnr(sc_bt, y_bt, data_range=dr_std)
            metrics_sum_std["st_combined_SSIM"] += ssim_torch(sc_bt, y_bt, data_range=dr_std)

            # save example images
            if di < args.num_examples:
                day_str = extract_yyyymmdd_from_name(p) or f"day{di:04d}"
                def sel_ch(t, ci): return t[:, ci:ci+1]
                for h in args.example_hours:
                    if h < y_std.shape[2]:
                        plot_full_maps(args.out_dir, day_str, plot_var, h,
                                       hr=sel_ch(y_std, ch_idx_plot),
                                       pred=sel_ch(pred_std, ch_idx_plot),
                                       bic=sel_ch(bic_std, ch_idx_plot),
                                       temp=sel_ch(temp_std, ch_idx_plot),
                                       stc=sel_ch(stc_std, ch_idx_plot),
                                       suffix="std")

        # --- DE-STANDARDIZED metrics/images ---
        if metrics_sum_dst is not None:
            # Convert both predictions and targets to physical units
            y_dst   = destandardize_tensor(y,    stats, var_names)
            pr_dst  = destandardize_tensor(pred, stats, var_names)
            bi_dst  = destandardize_tensor(bic,  stats, var_names)
            tp_dst  = destandardize_tensor(temp, stats, var_names)
            sc_dst  = destandardize_tensor(stc,  stats, var_names)

            B, C, T, H, W = y_dst.shape
            def bt(t): return t.permute(0,2,1,3,4).reshape(B*T, C, H, W)

            y_bt = bt(y_dst); pr_bt = bt(pr_dst); bi_bt = bt(bi_dst); tp_bt = bt(tp_dst); sc_bt = bt(sc_dst)

            # dynamic data range based on target (physical units)
            dr_dst = dynamic_data_range(y_bt, mode="percentile")

            metrics_sum_dst["model_RMSE"] += rmse(pr_bt, y_bt)
            metrics_sum_dst["model_MAE"]  += mae(pr_bt, y_bt)
            metrics_sum_dst["model_PSNR"] += psnr(pr_bt, y_bt, data_range=dr_dst)
            metrics_sum_dst["model_SSIM"] += ssim_torch(pr_bt, y_bt, data_range=dr_dst)

            metrics_sum_dst["bicubic_RMSE"] += rmse(bi_bt, y_bt)
            metrics_sum_dst["bicubic_MAE"]  += mae(bi_bt, y_bt)
            metrics_sum_dst["bicubic_PSNR"] += psnr(bi_bt, y_bt, data_range=dr_dst)
            metrics_sum_dst["bicubic_SSIM"] += ssim_torch(bi_bt, y_bt, data_range=dr_dst)

            metrics_sum_dst["tempinterp_RMSE"] += rmse(tp_bt, y_bt)
            metrics_sum_dst["tempinterp_MAE"]  += mae(tp_bt, y_bt)
            metrics_sum_dst["tempinterp_PSNR"] += psnr(tp_bt, y_bt, data_range=dr_dst)
            metrics_sum_dst["tempinterp_SSIM"] += ssim_torch(tp_bt, y_bt, data_range=dr_dst)

            metrics_sum_dst["st_combined_RMSE"] += rmse(sc_bt, y_bt)
            metrics_sum_dst["st_combined_MAE"]  += mae(sc_bt, y_bt)
            metrics_sum_dst["st_combined_PSNR"] += psnr(sc_bt, y_bt, data_range=dr_dst)
            metrics_sum_dst["st_combined_SSIM"] += ssim_torch(sc_bt, y_bt, data_range=dr_dst)

            # save example images
            if di < args.num_examples:
                day_str = extract_yyyymmdd_from_name(p) or f"day{di:04d}"
                def sel_ch(t, ci): return t[:, ci:ci+1]
                for h in args.example_hours:
                    if h < y_dst.shape[2]:
                        plot_full_maps(args.out_dir, day_str, plot_var, h,
                                       hr=sel_ch(y_dst, ch_idx_plot),
                                       pred=sel_ch(pr_dst, ch_idx_plot),
                                       bic=sel_ch(bi_dst, ch_idx_plot),
                                       temp=sel_ch(tp_dst, ch_idx_plot),
                                       stc=sel_ch(sc_dst, ch_idx_plot),
                                       suffix="destd")

        n_days += 1

    # Write CSV (with suffixes for clarity)
    out_csv = os.path.join(args.out_dir, args.results_csv)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        if metrics_sum_std is not None:
            for k, v in sorted(metrics_sum_std.items()):
                w.writerow([f"{k}_std", f"{(v / max(n_days,1)):.6f}"])
        if metrics_sum_dst is not None:
            for k, v in sorted(metrics_sum_dst.items()):
                w.writerow([f"{k}_destd", f"{(v / max(n_days,1)):.6f}"])

    print(f"[Done] Wrote metrics to {out_csv}")
    if metrics_sum_std is not None:
        print("=== Averages (standardized) ===")
        for k, v in sorted(metrics_sum_std.items()):
            print(f"{k:24s}: {(v / max(n_days,1)):.6f}")
    if metrics_sum_dst is not None:
        print("=== Averages (destandardized) ===")
        for k, v in sorted(metrics_sum_dst.items()):
            print(f"{k:24s}: {(v / max(n_days,1)):.6f}")


if __name__ == "__main__":
    main()
