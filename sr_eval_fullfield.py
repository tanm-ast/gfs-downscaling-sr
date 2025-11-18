#!/usr/bin/env python3
"""
Full-Field Super-Resolution Test & Baseline Evaluation (GFS Downscaling)
- Temporal interpolation: SciPy CubicSpline (direct import)
- SSIM: pure PyTorch implementation (no skimage, no torchmetrics)
- Baselines: spatial bicubic, temporal, spatio-temporal
- Scales: standardized, de-standardized, or both
- NEW: per-variable plots and per-variable metrics
"""

import argparse
import os
import math
import csv
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

# === project imports (your repo) ===
from data.gfs_downscaling import (
    DEFAULT_VARS,
    VarStats,
    build_lr_from_hr,
    apply_standardize,
    compute_var_stats_train,
    split_by_date,
    split_by_ratio,
    extract_yyyymmdd_from_name,
)
from models.factorized_sr import FactorizedSR
from train_factorized import find_files


# ---------------- Metrics ----------------
def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()

def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).item())

def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)).item())

def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 0.0:
        return float("inf")
    dr = max(float(data_range), 1e-6)
    return 20.0 * math.log10(dr) - 10.0 * math.log10(mse)

# ---- Pure-PyTorch SSIM (no skimage/torchmetrics) ----
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
    Args:
        pred, target: [N, C, H, W] tensors on the same device/dtype.
        data_range: dynamic range of the *data* (e.g., 1.0 if in [0,1], or 255.0).
        win_size: size of the Gaussian window (odd).
        sigma: stddev of the Gaussian window.
        reduction: "mean" | "none" (returns map if "none")

    Returns:
        float if reduction="mean"; else a tensor map.
    """
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"
    N, C, H, W = pred.shape
    device, dtype = pred.device, pred.dtype

    # Gaussian window (same for all channels; use depthwise conv)
    k = _gaussian_kernel2d(win_size, sigma, device, dtype)
    k = k.view(1, 1, win_size, win_size).repeat(C, 1, 1, 1)  # [C,1,K,K]

    # Pad so output size == input size
    pad = win_size // 2

    mu_x = F.conv2d(pred, k, padding=pad, groups=C)
    mu_y = F.conv2d(target, k, padding=pad, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(pred * pred, k, padding=pad, groups=C) - mu_x2
    sigma_y = F.conv2d(target * target, k, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(pred * target, k, padding=pad, groups=C) - mu_xy

    # SSIM constants
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    ssim_map = ((2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x + sigma_y + C2))

    if reduction == "none":
        return ssim_map  # type: ignore[return-value]

    return float(ssim_map.mean().item())

def dynamic_data_range(t: torch.Tensor, mode: str = "percentile") -> float:
    """
    Robust data_range for PSNR/SSIM on physical units.
    Uses 1–99th percentiles of the target by default.
    """
    arr = np.asarray(to_numpy(t))
    lo = np.percentile(arr, 1.0)
    hi = np.percentile(arr, 99.0)
    return float(max(hi - lo, 1e-6))


# ------------- (De-)Standardization -------------
def destandardize_tensor(x: torch.Tensor, stats: VarStats, var_names: List[str]) -> torch.Tensor:
    """
    x_std -> x_phys, channel-wise using stats.
    x: [B, C, T, H, W]
    """
    y = x.clone()
    for ci, v in enumerate(var_names):
        m = float(stats.mean[v]); s = float(stats.std[v])
        y[:, ci] = y[:, ci] * (s + 1e-6) + m
    return y


# ------------- Temporal / Spatial baselines -------------
def temporal_interp(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Cubic time interpolation using SciPy's CubicSpline (direct import).
    x: [B, C, T, H, W]  ->  [B, C, T*factor, H, W]
    """
    B, C, T, H, W = x.shape
    if factor <= 1 or T == 1:
        return x

    t_in = np.arange(T, dtype=np.float32)
    t_out = np.linspace(0, T - 1, T * factor, dtype=np.float32)

    x_np = to_numpy(x)  # (B,C,T,H,W)
    y_np = np.empty((B, C, T * factor, H, W), dtype=np.float32)

    for b in range(B):
        for c in range(C):
            # reshape (T, H*W) for vectorized spline along time axis
            series = x_np[b, c].reshape(T, -1)
            cs = CubicSpline(t_in, series, axis=0, bc_type="natural")
            yi = cs(t_out)  # (T*factor, H*W)
            y_np[b, c] = yi.reshape(T * factor, H, W)

    return torch.from_numpy(y_np).to(x.device, dtype=x.dtype)

def spatial_bicubic(x: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Bicubic upsampling per frame.
    Accepts [B,C,T,H,W] or [N,C,H,W].
    """
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
    Saves a multi-panel figure: HR vs Model vs Baselines (full domain).
    Inputs (already in chosen scale): [B=1, C=1, T, H, W]  (single variable)
    """
    os.makedirs(out_dir, exist_ok=True)
    B, C, T, H, W = hr.shape
    assert B == 1 and C == 1, "Pass single-variable tensors into plot_full_maps."

    def _get(img: Optional[torch.Tensor]) -> Optional[np.ndarray]:
        if img is None:
            return None
        return to_numpy(img[0, 0, hour_idx])  # [H,W]

    hr_img = _get(hr)
    md_img = _get(pred)
    bi_img = _get(bic)
    tp_img = _get(temp)
    sc_img = _get(stc)

    if hr_img is None:
        return

    hr_img = np.asarray(hr_img)
    vmin = vmin if vmin is not None else float(np.percentile(hr_img, 2))
    vmax = vmax if vmax is not None else float(np.percentile(hr_img, 98))

    panels = [("HR", hr_img), ("Model", md_img), ("Bicubic", bi_img), ("TempInterp", tp_img), ("ST-Combined", sc_img)]
    panels = [(t, im) for (t, im) in panels if im is not None]

    fig, axs = plt.subplots(1, len(panels), figsize=(5*len(panels), 4), constrained_layout=True)
    if len(panels) == 1:
        axs = [axs]
    for ax, (title, im) in zip(axs, panels):
        imshow_obj = ax.imshow(im, vmin=vmin, vmax=vmax, cmap="jet")
        ax.set_title(title)
        ax.axis("off")
        cbar = plt.colorbar(imshow_obj, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(f"{day_str}  var={var_name}  hour={hour_idx}  [{suffix}]")
    out_path = os.path.join(out_dir, f"{day_str}_{var_name}_h{hour_idx:02d}_{suffix}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ------------- Write per channel metrics to diffferent tables -------------
def _write_per_variable_tables(out_dir: str,
                               metrics_sum: dict[str, dict[str, float]],
                               scale_name: str,
                               n_days: int) -> None:
    """
    metrics_sum: { var_name: { 'model_RMSE': sum, 'model_MAE': sum, ... } }
    Writes per-variable tables with rows=methods and cols=metrics (averaged over days).
    """
    os.makedirs(out_dir, exist_ok=True)
    methods = ["model", "bicubic", "tempinterp", "st_combined"]
    cols = ["RMSE", "MAE", "PSNR", "SSIM"]
    denom = max(n_days, 1)

    for vname, met in sorted(metrics_sum.items()):
        # Build 2D table
        header = ["method"] + cols
        rows = []
        for m in methods:
            row = [m]
            for c in cols:
                key = f"{m}_{c}"
                val = met.get(key, 0.0) / denom
                row.append(f"{val:.6f}")
            rows.append(row)

        # Write a CSV per variable per scale
        csv_path = os.path.join(out_dir, f"table_{scale_name}_{vname}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"{scale_name} :: {vname}"])
            w.writerow(header)
            w.writerows(rows)

        # Pretty print to console
        print(f"\n[{scale_name.upper()}] {vname}")
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
        fmt = "  ".join("{:" + str(w) + "}" for w in widths)
        print(fmt.format(*header))
        for r in rows:
            print(fmt.format(*r))


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

    ap.add_argument("--batch_device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=str, default="eval_out")
    ap.add_argument("--results_csv", type=str, default="metrics_test.csv")
    ap.add_argument("--num_examples", type=int, default=4)
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

    def empty_metrics() -> Dict[str, float]:
        return {
            "model_RMSE":0.0,"model_MAE":0.0,"model_PSNR":0.0,"model_SSIM":0.0,
            "bicubic_RMSE":0.0,"bicubic_MAE":0.0,"bicubic_PSNR":0.0,"bicubic_SSIM":0.0,
            "tempinterp_RMSE":0.0,"tempinterp_MAE":0.0,"tempinterp_PSNR":0.0,"tempinterp_SSIM":0.0,
            "st_combined_RMSE":0.0,"st_combined_MAE":0.0,"st_combined_PSNR":0.0,"st_combined_SSIM":0.0,
        }

    var_names = list(args.variables)
    # Per-variable accumulators (optionally for std / destd)
    metrics_sum_std: Optional[Dict[str, Dict[str, float]]] = (
        {v: empty_metrics() for v in var_names} if args.scales in ("standard","both") else None
    )
    metrics_sum_dst: Optional[Dict[str, Dict[str, float]]] = (
        {v: empty_metrics() for v in var_names} if args.scales in ("destandard","both") else None
    )

    # keep a counter for averaging over files
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

        x = torch.from_numpy(np.transpose(X_lr_std, (3,0,1,2))).unsqueeze(0).float().to(device)  # [B=1, C, T, H, W]
        y = torch.from_numpy(np.transpose(Y_hr_std, (3,0,1,2))).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred = model(x)  # [B, C, T, H, W]

        # Baselines on standardized LR
        bic = spatial_bicubic(x, (y.shape[-2], y.shape[-1]))                 # spatial
        temp = temporal_interp(x, factor=args.temporal_factor)                # temporal
        if (temp.shape[-2], temp.shape[-1]) != (y.shape[-2], y.shape[-1]):
            temp = spatial_bicubic(temp, (y.shape[-2], y.shape[-1]))          # spatial align if needed
        stc = temp                                                            # (kept same as temp here)

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

        # Ensure spatial match just in case
        target_hw = (y.shape[-2], y.shape[-1])
        def ensure_hw(z: torch.Tensor) -> torch.Tensor:
            if z.shape[-2:] != target_hw:
                return spatial_bicubic(z, target_hw)
            return z

        pred = ensure_hw(pred)
        bic  = ensure_hw(bic)
        temp = ensure_hw(temp)
        stc  = ensure_hw(stc)

        # Helper: flatten [B, C, T, H, W] -> [B*T, C, H, W], inferring shapes from input
        def bt(t: torch.Tensor) -> torch.Tensor:
            B_, C_, T_, H_, W_ = t.shape
            return t.contiguous().reshape(B_ * T_, C_, H_, W_)

        # ---- STANDARDIZED per-variable metrics + plots ----
        if metrics_sum_std is not None:
            day_str = extract_yyyymmdd_from_name(p) or f"day{di:04d}"
            # clamp extremes for stability
            pred_std = pred.clamp(-10.0, 10.0); y_std = y
            bic_std  = bic.clamp(-10.0, 10.0);  temp_std = temp.clamp(-10.0, 10.0); stc_std = stc.clamp(-10.0, 10.0)

            # iterate variables
            for ci, vname in enumerate(var_names):
                def sel_ch(t: torch.Tensor, ci: int) -> torch.Tensor:
                    return t[:, ci:ci+1]  # [B,1,T,H,W]

                y_v   = sel_ch(y_std, ci)
                pr_v  = sel_ch(pred_std, ci)
                bi_v  = sel_ch(bic_std, ci)
                tp_v  = sel_ch(temp_std, ci)
                sc_v  = sel_ch(stc_std, ci)

                # flatten time
                y_bt  = bt(y_v)
                pr_bt = bt(pr_v)
                bi_bt = bt(bi_v)
                tp_bt = bt(tp_v)
                sc_bt = bt(sc_v)

                # fixed standardized range
                dr_std = 4.0

                m = metrics_sum_std[vname]
                m["model_RMSE"]       += rmse(pr_bt, y_bt)
                m["model_MAE"]        += mae(pr_bt, y_bt)
                m["model_PSNR"]       += psnr(pr_bt, y_bt, data_range=dr_std)
                m["model_SSIM"]       += ssim_torch(pr_bt, y_bt, data_range=dr_std)

                m["bicubic_RMSE"]     += rmse(bi_bt, y_bt)
                m["bicubic_MAE"]      += mae(bi_bt, y_bt)
                m["bicubic_PSNR"]     += psnr(bi_bt, y_bt, data_range=dr_std)
                m["bicubic_SSIM"]     += ssim_torch(bi_bt, y_bt, data_range=dr_std)

                m["tempinterp_RMSE"]  += rmse(tp_bt, y_bt)
                m["tempinterp_MAE"]   += mae(tp_bt, y_bt)
                m["tempinterp_PSNR"]  += psnr(tp_bt, y_bt, data_range=dr_std)
                m["tempinterp_SSIM"]  += ssim_torch(tp_bt, y_bt, data_range=dr_std)

                m["st_combined_RMSE"] += rmse(sc_bt, y_bt)
                m["st_combined_MAE"]  += mae(sc_bt, y_bt)
                m["st_combined_PSNR"] += psnr(sc_bt, y_bt, data_range=dr_std)
                m["st_combined_SSIM"] += ssim_torch(sc_bt, y_bt, data_range=dr_std)

                # Plots per variable (first N days only)
                if di < args.num_examples:
                    for h in args.example_hours:
                        if h < y_v.shape[2]:
                            plot_full_maps(
                                out_dir=args.out_dir,
                                day_str=day_str,
                                var_name=vname,
                                hour_idx=h,
                                hr=y_v,
                                pred=pr_v,
                                bic=bi_v,
                                temp=tp_v,
                                stc=sc_v,
                                suffix="std"
                            )

        # ---- DE-STANDARDIZED per-variable metrics + plots ----
        if metrics_sum_dst is not None:
            day_str = extract_yyyymmdd_from_name(p) or f"day{di:04d}"

            y_dst   = destandardize_tensor(y,    stats, var_names)
            pr_dst  = destandardize_tensor(pred, stats, var_names)
            bi_dst  = destandardize_tensor(bic,  stats, var_names)
            tp_dst  = destandardize_tensor(temp, stats, var_names)
            sc_dst  = destandardize_tensor(stc,  stats, var_names)

            for ci, vname in enumerate(var_names):
                def sel_ch(t: torch.Tensor, ci: int) -> torch.Tensor:
                    return t[:, ci:ci+1]  # [B,1,T,H,W]

                y_v   = sel_ch(y_dst, ci)
                pr_v  = sel_ch(pr_dst, ci)
                bi_v  = sel_ch(bi_dst, ci)
                tp_v  = sel_ch(tp_dst, ci)
                sc_v  = sel_ch(sc_dst, ci)

                # flatten time
                y_bt  = bt(y_v)
                pr_bt = bt(pr_v)
                bi_bt = bt(bi_v)
                tp_bt = bt(tp_v)
                sc_bt = bt(sc_v)

                # robust range from target of this variable only
                dr_dst = dynamic_data_range(y_bt, mode="percentile")

                m = metrics_sum_dst[vname]
                m["model_RMSE"]       += rmse(pr_bt, y_bt)
                m["model_MAE"]        += mae(pr_bt, y_bt)
                m["model_PSNR"]       += psnr(pr_bt, y_bt, data_range=dr_dst)
                m["model_SSIM"]       += ssim_torch(pr_bt, y_bt, data_range=dr_dst)

                m["bicubic_RMSE"]     += rmse(bi_bt, y_bt)
                m["bicubic_MAE"]      += mae(bi_bt, y_bt)
                m["bicubic_PSNR"]     += psnr(bi_bt, y_bt, data_range=dr_dst)
                m["bicubic_SSIM"]     += ssim_torch(bi_bt, y_bt, data_range=dr_dst)

                m["tempinterp_RMSE"]  += rmse(tp_bt, y_bt)
                m["tempinterp_MAE"]   += mae(tp_bt, y_bt)
                m["tempinterp_PSNR"]  += psnr(tp_bt, y_bt, data_range=dr_dst)
                m["tempinterp_SSIM"]  += ssim_torch(tp_bt, y_bt, data_range=dr_dst)

                m["st_combined_RMSE"] += rmse(sc_bt, y_bt)
                m["st_combined_MAE"]  += mae(sc_bt, y_bt)
                m["st_combined_PSNR"] += psnr(sc_bt, y_bt, data_range=dr_dst)
                m["st_combined_SSIM"] += ssim_torch(sc_bt, y_bt, data_range=dr_dst)

                # Plots per variable (first N days only)
                if di < args.num_examples:
                    for h in args.example_hours:
                        if h < y_v.shape[2]:
                            plot_full_maps(
                                out_dir=args.out_dir,
                                day_str=day_str,
                                var_name=vname,
                                hour_idx=h,
                                hr=y_v,
                                pred=pr_v,
                                bic=bi_v,
                                temp=tp_v,
                                stc=sc_v,
                                suffix="destd"
                            )

        n_days += 1

    # Write CSV (per-variable rows)
    out_csv = os.path.join(args.out_dir, args.results_csv)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scale", "variable", "metric", "value"])
        if metrics_sum_std is not None:
            for vname, met in sorted(metrics_sum_std.items()):
                for k, v in sorted(met.items()):
                    w.writerow(["standard", vname, k, f"{(v / max(n_days,1)):.6f}"])
        if metrics_sum_dst is not None:
            for vname, met in sorted(metrics_sum_dst.items()):
                for k, v in sorted(met.items()):
                    w.writerow(["destandard", vname, k, f"{(v / max(n_days,1)):.6f}"])

    print(f"[Done] Wrote metrics to {out_csv}")


    if metrics_sum_std is not None:
        _write_per_variable_tables(args.out_dir, metrics_sum_std, "standard", n_days)
    if metrics_sum_dst is not None:
        _write_per_variable_tables(args.out_dir, metrics_sum_dst, "destandard", n_days)


if __name__ == "__main__":
    main()
