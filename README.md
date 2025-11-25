# GFS Downscaling with Factorized Super-Resolution

This repository contains a prototype deep learning pipeline for spatio-temporal
super-resolution of GFS forecast fields.

## Features
- Factorized model with temporal (dilated 1D conv) + spatial (EDSR) upscaling
- Training on standardized patches with temporal coherence loss
- Baseline comparisons: bicubic, temporal spline, spatio-temporal
- Full-field evaluation with SSIM/PSNR/RMSE/MAE metrics
- Plotting and qualitative visualization of results

## Getting Started
```bash
pip install -r requirements.txt
python train_factorized.py --data_root ../datasets/GFS --ckpt ./checkpoints/...
