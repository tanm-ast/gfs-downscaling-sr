# utils/trainer_factory.py
# Single place to assemble Trainer, loggers, checkpoints, callbacks.
# Build a Lightning Trainer with W&B logger, checkpointing, LR monitor,
# and the robust ThroughputCallback.


# utils/trainer_factory.py
from __future__ import annotations
from typing import Tuple, Optional, Union, Literal, Any, cast
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from callbacks.throughput_callback import ThroughputCallback
from callbacks.failfast_tuner import FailFastTuner

# Match PL 2.4 precision union (ints and specific strings)
PrecisionArg = Union[
    Literal[16, 32, 64],
    Literal["16-true", "32-true", "64-true", "16-mixed", "bf16-mixed",
            "transformer-engine", "transformer-engine-float16"],
    None,
]

def _normalize_precision(p: Union[str, int, None]) -> PrecisionArg:
    """
    Map user/config inputs to PL's accepted precision literals.
    Handles common aliases and keeps Pylance happy.
    """
    if p is None:
        return None
    if isinstance(p, int):
        # allow 16/32/64 ints directly
        if p in (16, 32, 64):
            return cast(PrecisionArg, p)
        raise ValueError(f"Unsupported integer precision: {p}")
    # strings
    s = p.strip().lower()
    # common aliases -> canonical literals
    alias = {
        "fp16": "16-true",
        "16true": "16-true",
        "fp32": "32-true",
        "32true": "32-true",
        "fp64": "64-true",
        "64true": "64-true",
        "16": "16-true",
        "32": "32-true",
        "64": "64-true",
    }
    s = alias.get(s, s)
    allowed: set[str] = {
        "16-true", "32-true", "64-true", "16-mixed", "bf16-mixed",
        "transformer-engine", "transformer-engine-float16",
    }
    if s in allowed:
        return cast(PrecisionArg, s)
    raise ValueError(
        f"Unsupported precision '{p}'. "
        f"Use one of: 16,32,64 or "
        f"'16-true','32-true','64-true','16-mixed','bf16-mixed'."
    )

def build_trainer(
    out_dir: str,
    *,
    max_epochs: int = 30,
    precision: Union[str, int, None] = "16-mixed",
    grad_clip: float = 1.0,
    accumulate_grad_batches: int = 1,
    log_every_n_steps: int = 25,
    throughput_every_n_steps: int = 50,
    project: str = "gfs-new",
    run_name: str = "factorized-new-losses",
    wandb_mode: str = "online",
    failfast_patience: int = 2,
    accelerator: str = "gpu",
    devices: int = 1, 
) -> Tuple[pl.Trainer, ModelCheckpoint, ThroughputCallback, WandbLogger]:
    """
    Returns:
      trainer, checkpoint_callback, throughput_callback, wandb_logger
    """
    os.makedirs(out_dir, exist_ok=True)
    os.environ["WANDB_MODE"] = wandb_mode

    # Correct logger import for PL 2.4
    wandb_logger = WandbLogger(project=project, entity="tasthan", name=run_name, log_model=True)

    print("[W&B] entity=", wandb_logger.experiment.entity)
    print("[W&B] project=", wandb_logger.experiment.project)
    print("[W&B] url=", getattr(wandb_logger.experiment, "url", "n/a"))
    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir,
        filename="{epoch:03d}-{val_monitor:.4f}",
        monitor="val/monitor",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    th_cb = ThroughputCallback(every_n_steps=throughput_every_n_steps)
    ff_cb = FailFastTuner(monitor="val/monitor", patience=failfast_patience)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        precision=_normalize_precision(precision),  # <- typed & validated
        gradient_clip_val=grad_clip,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[ckpt_cb, lr_cb, th_cb, ff_cb],    # <- FailFastTuner included
        log_every_n_steps=log_every_n_steps,
        enable_checkpointing=True,
    )
    return trainer, ckpt_cb, th_cb, wandb_logger
