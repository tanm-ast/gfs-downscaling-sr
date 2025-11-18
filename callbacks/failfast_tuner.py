# callbacks/failfast_tuner.py
# Stops quickly on NaNs / exploding val metric; also applies simple LR decay on short plateaus.

import math
import pytorch_lightning as pl

class FailFastTuner(pl.Callback):
    def __init__(self, monitor: str = "val/monitor", patience: int = 2, explode_factor: float = 5.0, min_lr: float = 1e-6):
        self.monitor = monitor
        self.patience = patience
        self.explode_factor = explode_factor
        self.min_lr = min_lr
        self.best = None
        self.bad_epochs = 0

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        val = metrics.get(self.monitor, None)
        if val is None:
            return  # nothing to monitor yet
        val = float(val)

        # NaN/inf → stop
        if not math.isfinite(val):
            pl_module.print(f"[FailFast] {self.monitor} is not finite ({val}). Stopping.")
            trainer.should_stop = True
            return

        # Track best; plateau handling
        if (self.best is None) or (val < self.best):
            self.best = val
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                # simple LR decay, then reset patience
                try:
                    opt = trainer.optimizers[0]
                    for g in opt.param_groups:
                        g["lr"] = max(g["lr"] * 0.5, self.min_lr)
                    pl_module.log("tuner/lr", opt.param_groups[0]["lr"], prog_bar=True)
                except Exception:
                    pass
                self.bad_epochs = 0

        # Exploding vs best → stop
        if (self.best is not None) and (val > self.best * self.explode_factor):
            pl_module.print(f"[FailFast] {self.monitor} exploded: {val:.4g} vs best {self.best:.4g}. Stopping.")
            trainer.should_stop = True
