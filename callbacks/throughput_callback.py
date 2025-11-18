# callbacks/throughput_callback.py
# Logs "perf/throughput" (samples/sec) every N steps. Model-agnostic.


from __future__ import annotations
import time
from typing import Optional, Any
import torch
import pytorch_lightning as pl

class ThroughputCallback(pl.Callback):
    """
    Measures training throughput (samples/sec) over a sliding window of batches.
    - Robustly infers batch size from the batch first; falls back to dataloader or datamodule.
    - Avoids type checker warnings by guarding Optional values and callables.
    - Logs metric as "perf/throughput" on_step (not on_epoch) every `every_n_steps`.

    Attributes
    ----------
    every : int
        Log throughput every this many training batches.
    last_tps : Optional[float]
        The most recently computed throughput value (samples/sec).
    """
    def __init__(self, every_n_steps: int = 50):
        self.every: int = int(every_n_steps)
        self._t0: Optional[float] = None
        self._seen: int = 0
        self._count: int = 0
        self.last_tps: Optional[float] = None

      # -------- internals --------

    @staticmethod
    def _infer_batch_size_from_batch(batch: Any) -> Optional[int]:
        """
        Try to read B from batch[0], assuming a typical (x, y, ...) with x as Tensor or tuple/list of Tensors.
        Returns None if it cannot infer.
        """
        try:
            x = batch[0]
            if torch.is_tensor(x):
                return int(x.size(0))
            if isinstance(x, (list, tuple)) and len(x) > 0 and torch.is_tensor(x[0]):
                return int(x[0].size(0))
        except Exception:
            pass
        return None

    @staticmethod
    def _infer_batch_size_from_trainer(trainer: pl.Trainer) -> Optional[int]:
        """
        Try to read batch_size from trainer's dataloader(s) safely. Handles:
        - trainer.train_dataloader as object or as callable
        - trainer.datamodule.train_dataloader() if available
        Returns None if it cannot infer.
        """
        # Case 1: trainer.train_dataloader is a dataloader object
        dl_obj = getattr(trainer, "train_dataloader", None)
        if dl_obj is not None and not callable(dl_obj):
            try:
                bs = getattr(dl_obj, "batch_size", None)
                if isinstance(bs, int) and bs > 0:
                    return int(bs)
            except Exception:
                pass

        # Case 2: trainer.train_dataloader is a callable (LightningModule/Datamodule style)
        if callable(dl_obj):
            try:
                dl = dl_obj()
                bs = getattr(dl, "batch_size", None)
                if isinstance(bs, int) and bs > 0:
                    return int(bs)
            except Exception:
                pass

        # Case 3: datamodule
        dm = getattr(trainer, "datamodule", None)
        if dm is not None and hasattr(dm, "train_dataloader"):
            try:
                dl = dm.train_dataloader()
                bs = getattr(dl, "batch_size", None)
                if isinstance(bs, int) and bs > 0:
                    return int(bs)
            except Exception:
                pass

        return None

    # -------- Lightning hooks --------
    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch, batch_idx: int) -> None:
        # Initialize timing window on the very first seen batch.
        if self._t0 is None:
            self._t0 = time.time()

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        # 1) Infer batch size robustly
        bsz = self._infer_batch_size_from_batch(batch)
        if bsz is None:
            bsz = self._infer_batch_size_from_trainer(trainer)
        if bsz is None:
            bsz = 1  # ultimate fallback

        # 2) Accumulate counts
        self._seen += max(int(bsz), 1)
        self._count += 1

        # 3) If window reached, compute throughput
        if self._count % self.every == 0:
            # Guard _t0 to satisfy type checkers and avoid None subtraction
            if self._t0 is None:
                # Should not happen because we set it at batch_start, but be safe:
                self._t0 = time.time()
                return

            now = time.time()
            dt = max(now - self._t0, 1e-9)  # dt is strictly float here
            tps = float(self._seen) / dt
            self.last_tps = tps

            # Log on_step (per Lightning best practice for step-wise metrics)
            pl_module.log(
                "perf/throughput",
                tps,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

            # 4) Reset window
            self._t0 = now
            self._seen = 0
