# (Optional) Copy best checkpoint and supporting assets into a release folder.
# Lightweight runtime estimator from dataloader sizes and observed throughput.

from typing import Optional, Dict

def estimate_runtime(train_loader, observed_tps: Optional[float] = None) -> Dict[str, float]:
    """
    Returns a dict with:
      - steps_per_epoch
      - samples_per_epoch
      - secs_per_epoch  (estimated)

    If observed_tps (samples/sec) is None, it assumes a conservative default.
    """
    steps = len(train_loader)                                   # number of batches per epoch
    batch_size = getattr(train_loader, "batch_size", 1) or 1    # samples per batch
    samples = float(steps * batch_size)                          # total samples per epoch
    tps = float(observed_tps) if observed_tps is not None else 150.0  # default for a 3060-ish run
    secs = samples / tps
    return {
        "steps_per_epoch": float(steps),
        "samples_per_epoch": samples,
        "secs_per_epoch": secs,
    }