import os
from typing import Any, Dict

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=20, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, value: float):
        if self.best is None or value < self.best - self.delta:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            self.early_stop = self.counter >= self.patience


def save_checkpoint(model, optimizer, epoch: int, loss: float, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None):
    # Preserve compatibility with the original checkpoints and optimizer state.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("model_state", checkpoint))
    target = model.module if hasattr(model, "module") else model
    try:
        target.load_state_dict(state)
    except RuntimeError:
        # Legacy runs sometimes saved DataParallel-prefixed keys.
        if all(k.startswith("module.") for k in state):
            state = {k[7:]: v for k, v in state.items()}
            target.load_state_dict(state)
        else:
            raise
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def compute_metrics(
    predictions,
    labels,
    adjs,
    threshold=0.18,
    time_windows=(1, 3, 5, 8),
    core_atol=1e-6,
    threshold_mode="threshold",
    detail_level="compact",
) -> Dict[str, Any]:
    """Compute the paper's classification and spatiotemporal metrics.

    The old metric computes classification at sample level with
    scores=max(pred over T,S) and labels=any(core anomaly over T,S), then
    separately computes spatiotemporal hit rates using adjacency.
    """
    from .metrics import compute_metrics as legacy_compute_metrics

    return legacy_compute_metrics(
        predictions,
        labels,
        adjs,
        threshold=threshold,
        time_windows=time_windows,
        core_atol=core_atol,
        threshold_mode=threshold_mode,
        detail_level=detail_level,
    )
