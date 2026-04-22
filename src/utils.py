"""
src/utils.py — Shared Utilities
================================
Provides:
  • ``setup_logging``  — dual stdout + file logging
  • ``load_config``    — YAML config loader
  • ``save_checkpoint`` / ``load_checkpoint`` — model persistence
  • ``EarlyStopping``  — training callback with patience
  • ``compute_metrics`` — accuracy + AUC (standard MedMNIST metrics)
  • ``get_device``     — CUDA / MPS / CPU selection
  • ``save_results``   — JSON result serialisation
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    Configure the root logger to write simultaneously to stdout and a
    timestamped log file under ``log_dir``.

    Calling this function more than once in the same process is safe — the
    root logger's handlers are cleared before new ones are added.

    Args:
        log_dir:   Directory where the ``.log`` file is created.
        log_level: Python logging level name (``'DEBUG'``, ``'INFO'``, …).

    Returns:
        Root logger (already configured).
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_fmt  = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    level    = getattr(logging, log_level.upper(), logging.INFO)

    # Clear any handlers attached by a previous call
    root = logging.getLogger()
    root.handlers.clear()

    log_file = Path(log_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=level,
        format=log_fmt,
        datefmt=date_fmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    # Suppress noisy third-party loggers
    for lib in ("matplotlib", "PIL", "urllib3", "numba"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    return root


# =============================================================================
# Configuration
# =============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file and return it as a plain dictionary.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    logging.getLogger(__name__).info("Config loaded from: %s", config_path)
    return cfg


# =============================================================================
# Checkpoint I/O
# =============================================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
    is_best: bool = False,
) -> None:
    """
    Serialise model + optimiser state to a ``.pth`` file.

    When ``is_best=True`` the checkpoint is additionally saved as
    ``best_model.pth`` (overwriting any previous best).

    Args:
        model:          The PyTorch model.
        optimizer:      The optimiser (so training can be resumed exactly).
        epoch:          Current (1-based) epoch number.
        metrics:        Dictionary of metric values to embed in the file.
        checkpoint_dir: Destination directory.
        filename:       Name of the rolling checkpoint file.
        is_best:        Whether this is the best model so far.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics":              metrics,
    }

    # Rolling checkpoint (overwrites each epoch)
    torch.save(payload, Path(checkpoint_dir) / filename)

    # Best model (saved only when AUC improves)
    if is_best:
        best_path = Path(checkpoint_dir) / "best_model.pth"
        torch.save(payload, best_path)
        logging.getLogger(__name__).info(
            "★  New best model → %s  (epoch=%d, val_auc=%.4f)",
            best_path,
            epoch,
            metrics.get("val_auc", float("nan")),
        )


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Tuple[int, Dict[str, float]]:
    """
    Restore model (and optionally optimiser) state from a checkpoint file.

    Args:
        checkpoint_path: Path to the ``.pth`` file.
        model:           Model to restore weights into.
        optimizer:       Optionally restore optimiser state (for resuming).
        device:          Device to map checkpoint tensors onto.

    Returns:
        ``(epoch, metrics)`` tuple extracted from the checkpoint.
    """
    if device is None:
        device = torch.device("cpu")

    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    epoch   = payload.get("epoch", 0)
    metrics = payload.get("metrics", {})

    logging.getLogger(__name__).info(
        "Checkpoint loaded: %s  (epoch=%d)", checkpoint_path, epoch
    )
    return epoch, metrics


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """
    Stops training when a monitored metric has not improved for ``patience``
    consecutive epochs.

    Usage::

        early_stop = EarlyStopping(patience=15, mode='max')
        for epoch in range(num_epochs):
            ...
            if early_stop(val_auc, epoch):
                break   # triggers when should_stop is True

    Args:
        patience:  Number of epochs to wait without improvement.
        min_delta: Minimum absolute change that counts as an improvement.
        mode:      ``'max'`` for AUC/Accuracy, ``'min'`` for loss.
        verbose:   Print patience counter each epoch.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "max",
        verbose: bool = True,
    ) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.verbose   = verbose

        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.best_epoch = 0
        self.counter    = 0
        self.should_stop = False

    # ------------------------------------------------------------------
    def __call__(self, current: float, epoch: int) -> bool:
        """
        Update internal state and return ``True`` if training should stop.

        Args:
            current: Metric value for the current epoch.
            epoch:   Current (1-based) epoch number.

        Returns:
            ``True`` if the patience limit has been reached.
        """
        improved = (
            (self.mode == "max" and current > self.best_value + self.min_delta)
            or
            (self.mode == "min" and current < self.best_value - self.min_delta)
        )

        if improved:
            self.best_value = current
            self.best_epoch = epoch
            self.counter    = 0
        else:
            self.counter += 1
            if self.verbose:
                logging.getLogger(__name__).info(
                    "EarlyStopping: no improvement for %d/%d epochs "
                    "(best=%.4f @ epoch %d)",
                    self.counter, self.patience,
                    self.best_value, self.best_epoch,
                )

        if self.counter >= self.patience:
            self.should_stop = True
            logging.getLogger(__name__).info(
                "Early stopping fired at epoch %d. "
                "Best: %.4f @ epoch %d",
                epoch, self.best_value, self.best_epoch,
            )

        return self.should_stop


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    all_labels: np.ndarray,
    all_preds:  np.ndarray,
    all_probs:  np.ndarray,
) -> Dict[str, float]:
    """
    Compute the two standard MedMNIST evaluation metrics.

    Args:
        all_labels: Ground-truth class indices, shape (N,).
        all_preds:  Predicted class indices, shape (N,).
        all_probs:  Predicted probability for the *positive* class, shape (N,).

    Returns:
        Dict with keys ``'accuracy'`` and ``'auc'``.
    """
    accuracy = float(accuracy_score(all_labels, all_preds))

    try:
        auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        # Raised when only one class is present in the batch (can happen
        # with very small val sets or early training epochs).
        auc = 0.0

    return {"accuracy": accuracy, "auc": auc}


# =============================================================================
# Device selection
# =============================================================================

def get_device() -> torch.device:
    """
    Return the best available compute device in priority order:
    CUDA GPU → Apple MPS → CPU.

    Returns:
        ``torch.device`` object.
    """
    log = logging.getLogger(__name__)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Compute device: CUDA GPU — %s", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Compute device: Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        log.info("Compute device: CPU (no accelerator detected)")

    return device


# =============================================================================
# Results I/O
# =============================================================================

def save_results(
    results: Dict[str, Any],
    results_dir: str,
    filename: str = "results.json",
) -> None:
    """
    Serialise a results dictionary to a pretty-printed JSON file.

    Handles NumPy scalar / array types by converting them to Python
    native types before serialisation.

    Args:
        results:     Dictionary to save.
        results_dir: Destination directory (created if absent).
        filename:    Output filename.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        """Recursively convert NumPy types to JSON-serialisable Python types."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(i) for i in obj]
        return obj

    out_path = Path(results_dir) / filename
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(_convert(results), fh, indent=2)

    logging.getLogger(__name__).info("Results written to: %s", out_path)
