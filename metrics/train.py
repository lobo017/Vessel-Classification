"""
src/train.py — Training Loop for VesselMNIST3D
================================================
Features
--------
• AdamW optimiser with decoupled weight decay (bias / BN params excluded).
• Linear warm-up for ``warmup_epochs`` followed by cosine-annealing LR decay.
• Mixed-precision training (``torch.amp``) — ~2× GPU speedup, ~50 % VRAM saving.
• Class-weighted cross-entropy with label smoothing to handle class imbalance.
• Gradient clipping to guard against exploding gradients in 3-D convs.
• Early stopping monitored on validation AUC.
• Rolling checkpoint (``last_checkpoint.pth``) + best checkpoint (``best_model.pth``).
• TensorBoard logging of all metrics and the learning rate.

Run
---
    python -m src.train --config configs/config.yaml
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import (
    EarlyStopping,
    compute_metrics,
    get_device,
    load_config,
    save_checkpoint,
    setup_logging,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Single Epoch
# =============================================================================

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    cfg:       dict,
    is_train:  bool = True,
) -> Dict[str, float]:
    """
    Execute one full pass over the dataset (training **or** evaluation).

    During training:
      • Gradients are computed and the optimiser step is executed.
      • Mixed-precision forward/backward via ``torch.amp.autocast``.
      • Gradients are clipped before the optimiser step.

    During evaluation:
      • ``torch.no_grad()`` context suppresses gradient tracking.
      • Model is placed in ``eval()`` mode (disables dropout / BN stats update).

    Args:
        model:     The 3-D classification network.
        loader:    DataLoader for the split.
        criterion: Loss function (weighted cross-entropy).
        optimizer: Optimiser (only used when ``is_train=True``).
        device:    Compute device.
        cfg:       Full configuration dictionary.
        is_train:  If ``True`` update weights; if ``False`` run validation.

    Returns:
        Dict with keys ``'loss'``, ``'accuracy'``, ``'auc'``.
    """
    model.train() if is_train else model.eval()

    grad_clip  = float(cfg["training"].get("grad_clip", 1.0))
    use_amp    = device.type == "cuda"   # AMP only beneficial on CUDA
    dev_type   = device.type             # 'cuda' | 'mps' | 'cpu'

    # Gradient scaler — handles under/overflow in fp16 arithmetic
    # Created fresh each epoch to avoid stale scale factors
    scaler = torch.amp.GradScaler(device=dev_type, enabled=use_amp)

    running_loss = 0.0
    all_labels: list = []
    all_preds:  list = []
    all_probs:  list = []

    ctx_manager = torch.enable_grad() if is_train else torch.no_grad()
    phase_label = "Train" if is_train else "  Val"

    with ctx_manager:
        for volumes, labels in tqdm(loader, desc=phase_label, leave=False, unit="batch"):
            # ── Move to device ────────────────────────────────────────────────
            volumes = volumes.to(device, non_blocking=True)   # (B, 1, D, H, W)
            labels  = labels.to(device,  non_blocking=True)   # (B,)

            # ── Forward (with optional AMP) ────────────────────────────────
            with torch.amp.autocast(device_type=dev_type, enabled=use_amp):
                logits = model(volumes)         # (B, num_classes)
                loss   = criterion(logits, labels)

            if is_train:
                # ── Backward ─────────────────────────────────────────────────
                optimizer.zero_grad(set_to_none=True)   # slightly faster than zero_grad()
                scaler.scale(loss).backward()

                # Unscale before clipping so clip threshold is in true fp32 scale
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                scaler.step(optimizer)
                scaler.update()

            # ── Accumulate predictions ────────────────────────────────────
            running_loss += loss.item() * volumes.size(0)

            # Positive class (aneurysm) probability
            probs = torch.softmax(logits.detach(), dim=1)[:, 1]
            preds = logits.detach().argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # ── Epoch-level metrics ───────────────────────────────────────────────────
    epoch_loss = running_loss / len(loader.dataset)
    metrics    = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )
    metrics["loss"] = epoch_loss
    return metrics


# =============================================================================
# Optimiser
# =============================================================================

def build_optimizer(
    model: nn.Module,
    cfg:   dict,
) -> torch.optim.Optimizer:
    """
    Construct AdamW with selective weight decay.

    Weight decay is **not** applied to:
      • BatchNorm parameters (weight/bias)
      • All bias terms
    This matches the original AdamW paper recommendation and avoids
    regularising scale-and-shift parameters in BN layers.

    Args:
        model: The model whose parameters are to be optimised.
        cfg:   Full configuration dictionary.

    Returns:
        Configured ``torch.optim.AdamW`` instance.
    """
    t = cfg["training"]

    decay_params    = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Bias terms and all BN learnable params skip weight decay
        if "bias" in name or "bn" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": float(t.get("weight_decay", 0.01))},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=float(t.get("learning_rate", 1e-3)),
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    logger.info(
        "AdamW | lr=%.1e | wd=%.1e | decay params=%d | no-decay params=%d",
        t.get("learning_rate"),
        t.get("weight_decay"),
        len(decay_params),
        len(no_decay_params),
    )
    return optimizer


# =============================================================================
# LR Scheduler
# =============================================================================

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg:       dict,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Build a two-phase LR schedule:
      1. **Linear warm-up**: LR grows from 0 → ``learning_rate`` over
         ``warmup_epochs`` epochs.  Prevents instability from large random
         gradients at the very start of 3-D conv training.
      2. **Cosine annealing**: LR decays smoothly from ``learning_rate`` →
         ``eta_min`` for the remaining epochs, avoiding sharp LR drops.

    Args:
        optimizer: The optimiser to attach the schedule to.
        cfg:       Full configuration dictionary.

    Returns:
        ``torch.optim.lr_scheduler.SequentialLR`` combining both phases.
    """
    t = cfg["training"]
    num_epochs    = int(t.get("num_epochs", 100))
    warmup_epochs = int(t.get("warmup_epochs", 5))

    # ── Phase 1: linear warm-up ───────────────────────────────────────────────
    def _warmup_lambda(epoch: int) -> float:
        """Return scale factor for the warm-up phase."""
        return float(epoch + 1) / float(max(warmup_epochs, 1))

    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_warmup_lambda)

    # ── Phase 2: cosine annealing ─────────────────────────────────────────────
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(num_epochs - warmup_epochs, 1),
        eta_min=1e-6,
    )

    # SequentialLR switches from warmup to cosine at epoch == warmup_epochs
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )

    logger.info(
        "LR schedule: warmup(%d ep) → cosine(T_max=%d, eta_min=1e-6)",
        warmup_epochs,
        num_epochs - warmup_epochs,
    )
    return scheduler


# =============================================================================
# Main Training Function
# =============================================================================

def train(cfg: dict) -> None:
    """
    End-to-end training pipeline.

    Steps
    -----
    1. Setup: logging, device, output directories.
    2. Load datasets and construct DataLoaders.
    3. Build model, loss, optimiser, and scheduler.
    4. Run training loop with validation, checkpointing, and early stopping.
    5. Log all metrics to TensorBoard.

    Args:
        cfg: Configuration dictionary loaded from ``config.yaml``.
    """
    out = cfg.get("output", {})
    log_dir        = out.get("log_dir",        "./outputs/logs")
    checkpoint_dir = out.get("checkpoint_dir", "./outputs/checkpoints")
    log_level      = out.get("log_level",      "INFO")

    # ── 1. Setup ──────────────────────────────────────────────────────────────
    setup_logging(log_dir, log_level=log_level)
    device = get_device()

    for d in (log_dir, checkpoint_dir, out.get("results_dir", "./outputs/results")):
        Path(d).mkdir(parents=True, exist_ok=True)

    # ── 2. Data ────────────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("VESSELMNIST3D — 3D ResNet-SE Training")
    logger.info("=" * 70)
    logger.info("Loading datasets …")
    dataloaders = get_dataloaders(cfg)

    # ── 3a. Model ──────────────────────────────────────────────────────────────
    logger.info("Building model …")
    model = build_model(cfg).to(device)

    # ── 3b. Weighted loss (handles class imbalance) ────────────────────────────
    # Inverse-frequency weighting: minority class gets higher loss contribution.
    train_labels  = dataloaders["train"].dataset.labels.flatten().astype(int)
    class_counts  = np.bincount(train_labels, minlength=2)
    total_samples = len(train_labels)
    # weight[c] = total / (num_classes * count[c])   — sklearn convention
    weights = torch.tensor(
        [total_samples / (2.0 * max(c, 1)) for c in class_counts],
        dtype=torch.float32,
    ).to(device)

    logger.info(
        "Class counts → %s  |  CE weights → [%.3f, %.3f]",
        dict(enumerate(class_counts.tolist())),
        weights[0].item(),
        weights[1].item(),
    )

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.1)),
    )

    # ── 3c. Optimiser & Scheduler ─────────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # ── 3d. Early stopping ─────────────────────────────────────────────────────
    early_stopping = EarlyStopping(
        patience=int(cfg["training"].get("early_stopping_patience", 15)),
        mode="max",
        verbose=True,
    )

    # ── 3e. TensorBoard ────────────────────────────────────────────────────────
    tb_dir = Path(log_dir) / "tensorboard"
    writer = SummaryWriter(log_dir=str(tb_dir))
    logger.info("TensorBoard log dir: %s", tb_dir)
    logger.info("  → run: tensorboard --logdir %s", tb_dir)

    # ── 4. Training loop ───────────────────────────────────────────────────────
    num_epochs    = int(cfg["training"].get("num_epochs", 100))
    best_val_auc  = 0.0
    train_history = []   # kept for optional post-training plot

    logger.info("Starting training — %d epochs on %s", num_epochs, device)
    logger.info("=" * 70)

    for epoch in range(1, num_epochs + 1):
        t0 = time.perf_counter()

        # Current LR (read before stepping the scheduler)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Training epoch ────────────────────────────────────────────────────
        train_metrics = run_epoch(
            model, dataloaders["train"], criterion,
            optimizer, device, cfg, is_train=True,
        )

        # ── Validation epoch ──────────────────────────────────────────────────
        val_metrics = run_epoch(
            model, dataloaders["val"], criterion,
            optimizer, device, cfg, is_train=False,
        )

        # ── Advance LR schedule ───────────────────────────────────────────────
        scheduler.step()

        elapsed = time.perf_counter() - t0

        # ── Console log ───────────────────────────────────────────────────────
        logger.info(
            "Ep[%03d/%03d]  lr=%.2e  "
            "Train loss=%.4f acc=%.4f auc=%.4f  "
            "Val   loss=%.4f acc=%.4f auc=%.4f  "
            "(%ds)",
            epoch, num_epochs, current_lr,
            train_metrics["loss"], train_metrics["accuracy"], train_metrics["auc"],
            val_metrics["loss"],   val_metrics["accuracy"],   val_metrics["auc"],
            int(elapsed),
        )

        # ── TensorBoard ───────────────────────────────────────────────────────
        for phase, m in (("train", train_metrics), ("val", val_metrics)):
            for name, value in m.items():
                writer.add_scalar(f"{phase}/{name}", value, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        # ── Checkpointing ─────────────────────────────────────────────────────
        is_best = val_metrics["auc"] > best_val_auc
        if is_best:
            best_val_auc = val_metrics["auc"]

        combined_metrics = {
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}":   v for k, v in val_metrics.items()},
        }

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=combined_metrics,
            checkpoint_dir=checkpoint_dir,
            filename="last_checkpoint.pth",
            is_best=is_best,
        )

        # Record for potential history export
        train_history.append({"epoch": epoch, **combined_metrics, "lr": current_lr})

        # ── Early stopping ─────────────────────────────────────────────────────
        if early_stopping(val_metrics["auc"], epoch):
            logger.info("Early stopping — ending training.")
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    writer.close()
    logger.info("=" * 70)
    logger.info(
        "Training complete.  Best Val AUC: %.4f (epoch %d)",
        best_val_auc,
        early_stopping.best_epoch,
    )
    logger.info("Best checkpoint: %s/best_model.pth", checkpoint_dir)
    logger.info("=" * 70)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    """Parse CLI arguments and launch training."""
    parser = argparse.ArgumentParser(
        description="Train 3D ResNet-SE on VesselMNIST3D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    # Guard required on Windows for DataLoader multiprocessing
    main()
