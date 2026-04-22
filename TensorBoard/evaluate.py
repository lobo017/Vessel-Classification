"""
src/evaluate.py — Test-Set Evaluation for VesselMNIST3D
=========================================================
Loads the best saved model checkpoint and runs full inference
on the held-out test set.

Outputs saved to ``outputs/results/``:
  ┌──────────────────────────────────────────────────────────┐
  │  confusion_matrix.png    (raw counts + normalised)       │
  │  roc_curve.png           (ROC with AUC + optimal thresh) │
  │  test_results.json       (all metrics, reproducible log) │
  └──────────────────────────────────────────────────────────┘

Run
---
    python -m src.evaluate --config configs/config.yaml
    python -m src.evaluate --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pth
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import (
    get_device,
    load_checkpoint,
    load_config,
    save_results,
    setup_logging,
)

logger = logging.getLogger(__name__)

# Human-readable class names for VesselMNIST3D (class 0 = healthy, 1 = aneurysm)
CLASS_NAMES = ["Healthy Vessel", "Aneurysm"]


# =============================================================================
# Inference
# =============================================================================

def run_inference(
    model:  torch.nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run full forward-pass inference over a DataLoader.

    No gradients are computed; the model is placed in ``eval()`` mode.

    Args:
        model:  Trained classification model.
        loader: DataLoader (any split).
        device: Compute device.

    Returns:
        Dictionary with three arrays, each of length N (dataset size):
          ``'labels'`` — ground-truth class indices (int)
          ``'preds'``  — argmax predicted class indices (int)
          ``'probs'``  — predicted probability for the positive class (float)
    """
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for volumes, labels in tqdm(loader, desc="Inference", unit="batch"):
            volumes = volumes.to(device, non_blocking=True)

            logits = model(volumes)                    # (B, num_classes)
            probs  = torch.softmax(logits, dim=1)      # (B, num_classes)
            preds  = logits.argmax(dim=1)              # (B,)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())   # P(aneurysm)

    return {
        "labels": np.asarray(all_labels, dtype=int),
        "preds":  np.asarray(all_preds,  dtype=int),
        "probs":  np.asarray(all_probs,  dtype=float),
    }


# =============================================================================
# Plot helpers
# =============================================================================

def plot_confusion_matrix(
    labels:      np.ndarray,
    preds:       np.ndarray,
    class_names: list,
    save_path:   str,
) -> None:
    """
    Save a side-by-side confusion matrix (raw counts + row-normalised).

    The normalised panel makes per-class recall immediately visible even
    when class sizes differ significantly.

    Args:
        labels:      True class indices.
        preds:       Predicted class indices.
        class_names: List of human-readable class name strings.
        save_path:   File path for the saved PNG.
    """
    cm = confusion_matrix(labels, preds)
    # Row-normalise: each row sums to 1.0
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "VesselMNIST3D — Test Set Confusion Matrix",
        fontsize=15, fontweight="bold", y=1.01,
    )

    kw_common = dict(
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
    )

    # ── Left: raw counts ──────────────────────────────────────────────────────
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], **kw_common
    )
    axes[0].set_title("Counts", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("True Label",      fontsize=11)
    axes[0].set_xlabel("Predicted Label", fontsize=11)

    # ── Right: row-normalised ─────────────────────────────────────────────────
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues",
        ax=axes[1], vmin=0.0, vmax=1.0, **kw_common,
    )
    axes[1].set_title("Row-Normalised (Recall)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("True Label",      fontsize=11)
    axes[1].set_xlabel("Predicted Label", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix → %s", save_path)


def plot_roc_curve(
    labels:    np.ndarray,
    probs:     np.ndarray,
    save_path: str,
) -> Tuple[float, float]:
    """
    Compute and save the Receiver-Operating-Characteristic (ROC) curve.

    The optimal operating point is identified via Youden's J statistic:
        J = Sensitivity + Specificity − 1 = TPR − FPR

    Args:
        labels:    True binary labels.
        probs:     Predicted probabilities for the positive class.
        save_path: File path for the saved PNG.

    Returns:
        ``(auc, optimal_threshold)`` tuple.
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    # Youden's J: maximise TPR - FPR
    j_scores       = tpr - fpr
    optimal_idx    = int(np.argmax(j_scores))
    optimal_thresh = float(thresholds[optimal_idx])

    fig, ax = plt.subplots(figsize=(7.5, 7))

    # ── ROC curve ─────────────────────────────────────────────────────────────
    ax.plot(
        fpr, tpr, color="#1565C0", lw=2.5,
        label=f"ROC Curve (AUC = {auc:.4f})",
    )

    # ── Optimal threshold point ───────────────────────────────────────────────
    ax.scatter(
        fpr[optimal_idx], tpr[optimal_idx],
        s=100, color="#D32F2F", zorder=5,
        label=(
            f"Optimal Threshold = {optimal_thresh:.3f}\n"
            f"  TPR = {tpr[optimal_idx]:.3f}  FPR = {fpr[optimal_idx]:.3f}"
        ),
    )

    # ── Diagonal (random baseline) ────────────────────────────────────────────
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier (AUC=0.5)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)",       fontsize=12)
    ax.set_title("VesselMNIST3D — ROC Curve (Test Set)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("ROC curve → %s  (AUC=%.4f, optimal_thresh=%.3f)", save_path, auc, optimal_thresh)
    return auc, optimal_thresh


# =============================================================================
# Main evaluation pipeline
# =============================================================================

def evaluate(cfg: dict, checkpoint_path: str = None) -> Dict:
    """
    Full evaluation pipeline on the test split.

    Args:
        cfg:             Configuration dictionary.
        checkpoint_path: Explicit path to a ``.pth`` checkpoint.
                         Defaults to ``outputs/checkpoints/best_model.pth``.

    Returns:
        Dictionary containing all evaluation metrics and metadata.
    """
    out            = cfg.get("output", {})
    checkpoint_dir = out.get("checkpoint_dir", "./outputs/checkpoints")
    results_dir    = out.get("results_dir",    "./outputs/results")
    log_dir        = out.get("log_dir",        "./outputs/logs")

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, log_level=out.get("log_level", "INFO"))
    device = get_device()

    # ── Load best checkpoint ──────────────────────────────────────────────────
    if checkpoint_path is None:
        checkpoint_path = str(Path(checkpoint_dir) / "best_model.pth")

    logger.info("=" * 70)
    logger.info("VESSELMNIST3D — Test Set Evaluation")
    logger.info("=" * 70)
    logger.info("Loading checkpoint: %s", checkpoint_path)

    model = build_model(cfg).to(device)
    best_epoch, ckpt_metrics = load_checkpoint(checkpoint_path, model, device=device)

    logger.info(
        "Model from epoch %d | Val AUC recorded in ckpt: %.4f",
        best_epoch,
        ckpt_metrics.get("val_auc", float("nan")),
    )

    # ── Test DataLoader ───────────────────────────────────────────────────────
    logger.info("Loading test dataset …")
    dataloaders = get_dataloaders(cfg)

    # ── Inference ─────────────────────────────────────────────────────────────
    logger.info("Running inference on %d test samples …", len(dataloaders["test"].dataset))
    raw = run_inference(model, dataloaders["test"], device)

    labels = raw["labels"]
    preds  = raw["preds"]
    probs  = raw["probs"]

    # ── Scalar metrics ────────────────────────────────────────────────────────
    accuracy = float(accuracy_score(labels, preds))
    auc      = float(roc_auc_score(labels, probs))

    logger.info("")
    logger.info("╔══════════════════════════════════════╗")
    logger.info("║     TEST SET RESULTS — VesselMNIST3D ║")
    logger.info("╠══════════════════════════════════════╣")
    logger.info("║  Accuracy : %6.4f  (%5.2f %%)        ║", accuracy, accuracy * 100)
    logger.info("║  AUC      : %6.4f                    ║", auc)
    logger.info("╚══════════════════════════════════════╝")
    logger.info("")

    cls_report = classification_report(labels, preds, target_names=CLASS_NAMES)
    logger.info("Per-class Classification Report:\n%s", cls_report)

    # ── Visualisation ─────────────────────────────────────────────────────────
    cm_path  = str(Path(results_dir) / "confusion_matrix.png")
    roc_path = str(Path(results_dir) / "roc_curve.png")

    plot_confusion_matrix(labels, preds, CLASS_NAMES, cm_path)
    final_auc, optimal_thresh = plot_roc_curve(labels, probs, roc_path)

    # ── Persist results ───────────────────────────────────────────────────────
    cls_report_dict = classification_report(
        labels, preds, target_names=CLASS_NAMES, output_dict=True
    )

    results_summary = {
        "dataset":              "VesselMNIST3D",
        "checkpoint":           checkpoint_path,
        "checkpoint_epoch":     best_epoch,
        "checkpoint_val_auc":   ckpt_metrics.get("val_auc"),
        "test_accuracy":        accuracy,
        "test_auc":             auc,
        "optimal_threshold":    optimal_thresh,
        "confusion_matrix":     confusion_matrix(labels, preds).tolist(),
        "classification_report": cls_report_dict,
        "n_test_samples":       int(len(labels)),
        "class_names":          CLASS_NAMES,
    }

    save_results(results_summary, results_dir, "test_results.json")

    logger.info("=" * 70)
    logger.info("Evaluation complete.  Outputs in: %s", results_dir)
    logger.info("  confusion_matrix.png | roc_curve.png | test_results.json")
    logger.info("=" * 70)

    return results_summary


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    """Parse CLI arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate VesselMNIST3D model on the test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Override checkpoint path (default: outputs/checkpoints/best_model.pth).",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
