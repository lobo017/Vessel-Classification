"""
src/dataset.py — 3D Data Pipeline for VesselMNIST3D
=====================================================
Handles:
  • Downloading / loading the VesselMNIST3D MedMNIST dataset
  • Per-volume normalisation to [0, 1]
  • 3D augmentation transforms (flips, rotations, intensity shifts, noise)
  • PyTorch Dataset / DataLoader construction

VesselMNIST3D task:
  - Binary classification: 0 = Healthy vessel, 1 = Aneurysm
  - Volume shape: (28, 28, 28) single-channel grayscale
  - Split sizes (approx): train≈1335, val≈191, test≈382
"""

import logging
from typing import Any, Dict, Optional, Tuple

import medmnist
import numpy as np
import scipy.ndimage
import torch
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# 3-D Augmentation Primitives
# =============================================================================

class RandomFlip3D:
    """
    Randomly flip the 3-D volume independently along each spatial axis.

    Args:
        p: Probability of flipping along each individual axis.
           With three axes and p=0.5 there are 2^3=8 equally-likely outcomes.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        # Iterate over depth (0), height (1), width (2)
        for axis in range(3):
            if np.random.random() < self.p:
                volume = np.flip(volume, axis=axis).copy()
        return volume


class RandomRotate3D:
    """
    Randomly rotate the 3-D volume in all three orthogonal planes.

    Applies three independent planar rotations:
        (D, H) — axial          |  (D, W) — coronal  |  (H, W) — sagittal

    Uses scipy.ndimage.rotate with bilinear (order=1) interpolation and
    constant padding to avoid boundary artefacts.

    Args:
        max_degrees: Maximum absolute rotation angle in degrees.
        p:           Probability of applying the full rotation.
    """

    def __init__(self, max_degrees: float = 15.0, p: float = 0.5) -> None:
        self.max_degrees = max_degrees
        self.p = p

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return volume

        angle_dh = np.random.uniform(-self.max_degrees, self.max_degrees)
        angle_dw = np.random.uniform(-self.max_degrees, self.max_degrees)
        angle_hw = np.random.uniform(-self.max_degrees, self.max_degrees)

        # reshape=False keeps the output the same spatial size
        # cval=0.0 fills new voxels with background value
        volume = scipy.ndimage.rotate(
            volume, angle_dh, axes=(0, 1), reshape=False, order=1, cval=0.0
        )
        volume = scipy.ndimage.rotate(
            volume, angle_dw, axes=(0, 2), reshape=False, order=1, cval=0.0
        )
        volume = scipy.ndimage.rotate(
            volume, angle_hw, axes=(1, 2), reshape=False, order=1, cval=0.0
        )
        return volume


class RandomIntensityShift:
    """
    Independently shift and scale voxel intensities.

    Simulates scanner-to-scanner variability and minor exposure differences.
    Both operations are applied when triggered (single probability check).

    Args:
        shift_range: Additive offset sampled from U(-shift_range, +shift_range).
        scale_range: Multiplicative factor sampled from U(1-scale_range, 1+scale_range).
        p:           Probability of applying the transform.
    """

    def __init__(
        self,
        shift_range: float = 0.1,
        scale_range: float = 0.1,
        p: float = 0.5,
    ) -> None:
        self.shift_range = shift_range
        self.scale_range = scale_range
        self.p = p

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            shift = np.random.uniform(-self.shift_range, self.shift_range)
            scale = np.random.uniform(1.0 - self.scale_range, 1.0 + self.scale_range)
            volume = volume * scale + shift
        return volume


class GaussianNoise3D:
    """
    Inject zero-mean Gaussian noise into the volume.

    Models acquisition noise and acts as a regulariser.

    Args:
        std: Standard deviation of the noise distribution.
        p:   Probability of applying the transform.
    """

    def __init__(self, std: float = 0.02, p: float = 0.3) -> None:
        self.std = std
        self.p = p

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            noise = np.random.normal(0.0, self.std, volume.shape).astype(np.float32)
            volume = volume + noise
        return volume


class Compose3D:
    """
    Sequentially apply a list of 3-D transforms to a numpy volume.

    Args:
        transforms: Ordered list of callable transforms.
    """

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            volume = t(volume)
        return volume


# =============================================================================
# Dataset
# =============================================================================

class VesselMNIST3DDataset(Dataset):
    """
    PyTorch Dataset wrapper for the VesselMNIST3D MedMNIST split.

    Responsibilities:
      1. Downloads & caches the dataset via the official ``medmnist`` package.
      2. Normalises each volume to [0, 1] using per-volume min–max scaling.
      3. Optionally applies the supplied 3-D augmentation pipeline.
      4. Returns tensors of shape (1, D, H, W) — single-channel, channel-first —
         which is the format expected by PyTorch Conv3d layers.

    Note on ``as_rgb``:
        The ``as_rgb`` parameter applies only to *2-D* MedMNIST datasets and
        forces channel replication so greyscale images look like RGB.  For 3-D
        volumetric datasets (VesselMNIST3D) this parameter is irrelevant and
        must NOT be used, because the data is already a proper 3-D array.

    Args:
        split:     One of ``'train'``, ``'val'``, or ``'test'``.
        transform: Optional ``Compose3D`` (or any callable) to augment volumes.
        download:  Whether to auto-download if the dataset is not cached.
        data_root: Local directory where downloaded files are stored.
        size:      Spatial resolution — ``28`` (default) or ``64`` (MedMNIST+).
    """

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Any] = None,
        download: bool = True,
        data_root: str = "./data",
        size: int = 28,
    ) -> None:
        super().__init__()

        self.split = split
        self.transform = transform

        # ── Retrieve dataset metadata from MedMNIST registry ─────────────────
        info = INFO["vesselmnist3d"]
        DataClass = getattr(medmnist, info["python_class"])

        logger.info(
            "Loading VesselMNIST3D [%s] — %d samples | task: %s | labels: %s",
            split,
            info["n_samples"][split],
            info["task"],
            info["label"],
        )

        # ── Instantiate the MedMNIST dataset object ───────────────────────────
        # We deliberately pass transform=None here so that MedMNIST does NOT
        # apply its internal PIL-based transforms (which are designed for 2-D
        # images).  All volumetric augmentation is handled in __getitem__.
        self._medmnist_ds = DataClass(
            split=split,
            transform=None,     # 3-D: we handle transforms manually
            download=download,
            root=data_root,
            size=size,
        )

        # ── Cache raw arrays in memory for fast access ────────────────────────
        # .imgs  → numpy array (N, D, H, W)  — uint8 [0, 255]
        # .labels → numpy array (N, 1)        — int {0, 1}
        self.images: np.ndarray = self._medmnist_ds.imgs    # (N, D, H, W)
        self.labels: np.ndarray = self._medmnist_ds.labels  # (N, 1)

        logger.info(
            "  images: %s  dtype: %s | labels: %s",
            self.images.shape,
            self.images.dtype,
            self.labels.shape,
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.images)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        volume_tensor : torch.FloatTensor  — shape (1, D, H, W)
        label_tensor  : torch.LongTensor   — scalar {0, 1}
        """
        # ── 1. Fetch raw volume ───────────────────────────────────────────────
        volume: np.ndarray = self.images[idx].astype(np.float32)  # (D, H, W)

        # Handle the rare case where MedMNIST stores (D, H, W, 1)
        if volume.ndim == 4:
            volume = volume[..., 0]

        # ── 2. Per-volume min–max normalisation → [0, 1] ─────────────────────
        v_min, v_max = volume.min(), volume.max()
        if v_max > v_min:
            volume = (volume - v_min) / (v_max - v_min)
        else:
            # Constant volume — leave as zeros (edge case)
            volume = np.zeros_like(volume)

        # ── 3. Apply 3-D augmentation transforms (training only) ─────────────
        if self.transform is not None:
            volume = self.transform(volume)

        # ── 4. Safety clip after augmentation ────────────────────────────────
        volume = np.clip(volume, 0.0, 1.0)

        # ── 5. Add channel dimension: (D, H, W) → (1, D, H, W) ──────────────
        volume = np.expand_dims(volume, axis=0)

        # ── 6. Convert to PyTorch tensors ─────────────────────────────────────
        volume_tensor = torch.from_numpy(volume.copy()).float()
        label_tensor = torch.tensor(self.labels[idx].item(), dtype=torch.long)

        return volume_tensor, label_tensor


# =============================================================================
# Transform Factory
# =============================================================================

def get_transforms(split: str, cfg: Dict[str, Any]) -> Optional[Compose3D]:
    """
    Build the augmentation pipeline for a given data split.

    Only the training split receives augmentations; validation and test
    splits use pure min–max normalisation (applied in ``__getitem__``).

    Args:
        split: ``'train'``, ``'val'``, or ``'test'``.
        cfg:   Full configuration dictionary.

    Returns:
        A ``Compose3D`` pipeline for training, or ``None`` for val/test.
    """
    if split != "train":
        return None

    aug = cfg.get("augmentation", {})
    pipeline: list = []

    if aug.get("random_flip", True):
        pipeline.append(RandomFlip3D(p=0.5))

    if aug.get("random_rotate", True):
        pipeline.append(
            RandomRotate3D(
                max_degrees=float(aug.get("random_rotate_degrees", 15.0)),
                p=float(aug.get("random_rotate_prob", 0.5)),
            )
        )

    if aug.get("intensity_shift", 0.0) > 0:
        pipeline.append(
            RandomIntensityShift(
                shift_range=float(aug.get("intensity_shift", 0.1)),
                scale_range=float(aug.get("intensity_scale", 0.1)),
                p=float(aug.get("intensity_prob", 0.5)),
            )
        )

    if aug.get("gaussian_noise_std", 0.0) > 0:
        pipeline.append(
            GaussianNoise3D(
                std=float(aug.get("gaussian_noise_std", 0.02)),
                p=float(aug.get("gaussian_noise_prob", 0.3)),
            )
        )

    logger.info(
        "Training augmentations: [%s]",
        ", ".join(type(t).__name__ for t in pipeline),
    )

    return Compose3D(pipeline) if pipeline else None


# =============================================================================
# DataLoader Factory
# =============================================================================

def get_dataloaders(cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Construct train / val / test ``DataLoader`` objects for VesselMNIST3D.

    Args:
        cfg: Full configuration dictionary (from ``config.yaml``).

    Returns:
        Dictionary mapping split name → ``DataLoader``.
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    num_workers: int = int(data_cfg.get("num_workers", 4))
    # persistent_workers requires num_workers > 0
    use_persistent = num_workers > 0

    loaders: Dict[str, DataLoader] = {}

    for split in ("train", "val", "test"):
        transform = get_transforms(split, cfg)

        dataset = VesselMNIST3DDataset(
            split=split,
            transform=transform,
            download=bool(data_cfg.get("download", True)),
            data_root=str(data_cfg.get("data_root", "./data")),
            size=int(data_cfg.get("image_size", 28)),
        )

        is_train = split == "train"

        loaders[split] = DataLoader(
            dataset,
            batch_size=int(train_cfg.get("batch_size", 32)),
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=False,
            drop_last=is_train,          # Avoid incomplete final batch in training
            persistent_workers=use_persistent,
        )

        logger.info(
            "%s DataLoader — %d samples | %d batches | augment=%s",
            split.capitalize(),
            len(dataset),
            len(loaders[split]),
            transform is not None,
        )

    return loaders
