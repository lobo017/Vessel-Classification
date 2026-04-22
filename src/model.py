"""
src/model.py — 3-D ResNet-18 with Squeeze-and-Excitation Attention
===================================================================
Architecture overview
---------------------
Input: (B, 1, 28, 28, 28)  — batch × channel × depth × height × width

  Stem          3×3×3 Conv3D → BN → ReLU → MaxPool3D(2)      [ 28 →  14 ]
  Stage 1       2 × BasicBlock3D_SE  (32 ch,  stride=1)       [ 14 →  14 ]
  Stage 2       2 × BasicBlock3D_SE  (64 ch,  stride=2)       [ 14 →   7 ]
  Stage 3       2 × BasicBlock3D_SE  (128 ch, stride=2)       [  7 →   4 ]
  Stage 4       2 × BasicBlock3D_SE  (256 ch, stride=2)       [  4 →   2 ]
  Head          AdaptiveAvgPool3D(1) → Flatten → Dropout
                → Linear(256→128) → ReLU → Dropout → Linear(128→2)

SE block (per stage):
  GlobalAvgPool → FC(C→C/r) → ReLU → FC(C/r→C) → Sigmoid
  Output = input × sigmoid_weights   (channel-wise recalibration)

Design rationale
----------------
• ResNet-18 depth: eight residual blocks give enough capacity for 28³ volumes
  without over-fitting; deeper nets (ResNet-50) would require regularisation
  not justified by dataset size (~1 300 training samples).
• Small stem (3×3 not 7×7): a 7×7 stem tailored for 224² ImageNet inputs
  would collapse a 28³ volume to ~3³ before any residual learning.
• SE attention: adds < 1 % extra parameters but improves AUC ~1-2 % on
  volumetric medical imaging tasks (verified in literature).
• Kaiming initialisation: appropriate for ReLU-activated networks.

References
----------
He et al. (2015) — Deep Residual Learning for Image Recognition
Hu  et al. (2018) — Squeeze-and-Excitation Networks
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Squeeze-and-Excitation Block (3-D)
# =============================================================================

class SEBlock3D(nn.Module):
    """
    3-D Squeeze-and-Excitation channel-attention block.

    Learns a per-channel scaling vector from global context:
        squeeze  : global average pooling  (B, C, D, H, W) → (B, C)
        excite   : two-layer MLP with reduction bottleneck
        scale    : element-wise multiply input feature map by sigmoid output

    Args:
        channels:  Number of input (and output) channels C.
        reduction: Bottleneck reduction ratio r.  Bottleneck width = max(C/r, 1).
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()

        bottleneck = max(channels // reduction, 1)

        self.gap = nn.AdaptiveAvgPool3d(1)   # → (B, C, 1, 1, 1)
        self.fc1 = nn.Linear(channels, bottleneck, bias=False)
        self.fc2 = nn.Linear(bottleneck, channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        b, c = x.size(0), x.size(1)

        # Squeeze — global context descriptor
        s = self.gap(x).view(b, c)            # (B, C)

        # Excitation — channel importance weights
        s = self.relu(self.fc1(s))            # (B, C/r)
        s = self.sigmoid(self.fc2(s))         # (B, C)

        # Scale — broadcast over D, H, W
        return x * s.view(b, c, 1, 1, 1)


# =============================================================================
# Basic Residual Block (3-D)
# =============================================================================

class BasicBlock3D(nn.Module):
    """
    3-D Basic Residual Block with optional SE channel attention.

    Structure (pre-activation variant with standard BN placement):

        x → Conv3D(3) → BN → ReLU → [Dropout3D] →
            Conv3D(3) → BN → [SE] → (+shortcut(x)) → ReLU

    The shortcut is a 1×1×1 projection when channels or stride differ.

    Args:
        in_channels:   Input feature-map channels.
        out_channels:  Output feature-map channels.
        stride:        Spatial stride applied in the *first* conv (1 or 2).
        use_se:        Include SE block after second conv.
        se_reduction:  SE bottleneck reduction ratio.
        dropout_rate:  3-D spatial dropout probability (0 = disabled).
    """

    expansion: int = 1  # Channel expansion factor (no bottleneck for BasicBlock)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_se: bool = True,
        se_reduction: int = 8,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        # ── Main branch ───────────────────────────────────────────────────────
        self.conv1 = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Spatial dropout drops entire feature-map channels (better than
        # element-wise dropout for convolutional networks)
        self.dropout: Optional[nn.Module] = (
            nn.Dropout3d(p=dropout_rate) if dropout_rate > 0.0 else None
        )

        self.conv2 = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)

        # ── SE channel attention ──────────────────────────────────────────────
        self.se: Optional[nn.Module] = (
            SEBlock3D(out_channels, reduction=se_reduction) if use_se else None
        )

        # ── Shortcut / skip connection ─────────────────────────────────────────
        # Identity shortcut unless we need to match channel count or stride
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        # First convolution
        out = self.relu(self.bn1(self.conv1(x)))

        # Optional spatial dropout (regularisation during training)
        if self.dropout is not None:
            out = self.dropout(out)

        # Second convolution
        out = self.bn2(self.conv2(out))

        # Channel-attention rescaling
        if self.se is not None:
            out = self.se(out)

        # Residual addition then activation
        out = self.relu(out + identity)

        return out


# =============================================================================
# ResNet3D-SE (full network)
# =============================================================================

class ResNet3D_SE(nn.Module):
    """
    3-D ResNet-18 with Squeeze-and-Excitation for volumetric binary classification.

    Spatial resolution trace (28³ input, base_channels=32):
        Input   : (B,   1, 28, 28, 28)
        Stem    : (B,  32, 14, 14, 14)   ← 3×3 conv + MaxPool(2)
        Stage 1 : (B,  32, 14, 14, 14)   ← 2 × BasicBlock (stride=1)
        Stage 2 : (B,  64,  7,  7,  7)   ← 2 × BasicBlock (stride=2)
        Stage 3 : (B, 128,  4,  4,  4)   ← 2 × BasicBlock (stride=2)
        Stage 4 : (B, 256,  2,  2,  2)   ← 2 × BasicBlock (stride=2)
        Head    : (B, 256,  1,  1,  1)   ← AdaptiveAvgPool3d(1)
                  (B, 256)               ← Flatten
                  (B,   2)               ← FC → FC (logits)

    Args:
        in_channels:    Input channels (1 for single-channel 3-D MRI).
        num_classes:    Number of output classes (2 for binary).
        base_channels:  Channel count at stage 1 (doubles each stage).
        layers:         Blocks per stage — [2, 2, 2, 2] = ResNet-18 config.
        se_reduction:   SE reduction ratio.
        dropout_rate:   Dropout rate for both spatial dropout and head dropout.
        use_se:         Toggle SE attention blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 32,
        layers: Optional[List[int]] = None,
        se_reduction: int = 8,
        dropout_rate: float = 0.3,
        use_se: bool = True,
    ) -> None:
        super().__init__()

        if layers is None:
            layers = [2, 2, 2, 2]   # ResNet-18 configuration

        # Track current channel count as we build the network
        self._current_ch = base_channels

        # ── Stem ──────────────────────────────────────────────────────────────
        # 3×3 kernel (not 7×7) because input volumes are only 28³.
        # Single MaxPool(2) halves resolution: 28 → 14.
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels, base_channels,
                kernel_size=3, stride=1, padding=1, bias=False,
            ),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 28 → 14
        )

        # ── Residual stages ───────────────────────────────────────────────────
        self.layer1 = self._make_stage(
            out_ch=base_channels,           # 32
            num_blocks=layers[0],
            stride=1,                       # No further downsampling
            use_se=use_se,
            se_reduction=se_reduction,
            dropout_rate=dropout_rate,
        )
        self.layer2 = self._make_stage(
            out_ch=base_channels * 2,       # 64
            num_blocks=layers[1],
            stride=2,                       # 14 → 7
            use_se=use_se,
            se_reduction=se_reduction,
            dropout_rate=dropout_rate,
        )
        self.layer3 = self._make_stage(
            out_ch=base_channels * 4,       # 128
            num_blocks=layers[2],
            stride=2,                       # 7 → 4
            use_se=use_se,
            se_reduction=se_reduction,
            dropout_rate=dropout_rate,
        )
        self.layer4 = self._make_stage(
            out_ch=base_channels * 8,       # 256
            num_blocks=layers[3],
            stride=2,                       # 4 → 2
            use_se=use_se,
            se_reduction=se_reduction,
            dropout_rate=dropout_rate,
        )

        # ── Classification head ───────────────────────────────────────────────
        final_ch = base_channels * 8  # 256

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),     # → (B, 256, 1, 1, 1)
            nn.Flatten(),                # → (B, 256)
            nn.Dropout(p=dropout_rate),
            nn.Linear(final_ch, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(128, num_classes), # → (B, 2)  — logits
        )

        # ── Weight initialisation ─────────────────────────────────────────────
        self._init_weights()

        # ── Log model statistics ──────────────────────────────────────────────
        n_total = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "ResNet3D-SE built | base_ch=%d | SE=%s | total params: %s | trainable: %s",
            base_channels,
            use_se,
            f"{n_total:,}",
            f"{n_trainable:,}",
        )

    # ------------------------------------------------------------------
    def _make_stage(
        self,
        out_ch: int,
        num_blocks: int,
        stride: int,
        use_se: bool,
        se_reduction: int,
        dropout_rate: float,
    ) -> nn.Sequential:
        """Assemble one residual stage from BasicBlock3D units."""
        blocks: List[nn.Module] = []

        # First block: handles stride (downsampling) and channel projection
        blocks.append(
            BasicBlock3D(
                self._current_ch, out_ch,
                stride=stride,
                use_se=use_se,
                se_reduction=se_reduction,
                dropout_rate=dropout_rate,
            )
        )
        self._current_ch = out_ch

        # Subsequent blocks: same channel count, stride=1
        for _ in range(1, num_blocks):
            blocks.append(
                BasicBlock3D(
                    out_ch, out_ch,
                    stride=1,
                    use_se=use_se,
                    se_reduction=se_reduction,
                    dropout_rate=dropout_rate,
                )
            )

        return nn.Sequential(*blocks)

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """
        Apply standard He (Kaiming) initialisation for Conv3d layers,
        constant initialisation for BN, and Xavier for Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the full network.

        Args:
            x: Float tensor of shape (B, 1, D, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).  Apply softmax
            externally if you need probabilities.
        """
        x = self.stem(x)        # (B,  32, 14, 14, 14)
        x = self.layer1(x)      # (B,  32, 14, 14, 14)
        x = self.layer2(x)      # (B,  64,  7,  7,  7)
        x = self.layer3(x)      # (B, 128,  4,  4,  4)
        x = self.layer4(x)      # (B, 256,  2,  2,  2)
        x = self.classifier(x)  # (B, num_classes)
        return x


# =============================================================================
# Factory function
# =============================================================================

def build_model(cfg: Dict) -> ResNet3D_SE:
    """
    Construct a ``ResNet3D_SE`` instance from the configuration dictionary.

    Args:
        cfg: Full configuration dict (loaded from ``config.yaml``).

    Returns:
        Initialised, untrained ``ResNet3D_SE``.
    """
    m = cfg.get("model", {})
    model = ResNet3D_SE(
        in_channels=int(m.get("in_channels", 1)),
        num_classes=int(m.get("num_classes", 2)),
        base_channels=int(m.get("base_channels", 32)),
        se_reduction=int(m.get("se_reduction", 8)),
        dropout_rate=float(m.get("dropout_rate", 0.3)),
        use_se=bool(m.get("use_se", True)),
    )
    return model
