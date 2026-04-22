# VesselMNIST3D — 3D ResNet-SE Classification

> **Binary classification of brain MRA vessel segments:**  
> `0` = Healthy Vessel · `1` = Aneurysm  
> Architecture: 3D ResNet-18 + Squeeze-and-Excitation attention blocks

---

## Project Structure

```
vessel_clf/
├── src/
│   ├── __init__.py       # Python package marker
│   ├── dataset.py        # MedMNIST data loading + 3D augmentations
│   ├── model.py          # 3D ResNet-18 + SE attention architecture
│   ├── utils.py          # Logging, checkpoints, early stopping, metrics
│   ├── train.py          # Training loop (AMP, LR schedule, TensorBoard)
│   └── evaluate.py       # Test evaluation + plots + JSON report
├── configs/
│   └── config.yaml       # All hyperparameters (edit here, not in code)
├── data/                 # Auto-created; MedMNIST downloads here
├── outputs/
│   ├── checkpoints/      # best_model.pth  +  last_checkpoint.pth
│   ├── logs/             # run_*.log  +  tensorboard/
│   └── results/          # confusion_matrix.png  roc_curve.png  test_results.json
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Architecture Summary

| Component | Detail |
|---|---|
| **Stem** | Conv3D(3×3×3) → BN → ReLU → MaxPool(2) — 28³ → 14³ |
| **Stage 1** | 2 × BasicBlock3D-SE (32 ch, stride=1) — 14³ → 14³ |
| **Stage 2** | 2 × BasicBlock3D-SE (64 ch, stride=2) — 14³ → 7³ |
| **Stage 3** | 2 × BasicBlock3D-SE (128 ch, stride=2) — 7³ → 4³ |
| **Stage 4** | 2 × BasicBlock3D-SE (256 ch, stride=2) — 4³ → 2³ |
| **Head** | AdaptiveAvgPool → Dropout → FC(256→128) → ReLU → FC(128→2) |
| **SE Block** | GlobalAvgPool → FC(C→C/8) → ReLU → FC(C/8→C) → Sigmoid |
| **Params** | ~2.1 M total (base_channels=32) |

---

## Quick-Start: PowerShell (Windows)

### Option A — Virtual Environment (Recommended)

```powershell
# ── 1. Navigate to your workspace and create project ─────────────────────────
cd C:\Users\<YourName>\Projects
New-Item -ItemType Directory -Name vessel_clf
cd vessel_clf

# ── 2. Copy project files here (or clone your repo) ──────────────────────────
#    Ensure all files from the zip/download are in C:\...\vessel_clf\

# ── 3. Create & activate virtual environment ─────────────────────────────────
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# If you get an execution policy error, run this first (once):
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# ── 4. Upgrade pip and install dependencies ───────────────────────────────────
python -m pip install --upgrade pip
pip install -r requirements.txt

# ── 5. Verify GPU is detected (optional) ─────────────────────────────────────
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Option B — Docker (GPU)

```powershell
# Requires Docker Desktop with WSL2 backend + NVIDIA Container Toolkit

# ── 1. Build the image ────────────────────────────────────────────────────────
docker build -t vessel-clf:latest .

# ── 2. Run training (GPU) ─────────────────────────────────────────────────────
docker run --gpus all `
  -v ${PWD}/data:/workspace/data `
  -v ${PWD}/outputs:/workspace/outputs `
  vessel-clf:latest

# ── 3. Run evaluation inside the same container ───────────────────────────────
docker run --gpus all `
  -v ${PWD}/data:/workspace/data `
  -v ${PWD}/outputs:/workspace/outputs `
  vessel-clf:latest `
  python -m src.evaluate --config configs/config.yaml
```

---

## Step-by-Step Execution

### Step 1 — Run Training

```powershell
# From the vessel_clf\ root directory with .venv activated:
python -m src.train --config configs/config.yaml
```

Expected console output:
```
2025-04-01 12:00:00 | INFO     | VESSELMNIST3D — 3D ResNet-SE Training
2025-04-01 12:00:00 | INFO     | Loading datasets …
2025-04-01 12:00:05 | INFO     | ResNet3D-SE built | base_ch=32 | SE=True | total params: 2,123,458
2025-04-01 12:00:05 | INFO     | Starting training — 100 epochs on cuda
...
2025-04-01 12:02:10 | INFO     | Ep[001/100]  lr=2.00e-04  Train loss=0.6123 acc=0.6521 auc=0.6834  Val loss=0.5982 acc=0.6910 auc=0.7213  (24s)
2025-04-01 12:04:15 | INFO     | ★  New best model → outputs\checkpoints\best_model.pth  (epoch=2, val_auc=0.7541)
...
```

### Step 2 — Monitor with TensorBoard (optional, run in a second terminal)

```powershell
# Keep training running in terminal 1; open terminal 2:
.\.venv\Scripts\Activate.ps1
tensorboard --logdir outputs/logs/tensorboard
# Open http://localhost:6006 in your browser
```

### Step 3 — Run Evaluation

```powershell
python -m src.evaluate --config configs/config.yaml
# Optionally specify a checkpoint explicitly:
python -m src.evaluate --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pth
```

Expected output:
```
╔══════════════════════════════════════╗
║     TEST SET RESULTS — VesselMNIST3D ║
╠══════════════════════════════════════╣
║  Accuracy :  0.8220  ( 82.20 %)      ║
║  AUC      :  0.8897                  ║
╚══════════════════════════════════════╝

Per-class Classification Report:
                 precision    recall  f1-score   support
 Healthy Vessel       0.84      0.86      0.85       240
       Aneurysm       0.79      0.76      0.77       142
       accuracy                           0.82       382
```

### Step 4 — View Results

```powershell
# Open result images
Start-Process outputs\results\confusion_matrix.png
Start-Process outputs\results\roc_curve.png

# Inspect metrics JSON
Get-Content outputs\results\test_results.json
```

---

## Configuration Tuning

All hyperparameters live in `configs/config.yaml`. Key knobs:

| Parameter | Default | Effect |
|---|---|---|
| `training.batch_size` | 32 | Reduce to 16 if GPU OOM |
| `training.learning_rate` | 0.001 | Peak LR after warmup |
| `training.num_epochs` | 100 | Max epochs (early stopping may cut short) |
| `training.early_stopping_patience` | 15 | Patience in epochs |
| `model.base_channels` | 32 | Increase to 64 for more capacity |
| `model.dropout_rate` | 0.3 | Reduce if underfitting |
| `data.image_size` | 28 | Set to 64 for MedMNIST+ (higher res) |
| `data.num_workers` | 4 | Set to 0 on Windows if DataLoader errors |

---

## Windows-Specific Notes

1. **`num_workers: 0`** — If you see `BrokenPipeError` or freezing DataLoaders,
   set `num_workers: 0` in `config.yaml`.  Multi-process DataLoaders on Windows
   require that all code runs inside `if __name__ == '__main__':`.

2. **Execution Policy** — If `.venv\Scripts\Activate.ps1` is blocked:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **CUDA** — Install [PyTorch with CUDA](https://pytorch.org/get-started/locally/)
   matching your driver version.  The `requirements.txt` installs the CPU build
   by default; for GPU replace the `torch` lines with the CUDA wheel:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

---

## Expected Performance (VesselMNIST3D, 28³)

| Metric | Typical Range |
|---|---|
| Test Accuracy | 80 – 87 % |
| Test AUC | 0.87 – 0.93 |

Training time: ~30 min on a single RTX 3060 / ~90 min on CPU (reduce epochs to 30 for CPU testing).

---

## References

- Yang et al. (2023) — *MedMNIST v2*: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification
- He et al. (2015) — *Deep Residual Learning for Image Recognition*
- Hu et al. (2018) — *Squeeze-and-Excitation Networks*
