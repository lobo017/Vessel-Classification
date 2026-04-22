# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — VesselMNIST3D 3D Classification
#
# Base image: Official PyTorch + CUDA 11.8 runtime (NVIDIA GPU support).
# Falls back to CPU-only if no GPU is available when running the container.
#
# Build:  docker build -t vessel-clf .
# Run:    docker run --gpus all -v ${PWD}/data:/workspace/data \
#                               -v ${PWD}/outputs:/workspace/outputs \
#                               vessel-clf
# ─────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

LABEL maintainer="vessel-clf"
LABEL description="3D ResNet-SE for VesselMNIST3D binary classification"

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /workspace

# ── Python dependencies (cached layer) ────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy project source ────────────────────────────────────────────────────────
COPY . .

# ── Create runtime directories ─────────────────────────────────────────────────
RUN mkdir -p data \
             outputs/checkpoints \
             outputs/logs \
             outputs/results

# ── Default command: run training ─────────────────────────────────────────────
CMD ["python", "-m", "src.train", "--config", "configs/config.yaml"]
