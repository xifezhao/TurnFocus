# Portable CUDA 12.4 container for reproducible TurnFocus execution.
# Note: The archived Colab A100 sanity-check run (results/synthetic_gpu_sanity_check/provenance.json)
# executed on Colab's managed runtime (Python 3.13.16, CUDA 13.0, torch 2.14.1+cu130, transformers 5.24.1).
# For CPU-only verification, install `requirements-cpu.txt` without CUDA.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/turnfocus
COPY . /workspace/turnfocus

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements-colab.txt && \
    pip3 install --no-cache-dir --no-deps -e .

CMD ["bash", "scripts/run_smoke_test.sh"]
