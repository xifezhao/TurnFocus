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
    pip3 install --no-cache-dir -e ".[gpu,dev]"

CMD ["bash", "scripts/run_smoke_test.sh"]
