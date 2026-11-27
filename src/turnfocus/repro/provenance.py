"""Cryptographic provenance, deterministic seeding, and compute logger (Table 5, tab:app_repro)."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np


def set_deterministic_seed(seed: int) -> None:
    """Pin Python, NumPy, and PyTorch/CUDA RNG states deterministically."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_environment_provenance(extra_models: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Capture complete hardware, OS, CUDA, PyTorch, Transformers, and checkpoint revision SHAs."""
    prov: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
    }

    try:
        import scipy

        prov["scipy_version"] = scipy.__version__
    except ImportError:
        prov["scipy_version"] = "not_installed"

    try:
        import torch

        prov["torch_version"] = torch.__version__
        prov["cuda_available"] = bool(torch.cuda.is_available())
        prov["cuda_version"] = torch.version.cuda or "none"
        if torch.cuda.is_available():
            prov["gpu_count"] = torch.cuda.device_count()
            prov["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            prov["gpu_vram_gb"] = round(props.total_memory / (1024**3), 2)
    except ImportError:
        prov["torch_version"] = "not_installed"
        prov["cuda_available"] = False

    try:
        import transformers

        prov["transformers_version"] = transformers.__version__
    except ImportError:
        prov["transformers_version"] = "not_installed"

    try:
        git_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        prov["git_commit"] = git_sha
    except Exception:
        prov["git_commit"] = "standalone_archive"

    if extra_models:
        prov["pinned_model_shas"] = dict(extra_models)

    return prov


def save_provenance_json(prov: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(prov, indent=2, sort_keys=True), encoding="utf-8")
