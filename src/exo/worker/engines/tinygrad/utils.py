"""Utility functions for tinygrad NVIDIA engine."""

import os
import subprocess
from typing import NamedTuple


class GPUInfo(NamedTuple):
    name: str
    memory_total_mb: int
    memory_free_mb: int
    memory_used_mb: int
    cuda_version: str
    driver_version: str
    compute_capability: str


def is_nvidia_available() -> bool:
    """Check if NVIDIA GPU is available via tinygrad."""
    try:
        from tinygrad import Device
        return Device.DEFAULT in ("NV", "CUDA", "GPU")
    except ImportError:
        return False


def get_gpu_info() -> GPUInfo | None:
    """Get NVIDIA GPU information via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        parts = result.stdout.strip().split(", ")
        if len(parts) < 5:
            return None

        # Get CUDA version
        cuda_ver = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        cuda_version = "unknown"
        for line in cuda_ver.stdout.split("\n"):
            if "release" in line.lower():
                cuda_version = line.strip().split("release ")[-1].split(",")[0]
                break

        return GPUInfo(
            name=parts[0].strip(),
            memory_total_mb=int(float(parts[1].strip())),
            memory_free_mb=int(float(parts[2].strip())),
            memory_used_mb=int(float(parts[3].strip())),
            cuda_version=cuda_version,
            driver_version=parts[4].strip(),
            compute_capability="8.6",  # RTX 3090 = SM86
        )
    except Exception:
        return None


def get_system_memory_mb() -> int:
    """Get total system RAM in MB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0


def report_device_capabilities() -> dict:
    """Report full device capabilities for topology-aware scheduling."""
    gpu = get_gpu_info()
    sys_mem = get_system_memory_mb()

    return {
        "platform": "linux",
        "backend": "tinygrad-nv",
        "gpu": {
            "name": gpu.name if gpu else "unknown",
            "vram_mb": gpu.memory_total_mb if gpu else 0,
            "cuda_version": gpu.cuda_version if gpu else "none",
            "driver_version": gpu.driver_version if gpu else "none",
            "compute_capability": gpu.compute_capability if gpu else "none",
        },
        "system": {
            "ram_mb": sys_mem,
        },
    }
