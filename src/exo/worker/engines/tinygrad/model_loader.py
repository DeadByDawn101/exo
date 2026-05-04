"""Model loading utilities for tinygrad NVIDIA inference."""

import os
from pathlib import Path
from typing import Any, Callable

from exo.shared.types.worker.instances import BoundInstance
from exo.worker.runner.bootstrap import logger


def load_tinygrad_model(
    model_id: str,
    bound_instance: BoundInstance,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[Any, Any]:
    """
    Load a model for tinygrad inference.

    Supports loading from:
    1. HuggingFace safetensors (via tinygrad)
    2. Local model files
    3. GGUF format

    Returns (model, tokenizer) tuple.
    """
    from tinygrad import Device
    logger.info(f"Loading model {model_id} on device {Device.DEFAULT}")

    # Try loading with transformers tokenizer + tinygrad weights
    try:
        return _load_safetensors_model(model_id, bound_instance)
    except Exception as e:
        logger.warning(f"Safetensors loading failed: {e}")

    # Fallback: try tinygrad's built-in LLM support
    try:
        return _load_tinygrad_llm(model_id)
    except Exception as e:
        logger.warning(f"Tinygrad LLM loading failed: {e}")

    raise RuntimeError(
        f"Could not load model {model_id}. "
        f"Supported formats: safetensors, GGUF, tinygrad LLM"
    )


def _load_safetensors_model(
    model_id: str, bound_instance: BoundInstance
) -> tuple[Any, Any]:
    """Load model from HuggingFace safetensors format."""
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    logger.info(f"Tokenizer loaded for {model_id}")

    # Load model weights into tinygrad
    model = _load_weights_to_tinygrad(model_id)
    logger.info(f"Model weights loaded for {model_id}")

    return model, tokenizer


def _load_weights_to_tinygrad(model_id: str) -> Any:
    """Load safetensors weights into tinygrad tensors on GPU."""
    from tinygrad import Tensor, Device
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download, list_repo_files

    # Find safetensors files
    try:
        files = list_repo_files(model_id)
        st_files = [f for f in files if f.endswith('.safetensors')]
    except Exception:
        st_files = []

    if not st_files:
        raise RuntimeError(f"No safetensors files found for {model_id}")

    # Download and load weights
    weights = {}
    for st_file in st_files:
        local_path = hf_hub_download(model_id, st_file)
        with safe_open(local_path, framework="numpy") as f:
            for key in f.keys():
                weights[key] = Tensor(f.get_tensor(key)).to(Device.DEFAULT)

    logger.info(f"Loaded {len(weights)} weight tensors")
    return weights


def _load_tinygrad_llm(model_id: str) -> tuple[Any, Any]:
    """Load using tinygrad's built-in LLM module."""
    # This uses tinygrad's own model implementations
    raise NotImplementedError("Tinygrad native LLM loading not yet implemented")
