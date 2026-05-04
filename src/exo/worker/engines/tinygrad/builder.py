"""TinygradBuilder: Model loading and engine construction for NVIDIA GPUs."""

import contextlib
import os
import sys
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import logger


@dataclass
class TinygradBuilder(Builder):
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    model: Any | None = None
    tokenizer: Any | None = None

    def connect(self, bound_instance: BoundInstance) -> None:
        """Initialize tinygrad device for CUDA/NV inference."""
        try:
            from tinygrad import Device
            device = Device.DEFAULT
            logger.info(f"TinygradBuilder: Connected to device {device}")
            if device not in ("NV", "CUDA", "GPU"):
                logger.warning(
                    f"TinygradBuilder: Device {device} may not be GPU-accelerated. "
                    f"Expected NV or CUDA."
                )
        except ImportError:
            raise RuntimeError(
                "tinygrad is not installed. Install with: "
                "pip install git+https://github.com/tinygrad/tinygrad.git"
            )

    def load(
        self, bound_instance: BoundInstance
    ) -> Generator[ModelLoadingResponse]:
        """Load model weights using tinygrad on NVIDIA GPU."""
        from exo.worker.engines.tinygrad.model_loader import load_tinygrad_model

        model_card = bound_instance.bound_shard.model_card
        model_id = model_card.model_id
        logger.info(f"TinygradBuilder: Loading model {model_id}")

        yield ModelLoadingResponse(
            progress=0.0,
            eta_seconds=None,
            download_percentage=None,
        )

        try:
            self.model, self.tokenizer = load_tinygrad_model(
                model_id=model_id,
                bound_instance=bound_instance,
                progress_callback=lambda p: None,
            )
            logger.info(f"TinygradBuilder: Model {model_id} loaded successfully")
        except Exception as e:
            logger.error(f"TinygradBuilder: Failed to load model: {e}")
            raise

        yield ModelLoadingResponse(
            progress=1.0,
            eta_seconds=None,
            download_percentage=None,
        )

    def build(self) -> Engine:
        """Build the TinygradEngine with loaded model."""
        assert self.model is not None, "Model not loaded"
        assert self.tokenizer is not None, "Tokenizer not loaded"

        from exo.worker.engines.tinygrad.engine import TinygradEngine

        return TinygradEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            cancel_receiver=self.cancel_receiver,
        )

    def close(self) -> None:
        """Cleanup tinygrad resources."""
        with contextlib.suppress(Exception):
            del self.model
        with contextlib.suppress(Exception):
            del self.tokenizer
        logger.info("TinygradBuilder: Closed")
