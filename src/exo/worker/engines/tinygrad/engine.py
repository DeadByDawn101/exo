"""TinygradEngine: NVIDIA GPU inference engine for exo."""

import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, BinaryIO

from exo.shared.types.chunks import Chunk
from exo.shared.types.tasks import GenerationTask, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
)
from exo.utils.channels import MpReceiver
from exo.worker.engines.base import Engine
from exo.worker.runner.bootstrap import logger
from exo.worker.disaggregated.server import PrefillRequest


@dataclass
class TinygradEngine(Engine):
    model: Any
    tokenizer: Any
    cancel_receiver: MpReceiver[TaskId]
    _cancelled_tasks: set[TaskId] = field(default_factory=set)
    _task_queue: deque = field(default_factory=deque)
    _active_tasks: dict = field(default_factory=dict)

    def warmup(self) -> None:
        """Warmup the GPU with a dummy forward pass."""
        from tinygrad import Tensor, Device
        logger.info(f"TinygradEngine: Warming up on {Device.DEFAULT}")
        # Small matmul to warm up GPU pipelines
        x = Tensor.rand(64, 64)
        _ = (x @ x).realize()
        logger.info("TinygradEngine: Warmup complete")

    def submit(self, task: GenerationTask) -> None:
        """Submit a generation task to the queue."""
        logger.debug(f"TinygradEngine: Submitted task {task.task_id}")
        self._task_queue.append(task)

    def step(
        self,
    ) -> Iterable[tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]]:
        """Process one step of inference and yield results."""
        # Check for cancellations
        try:
            while True:
                cancelled_id = self.cancel_receiver.recv_nowait()
                self._cancelled_tasks.add(cancelled_id)
        except Exception:
            pass

        results = []

        if not self._task_queue:
            return results

        task = self._task_queue.popleft()

        if self.should_cancel(task.task_id):
            self._cancelled_tasks.discard(task.task_id)
            results.append((task.task_id, CancelledResponse()))
            return results

        try:
            # Run inference
            output_tokens = self._generate(task)

            if output_tokens is not None:
                chunk = Chunk(
                    tokens=output_tokens,
                    is_finished=False,
                )
                results.append((task.task_id, chunk))
        except Exception as e:
            logger.error(f"TinygradEngine: Error in step: {e}")
            results.append((task.task_id, FinishedResponse(
                finish_reason="error",
            )))

        return results

    def _generate(self, task: GenerationTask) -> list[int] | None:
        """Run tinygrad inference for a single generation step."""
        from tinygrad import Tensor
        import numpy as np

        # This is the core inference path
        # For now, use a simple greedy decode step
        # TODO: Full model-specific inference with KV cache
        try:
            input_ids = task.prompt_tokens if hasattr(task, 'prompt_tokens') else []
            if not input_ids:
                return None

            # Convert to tinygrad tensor
            input_tensor = Tensor(input_ids).reshape(1, -1)

            # Forward pass through model
            logits = self.model(input_tensor)

            # Greedy decode: take argmax of last position
            if hasattr(logits, 'numpy'):
                next_token = int(logits[:, -1, :].argmax(axis=-1).numpy()[0])
            else:
                next_token = int(np.argmax(logits[:, -1, :]))

            return [next_token]

        except Exception as e:
            logger.error(f"TinygradEngine: Generation error: {e}")
            return None

    def close(self) -> None:
        """Cleanup engine resources."""
        self._task_queue.clear()
        self._active_tasks.clear()
        self._cancelled_tasks.clear()
        logger.info("TinygradEngine: Closed")

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        """Handle prefill requests for disaggregated inference."""
        # TODO: Implement prefill serving for distributed inference
        logger.warning("TinygradEngine: serve_prefill not yet implemented")
