"""
Shared logging utilities for SpineRankNet.

Provides a TensorBoard-backed training logger used across all training scripts.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple


class TrainingLogger:
    """Lightweight wrapper around TensorBoard SummaryWriter.

    Also accumulates metric history in-memory for programmatic access.
    """

    def __init__(self, logdir: str, use_tensorboard: bool = True) -> None:
        self.logdir = logdir
        self.use_tb = use_tensorboard
        if self.use_tb:
            # Imported lazily: needs the optional ``tensorboard`` package.
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(logdir, exist_ok=True)
            self.writer = SummaryWriter(logdir)
        self.metrics: Dict[str, List[Tuple[int, float]]] = {}

    def log(self, name: str, value: float, step: int) -> None:
        if self.use_tb:
            self.writer.add_scalar(name, value, step)
        self.metrics.setdefault(name, []).append((step, value))

    def close(self) -> None:
        if self.use_tb:
            self.writer.close()
