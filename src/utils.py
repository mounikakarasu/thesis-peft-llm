from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator

import numpy as np
import torch


@dataclass(frozen=True)
class GlobalTrainingConfig:
    """Holds the shared defaults specified for every experiment."""

    seed: int = 42
    max_length: int = 96
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    tf32: bool = True
    evaluation_strategy: str = "epoch"
    save_strategy: str = "no"
    logging_steps: int = 50


GLOBAL_CONFIG = GlobalTrainingConfig()


def set_seed_everywhere(seed: int) -> None:
    """Seed python, numpy, and torch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_tf32(enable: bool = True) -> None:
    """Enable or disable TF32 where the runtime supports it."""

    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = enable
    torch.backends.cudnn.allow_tf32 = enable


def get_device() -> torch.device:
    """Pick the best available compute device."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create a directory (and parents) if needed."""

    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


@contextmanager
def track_time() -> Iterator[Callable[[], float]]:
    """Simple timer that returns an elapsed-time callable."""

    start_time = time.perf_counter()

    def _elapsed() -> float:
        return time.perf_counter() - start_time

    yield _elapsed


def write_json(data: Dict[str, Any], path: str | os.PathLike[str]) -> None:
    """Persist a dictionary as nicely formatted JSON."""

    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def read_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load a JSON dictionary if it exists, otherwise raise."""

    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "GLOBAL_CONFIG",
    "GlobalTrainingConfig",
    "configure_tf32",
    "ensure_dir",
    "get_device",
    "read_json",
    "set_seed_everywhere",
    "track_time",
    "write_json",
]
