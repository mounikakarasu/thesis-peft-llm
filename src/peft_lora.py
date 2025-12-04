from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from peft import LoraConfig, TaskType


@dataclass(frozen=True)
class LoraHyperParams:
    r: int = 32
    alpha: int = 64
    dropout: float = 0.1
    target_modules: Sequence[str] = ("query", "key", "value")


def build_lora_config(
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str] | None,
    task_type: TaskType = TaskType.SEQ_CLS,
) -> LoraConfig:
    """Construct a LoRA config that adheres to our shared defaults."""

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules) if target_modules else None,
        bias="none",
        task_type=task_type,
    )


def default_lora_config() -> LoraConfig:
    """Return the thesis-specified configuration."""

    params = LoraHyperParams()
    return build_lora_config(
        r=params.r,
        alpha=params.alpha,
        dropout=params.dropout,
        target_modules=params.target_modules,
    )
