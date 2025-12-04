from __future__ import annotations

from dataclasses import dataclass


def count_parameters(model) -> dict[str, int]:
    """Return total and trainable parameter counts."""

    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable}


def efficiency_metrics(accuracy: float, trainable_params: int, train_seconds: float) -> dict[str, float]:
    """Compute the requested efficiency KPIs."""

    per_million_params = accuracy / (trainable_params / 1_000_000) if trainable_params else 0.0
    minutes = train_seconds / 60.0 if train_seconds else 0.0
    per_minute = accuracy / minutes if minutes else 0.0
    return {
        "accuracy_per_million_params": per_million_params,
        "accuracy_per_minute": per_minute,
    }
