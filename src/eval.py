from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import EvalPrediction

from .utils import get_device


def compute_classification_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Return accuracy and F1 for binary SST-2 classification."""

    preds = predictions.argmax(axis=-1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": float(accuracy), "f1": float(f1)}


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    on_device = {}
    for key, value in batch.items():
        on_device[key] = value.to(device) if hasattr(value, "to") else value
    return on_device


def evaluate_model(
    model,
    dataset,
    batch_size: int,
    device: torch.device | None = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate a model over the provided dataset."""

    model.eval()
    device = device or get_device()
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size)

    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = dict(batch)
            batch.pop("teacher_logits", None)
            batch = _to_device(batch, device)
            logits = model(**batch).logits
            preds.append(logits.detach().cpu().numpy())
            label_key = "labels" if "labels" in batch else "label"
            labels.append(batch[label_key].detach().cpu().numpy())

    predictions = np.concatenate(preds, axis=0)
    label_array = np.concatenate(labels, axis=0)
    metrics = compute_classification_metrics(predictions, label_array)
    return metrics, predictions


def generate_logits(
    model,
    dataset,
    batch_size: int,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return raw logits for each example in `dataset`."""

    _, predictions = evaluate_model(model, dataset, batch_size, device=device)
    return predictions


def trainer_compute_metrics(eval_pred: EvalPrediction | Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Wrapper to plug `compute_classification_metrics` into HF Trainer."""

    if isinstance(eval_pred, EvalPrediction):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        predictions, labels = eval_pred
    return compute_classification_metrics(predictions, labels)
