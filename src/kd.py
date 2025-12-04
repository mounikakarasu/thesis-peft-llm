from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers import Trainer


class DistillationTrainer(Trainer):
    """Trainer that mixes KL divergence with the standard CE loss."""

    def __init__(self, *args, alpha: float = 0.5, temperature: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs: dict[str, Any], return_outputs: bool = False):
        labels = inputs["labels"]
        teacher_logits = inputs.pop("teacher_logits", None)
        outputs = model(**inputs)
        logits = outputs.logits

        ce_loss = outputs.loss if outputs.loss is not None else F.cross_entropy(logits, labels)
        loss = ce_loss

        if teacher_logits is not None:
            kd_loss = F.kl_div(
                F.log_softmax(logits / self.temperature, dim=-1),
                F.softmax(teacher_logits / self.temperature, dim=-1),
                reduction="batchmean",
            ) * (self.temperature**2)
            loss = self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss

        if return_outputs:
            return loss, outputs
        return loss
