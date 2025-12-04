from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from peft import PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .peft_lora import build_lora_config

TEACHER_CHECKPOINTS: Sequence[str] = (
    "textattack/bert-large-uncased-SST-2",
    "philschmid/bert-large-uncased-sst2",
    "yoshitomo-matsubara/bert-large-uncased-sst2",
)


def load_model_and_tokenizer(model_name: str, num_labels: int = 2) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a classification model/tokenizer pair."""

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer


def apply_lora(
    model: PreTrainedModel,
    target_modules: Sequence[str] | None = None,
    r: int = 32,
    alpha: int = 64,
    dropout: float = 0.1,
) -> PeftModel:
    """Attach LoRA adapters to the supplied transformer."""

    lora_config = build_lora_config(
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules or (),
        task_type=TaskType.SEQ_CLS,
    )
    return get_peft_model(model, lora_config)


def enable_gradient_checkpointing(model: PreTrainedModel) -> None:
    """Enable gradient checkpointing when the architecture supports it."""

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


def enable_input_require_grads(model: PreTrainedModel) -> None:
    """Required when combining gradient checkpointing with PEFT."""

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def unfreeze_last_bert_layers(model: PreTrainedModel, num_layers: int = 2) -> None:
    """Allow the last few encoder layers to update alongside LoRA adapters."""

    encoder_layers = getattr(getattr(model, "bert", None), "encoder", None)
    if encoder_layers is None or not hasattr(encoder_layers, "layer"):
        return

    for layer in encoder_layers.layer[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
