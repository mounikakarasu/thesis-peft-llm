from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Type

from transformers import Trainer, TrainingArguments, default_data_collator

from .utils import GLOBAL_CONFIG


def build_training_arguments(
    output_dir: str,
    *,
    num_train_epochs: int,
    learning_rate: float,
    gradient_checkpointing: bool = False,
    warmup_ratio: float = 0.0,
    report_to: Optional[Sequence[str]] = None,
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> TrainingArguments:
    """Create TrainingArguments with the global defaults baked in."""

    cfg = GLOBAL_CONFIG
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "gradient_checkpointing": gradient_checkpointing,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "evaluation_strategy": cfg.evaluation_strategy,
        "save_strategy": cfg.save_strategy,
        "logging_steps": cfg.logging_steps,
        "fp16": cfg.fp16,
        "warmup_ratio": warmup_ratio,
        "report_to": list(report_to) if report_to else [],
        "load_best_model_at_end": False,
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    return TrainingArguments(**kwargs)


def create_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args: TrainingArguments,
    *,
    compute_metrics: Optional[Callable[[Any], dict[str, float]]] = None,
    trainer_cls: Type[Trainer] = Trainer,
    **trainer_kwargs: Any,
) -> Trainer:
    """Instantiate a Trainer (or subclass) with shared defaults."""

    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        **trainer_kwargs,
    )
    return trainer


def train_and_evaluate(trainer: Trainer) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the full training loop followed by evaluation."""

    train_metrics = trainer.train()
    eval_metrics = trainer.evaluate()
    return train_metrics.metrics, eval_metrics
