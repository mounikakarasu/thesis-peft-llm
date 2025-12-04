from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, Sequence

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerBase

DEFAULT_TEXT_COLUMN = "sentence"
KD_LOGITS_COLUMN = "teacher_logits"


def load_sst2(cache_dir: str | None = None) -> DatasetDict:
    """Load the GLUE SST-2 dataset."""

    return load_dataset("glue", "sst2", cache_dir=cache_dir)


def tokenize_text_dataset(
    dataset: Dataset | DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    text_column: str = DEFAULT_TEXT_COLUMN,
    padding: str = "max_length",
) -> Dataset | DatasetDict:
    """Tokenize every split while preserving labels."""

    def _preprocess(batch: dict) -> dict:
        return tokenizer(
            batch[text_column],
            max_length=max_length,
            truncation=True,
            padding=padding,
        )

    reference_columns = dataset["train"].column_names if isinstance(dataset, DatasetDict) else dataset.column_names
    remove_columns = [col for col in reference_columns if col == "idx"]
    return dataset.map(_preprocess, batched=True, remove_columns=remove_columns)


def sample_subset(dataset: Dataset, sample_size: int, seed: int) -> Dataset:
    """Take a deterministic subset for KD experiments."""

    if sample_size > len(dataset):
        raise ValueError("sample_size cannot exceed dataset length")
    shuffled = dataset.shuffle(seed=seed)
    return shuffled.select(range(sample_size))


def add_teacher_logits(dataset: Dataset, logits: Sequence[Sequence[float]]) -> Dataset:
    """Append teacher logits to a dataset."""

    if len(dataset) != len(logits):
        raise ValueError("Dataset and logits length mismatch.")
    listified = [list(map(float, row)) for row in logits]
    if KD_LOGITS_COLUMN in dataset.column_names:
        dataset = dataset.remove_columns(KD_LOGITS_COLUMN)
    return dataset.add_column(KD_LOGITS_COLUMN, listified)


def save_dataset(dataset: Dataset, path: str | Path) -> None:
    """Persist a dataset locally for later notebook stages."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    dataset.save_to_disk(str(target))


def load_local_dataset(path: str | Path) -> Dataset:
    """Load a dataset previously saved via `save_dataset`."""

    return load_from_disk(str(path))


def get_model_input_columns(dataset: Dataset, include_teacher_logits: bool = False) -> Iterable[str]:
    """Determine which columns should be converted to torch tensors."""

    columns: list[str] = ["input_ids", "attention_mask"]
    if "token_type_ids" in dataset.column_names:
        columns.append("token_type_ids")
    columns.append("label")
    if include_teacher_logits and KD_LOGITS_COLUMN in dataset.column_names:
        columns.append(KD_LOGITS_COLUMN)
    return columns


def format_for_torch(dataset: Dataset, include_teacher_logits: bool = False) -> Dataset:
    """Return a dataset with torch tensors ready for Trainer/DataLoader."""

    columns = list(get_model_input_columns(dataset, include_teacher_logits=include_teacher_logits))
    return dataset.with_format("torch", columns=columns)
