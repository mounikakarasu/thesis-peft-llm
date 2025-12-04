# Thesis PEFT Pipeline (Version 2)

Minimal end-to-end workflow for the SST-2 experiments described in the thesis. Everything runs on a single Colab T4 session within ~2 hours, following the five phases A-D plus the shared setup notebook.

## Repository Layout

- `environment.yml` - optional Conda environment if you are not in Colab.
- `src/` - shared Python helpers for data handling, KD, PEFT, evaluation, and training.
- `notebooks/` - Colab-ready notebooks for each experiment phase.
- `outputs/` - notebooks write KD subsets, Hugging Face `save_to_disk` artifacts, and JSON metric reports here.

## Quickstart

1. **Clone or upload** the repo to your Colab workspace.
2. **Open `notebooks/00_shared_environment.ipynb`** and run every cell. This installs dependencies (if needed), seeds, enables TF32, and downloads SST-2.
3. **Run the remaining notebooks in this order:**
   1. `A_teacher_baseline.ipynb` - evaluates the public BERT-large teacher and writes KD subsets (`outputs/kd/kd_1000`, `outputs/kd/kd_500`).
   2. `C1_distilbert_full_ft.ipynb` - full fine-tuning baseline for DistilBERT.
   3. `C2_distilbert_kd.ipynb` - DistilBERT KD with the 1k subset (requires Phase A outputs).
   4. `B_bert_large_lora.ipynb` - LoRA on BERT-large (1 epoch).
   5. `D_hybrid_lora_kd.ipynb` - LoRA + KD hybrid with 500 samples (requires Phase A outputs).

Each notebook logs dev/test accuracy + F1, train runtime, parameter counts, and efficiency metrics (`accuracy_per_million_params`, `accuracy_per_minute`) to `outputs/reports/phase_<phase>.json`.

## Evaluation & Reporting

After all notebooks finish you have five JSON artifacts:

- `phase_a_metrics.json`
- `phase_b_metrics.json`
- `phase_c1_metrics.json`
- `phase_c2_metrics.json`
- `phase_d_metrics.json`

These records contain everything needed for Chapter 4 comparisons: checkpoint names, accuracy/F1 for dev & test, trainable parameter counts, runtime, efficiency metrics, and KD subset paths. Load them into pandas or copy into your thesis tables.

## Tips

- The KD subsets are deterministic (`seed=42`). Re-running Phase A overwrites previous subsets/metrics to keep the workflow clean.
- If you change checkpoints or hyperparameters, edit the relevant cells in each notebook - all shared logic lives in `src/`.
- Keep Colab's `Runtime -> Change runtime type -> GPU` set to `T4` to match the budget assumptions (FP16 + TF32).
