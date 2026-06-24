# Job Posting → JSON: a hands-on fine-tuning walkthrough

Fine-tune a small LLM to turn a free-text job posting into schema-valid JSON.
The repo is split into **three numbered steps** so you can follow the whole
fine-tuning lifecycle in order and *see* what each stage buys you.

See [`GUIDE.md`](GUIDE.md) for the deeper "why" and a menu of experiments.

## Setup (uv)

```bash
uv sync                 # installs deps into .venv
```

Runs unmodified on a CUDA GPU (Colab/Kaggle), Apple Silicon (MPS), or CPU —
`common.py` picks the right device + dtype for you (CUDA→bf16, MPS→fp16, CPU→fp32).

## The three steps — run them in order

```bash
uv run python step1_baseline.py     # 1. base model: zero-shot vs few-shot
uv run python step2_train.py        # 2. LoRA fine-tune, saves an adapter
uv run python step3_evaluate.py     # 3. fine-tuned: zero-shot vs few-shot (+ base for reference)
```

| Step | Script | What it answers |
|------|--------|-----------------|
| 1 | `step1_baseline.py` | How good is the model *before* training — and how far does **prompting alone** (few-shot) get you? Fine-tuning has to beat this. |
| 2 | `step2_train.py` | Train a LoRA adapter. Watch `loss` fall and `eval_loss` for overfitting. |
| 3 | `step3_evaluate.py` | Does the fine-tuned model beat few-shot using a **short** prompt? Does few-shot still help after training? |

**The win condition for fine-tuning:** match or beat *base few-shot* using a
*zero-shot* prompt — the behavior is baked into the weights, so you stop paying
for demo tokens on every request.

## Files

| File | Role |
|------|------|
| `common.py` | Shared config (the experiment **dashboard**), schema/system-prompt, seeded data split, few-shot builder, generation + scoring, model loading. **Change one knob here, re-run, note what changed.** |
| `step1_baseline.py` / `step2_train.py` / `step3_evaluate.py` | The three steps above. |
| `data/job_postings.jsonl` | 20 hand-labeled examples (raw posting + gold JSON). |
| `data/{train,val,test}_split.jsonl` | Written by `prepare_splits()`; the seeded split every step reuses (no leakage). |

## Metrics that matter

Not loss — **task** metrics: JSON validity rate, field-level exact match, and
skills-list F1. Always read the actual failures, not just the aggregate.

## What to try next

Open `common.py` and change **one** knob at a time (see `GUIDE.md` for the full
list): `FEWSHOT_N`, dataset size, `LEARNING_RATE`, `EPOCHS`, LoRA `r`/`alpha`,
`USE_4BIT` (QLoRA, CUDA only), or swap `MODEL_NAME` to another family.
