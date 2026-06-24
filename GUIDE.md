# Job Posting → Structured JSON: a fine-tuning starter project

Fine-tune a small LLM to convert a free-text job posting into a clean, schema-valid
JSON object. This is a deliberately small, fully runnable scaffold whose real job is
to make you *touch every part of the fine-tuning lifecycle* and build judgment.

## Files
- `data/job_postings.jsonl` — 20 hand-labeled examples (raw posting + gold JSON labels)
- `common.py` — shared config (your experiment **dashboard**), schema, seeded split, few-shot builder, generation + scoring, model loading
- `step1_baseline.py` — base model: zero-shot vs few-shot (run **first**)
- `step2_train.py` — LoRA supervised fine-tuning with TRL's `SFTTrainer`
- `step3_evaluate.py` — base vs fine-tuned, zero-shot vs few-shot, on the held-out test split
- `data/{train,val,test}_split.jsonl` — written by `prepare_splits()` so every step uses the identical split

## The target schema
Defined once, in the system prompt inside `common.py` (imported by every step):
`title, company, location, workplace_type, employment_type, seniority, salary_min,
salary_max, salary_currency, salary_period, required_skills, preferred_skills,
min_years_experience, education`. The interesting parts are the **enums**
(workplace/employment/seniority/period), the **nulls** (most postings omit fields),
the **lists** (skills), and the **numbers** (salary, years).

## Quick start (uv — runs on Colab/Kaggle CUDA, Apple Silicon MPS, or CPU)
```bash
uv sync                            # installs deps into .venv
uv run python step1_baseline.py    # baseline: zero-shot vs few-shot (BEFORE you train)
uv run python step2_train.py       # trains, saves LoRA adapter, writes the splits
uv run python step3_evaluate.py    # base vs fine-tuned: JSON validity + field accuracy
```
`common.py` auto-selects device/dtype (CUDA→bf16, MPS→fp16, CPU→fp32), so the same
code runs unmodified on a free GPU or locally. A 0.5B model trains in a couple of
minutes on a GPU (longer on MPS/CPU) — fast enough to run many experiments.

> QLoRA (`USE_4BIT=True`) needs `bitsandbytes`, which is **CUDA-only** — leave it off on Mac/MPS.

> Library APIs in this space move fast. If something errors, check the current
> TRL/PEFT docs — `SFTConfig`/`LoraConfig` argument names occasionally change.

---

## Do these experiments, roughly in order

Each maps to a knob in `common.py`. **Change one thing at a time** and write down what happened.

### 1. Establish baselines FIRST (before you train anything)
Run `step1_baseline.py`. It prints the base model's JSON-validity rate and field
accuracy for **both zero-shot and few-shot** (the few-shot demo count is the
`FEWSHOT_N` knob in `common.py`). **Fine-tuning only counts if it beats this** —
and beats the few-shot prompt. Tip: few-shot helps the *base* model a lot, but can
*hurt* a fine-tuned one — at inference, match the prompt shape you trained on.

### 2. The data pipeline
- The train/val/test split happens *before* any peeking (`prepare_splits()` in `common.py`) — that's how you avoid leakage. Few-shot demos are drawn from the **train** split only, for the same reason.
- `to_chat()` (in `common.py`) builds the conversational format; the assistant turn is the gold JSON.
- `assistant_only_loss=True` does **loss masking** — loss is computed only on the
  assistant tokens. Getting this wrong is the #1 silent fine-tuning bug.
- **Chat template**: the trainer auto-patches the template for known families
  (Qwen2.5/Qwen3/Llama 3). A template mismatch produces garbage with no error — be aware of it.

### 3. Data-size ablation (do this early — it's the most eye-opening)
Train on the first 5, then 10, then all 14 train rows. Plot accuracy vs size.
You'll be surprised how far a tiny dataset gets you for a narrow format task.

### 4. Synthetic data
Use a larger model to generate 100+ new (posting, JSON) pairs, mix them in, and
compare against the hand-labeled set. This is your intro to distillation /
quality-vs-quantity tradeoffs.

### 5. Methods
- Start with **LoRA** (default). Then flip `USE_4BIT = True` for **QLoRA** and feel the memory drop (CUDA only — `bitsandbytes` won't run on MPS).
- Sweep LoRA `r` (4 → 64) and `alpha`; change `TARGET_MODULES` (try `"all-linear"`).
- Do one **full fine-tune** (drop `peft_config`, lower the LR to ~2e-5) and compare.
- Compare a **base vs instruct** starting model, and swap **model families**
  (Qwen ↔ Llama ↔ Gemma) on the same data.

### 6. Training mechanics / hyperparameters
- **Learning rate** is the highest-leverage knob — sweep `5e-5 … 5e-4`.
- Watch `eval_loss` diverge from train loss → that's **overfitting**; tune `EPOCHS`.
- Effective batch size = `BATCH_SIZE * GRAD_ACCUM` (gradient accumulation).
- `warmup_ratio`, cosine scheduler, `bf16`, `gradient_checkpointing` are all set — try toggling them.
- Set `report_to="wandb"` to actually *look at* your loss curves.

### 7. Evaluation (in `step3_evaluate.py`, scoring in `common.py`)
- **JSON validity rate**, **field-level exact match**, and **skills-list F1** — task-specific metrics, not just loss.
- Always read the actual failures, not just the aggregate number.
- Later: add an **LLM-as-judge** pass for fuzzier qualities.

### 8. Deployment (do it once, end to end)
- **Merge** the adapter: `model.merge_and_unload()` then `save_pretrained()`.
- Export to **GGUF** → run in Ollama/llama.cpp, or serve with **vLLM**.
- Push to the **Hugging Face Hub**.

### 9. Concepts to meet here, then explore in a later project
- The decision rule this project embodies: **fine-tune for format/behavior, use RAG for knowledge.**
- Preference tuning (DPO/ORPO), catastrophic forgetting (check general ability didn't drop),
  continued pretraining vs SFT.

---

## A note on scope
20 rows and a 0.5B model is intentionally tiny — it's fast enough to iterate and
big enough to show real signal on a narrow task. For anything you'd actually ship,
scale the data to a few hundred to a few thousand examples and reassess.
