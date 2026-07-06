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
- `step4_ablation.py` — data-size ablation: train on 5/10/14 rows, print accuracy vs size
- `step5_gen_synthetic.py` — generate synthetic (posting, JSON) pairs via a larger model (needs `.env` API key)
- `step6_synthetic_compare.py` — hand vs synthetic vs mixed training, all evaluated on the gold test set
- `data/{train,val,test}_split.jsonl` — written by `prepare_splits()` so every step uses the identical split
- `data/synthetic.jsonl` — written by `step5`

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
code runs unmodified on a free GPU or locally.

> **Timing reality.** On a CUDA GPU a 0.5B LoRA trains in a couple of minutes. On
> Apple Silicon (MPS) it's much slower, and back-to-back runs **thermal-throttle** —
> a single 14-row train can take ~15 min and a 100-row run can take *hours*, with
> later runs in a sweep penalized regardless of size. Judge experiments by the task
> metrics, not wall-clock; run long jobs in the background.

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
Run `step4_ablation.py`: it trains the same recipe on the first 5, then 10, then
all 14 train rows and prints accuracy vs size. (Edit `TRAIN_SIZES` to change the
points.) You'll be surprised how far a tiny dataset gets you for a narrow format
task.

### 4. Synthetic data
`step5_gen_synthetic.py` uses a larger model (Claude Opus 4.8) to generate 100+
new (posting, JSON) pairs via **reverse generation** — sample the gold fields in
code (so labels are correct by construction and you control enum/null coverage),
then have the model write a posting that realizes them. `step6_synthetic_compare.py`
then trains hand-only vs synthetic-only vs mixed and evaluates all three on the
**hand-labeled gold** test set (synthetic never touches eval). This is your intro
to distillation / quality-vs-quantity tradeoffs.

**Setup:** put your key in a git-ignored `.env` as `ANTHROPIC_API_KEY=sk-ant-...`
(loaded via `python-dotenv`), then `uv run python step5_gen_synthetic.py`.
**Always spot-check** a few generated rows by eye — does the prose match the labels,
with nulls omitted and nothing invented? — before training on 100 of them.

What you'll likely see: synthetic-only (zero hand rows) gets most of the way but
lags real data on scalar *conventions* (location/salary normalization); **mixed**
tends to win because the hand rows anchor the real distribution while synthetic adds
coverage. Widen the sampler pools in `step5` for diversity; correlate salary size
with seniority if you want realistic magnitudes. On MPS the 100-row runs take hours —
lower `EPOCHS` (more data needs fewer passes) and/or trim `TRAIN_SOURCES`/`N_EXAMPLES`.

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
- `warmup_ratio` and a cosine scheduler are set. `bf16`/`gradient_checkpointing` are **device-gated** (on for CUDA, off on MPS/CPU where they misbehave) in `common.train_lora` — the single recipe `step2` and `step4`/`step6` all share.
- Set `report_to="wandb"` to actually *look at* your loss curves.

### 7. Evaluation (in `step3_evaluate.py`, scoring in `common.py`)
- **JSON validity rate**, **field-level exact match**, and **skills-list F1** — task-specific metrics, not just loss.
- Always read the actual failures, not just the aggregate number.
- **Mind the resolution of your test set.** With only 3 gold rows (36 field-slots),
  an 86% vs 89% difference is a *single field* — below the noise floor, and skills F1
  moves over just 6 list-slots. Once model deltas shrink to the size of your
  measurement noise, the highest-leverage move is **a bigger held-out test set**
  (hand-label or reverse-generate-then-verify 15–25 more rows), not another training
  knob. Watch for confounds too: comparing runs at different `EPOCHS` muddies the result.
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
