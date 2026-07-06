"""
Shared config + helpers for the job-posting -> JSON fine-tuning walkthrough.

The three numbered scripts all import from here, so the SCHEMA, the DATA SPLIT,
and the SCORING are guaranteed identical at every step -- that's what makes the
baseline and the fine-tuned numbers a fair comparison.

    python step1_baseline.py     # base model: zero-shot vs few-shot
    python step2_train.py        # LoRA fine-tune, saves an adapter
    python step3_evaluate.py     # fine-tuned model: zero-shot vs few-shot

The CONFIG block below is your experiment dashboard. Change ONE knob, re-run,
and write down what happened. That discipline is the whole game.
"""

import json
import random
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# CONFIG  -- change ONE knob at a time.
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"   # swap: Qwen/Qwen3-0.6B, meta-llama/Llama-3.2-1B-Instruct, google/gemma-3-1b-it
DATA_DIR = "data"
DATA_PATH = f"{DATA_DIR}/job_postings.jsonl"
OUTPUT_DIR = "job-extractor-lora"
SEED = 42

# Data split (only 20 rows -- deliberately tiny so you feel the data-size effect)
N_VAL = 3
N_TEST = 3
TRAIN_SPLIT = f"{DATA_DIR}/train_split.jsonl"
VAL_SPLIT = f"{DATA_DIR}/val_split.jsonl"
TEST_SPLIT = f"{DATA_DIR}/test_split.jsonl"
SYNTHETIC_PATH = f"{DATA_DIR}/synthetic.jsonl"   # written by step5_gen_synthetic.py

# Few-shot: how many worked examples to drop into the prompt at eval time.
FEWSHOT_N = 3              # try 0, 2, 3, 5

# LoRA knobs
LORA_R = 16               # try 4, 8, 16, 32, 64
LORA_ALPHA = 32           # rule of thumb: alpha = 2 * r
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]   # or "all-linear"

# Training knobs (highest-leverage = LEARNING_RATE)
LEARNING_RATE = 2e-4      # LoRA likes higher LRs than full fine-tuning. Sweep 5e-5 .. 5e-4
EPOCHS = 5                # watch eval_loss for overfitting
BATCH_SIZE = 2
GRAD_ACCUM = 4            # effective batch size = BATCH_SIZE * GRAD_ACCUM = 8
MAX_LEN = 1024
WARMUP_RATIO = 0.1
USE_4BIT = False          # QLoRA -- CUDA-only (needs bitsandbytes). Leave False on Mac/MPS.

# Evaluation: which fields we score, and which are lists (scored with F1).
FIELDS = ["title", "company", "location", "workplace_type", "employment_type",
          "seniority", "salary_min", "salary_max", "salary_currency",
          "salary_period", "min_years_experience", "education"]
LIST_FIELDS = ["required_skills", "preferred_skills"]

# The schema lives in the SYSTEM PROMPT. The model learns to fill THIS shape.
SYSTEM_PROMPT = """You extract structured data from job postings.
Return ONLY a single JSON object, no prose, matching exactly this schema:
{
  "title": string,
  "company": string|null,
  "location": string|null,
  "workplace_type": "remote"|"hybrid"|"onsite"|null,
  "employment_type": "full_time"|"part_time"|"contract"|"internship"|"temporary"|null,
  "seniority": "intern"|"junior"|"mid"|"senior"|"lead"|"manager"|"executive"|null,
  "salary_min": number|null,
  "salary_max": number|null,
  "salary_currency": string|null,
  "salary_period": "year"|"month"|"week"|"day"|"hour"|null,
  "required_skills": string[],
  "preferred_skills": string[],
  "min_years_experience": number|null,
  "education": string|null
}
Use null when a field is not stated. Do not invent values."""


# ---------------------------------------------------------------------------
# Device / dtype  -- picks the right backend so this runs unmodified on Colab
# (CUDA), Apple Silicon (MPS), or CPU. This is why we don't hard-code bf16.
# ---------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dtype():
    dev = get_device()
    if dev == "cuda":
        return torch.bfloat16      # CUDA tensor cores love bf16
    if dev == "mps":
        return torch.float16       # MPS bf16 is unreliable; fp16 is solid
    return torch.float32           # CPU: keep it simple


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def prepare_splits():
    """Deterministically split the data and write the three split files.

    Splitting up front -- BEFORE any model sees anything -- is how you avoid
    train/test leakage. The fixed SEED makes it reproducible, so every step
    reuses the EXACT same test set and the numbers are comparable.
    """
    rows = load_rows(DATA_PATH)
    random.Random(SEED).shuffle(rows)        # seeded -> identical split every run
    test = rows[:N_TEST]
    val = rows[N_TEST:N_TEST + N_VAL]
    train = rows[N_TEST + N_VAL:]
    Path(DATA_DIR).mkdir(exist_ok=True)
    for path, split in [(TRAIN_SPLIT, train), (VAL_SPLIT, val), (TEST_SPLIT, test)]:
        Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in split))
    return train, val, test


def to_chat(row):
    """Conversational format SFTTrainer expects. The assistant turn is the gold
    JSON -- and with assistant_only_loss=True those are the ONLY tokens loss is
    computed on (loss masking). Getting that wrong is the #1 silent SFT bug."""
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": row["posting"]},
        {"role": "assistant", "content": json.dumps(row["labels"], ensure_ascii=False)},
    ]}


def load_fewshot_demos(n):
    """n worked (posting, gold-JSON) demos drawn from the TRAIN split ONLY.

    Demos must never come from the test set -- that would leak answers into the
    prompt and inflate the score for free. Same leakage rule as the split.
    """
    demos = []
    for r in load_rows(TRAIN_SPLIT)[:n]:
        demos.append({"role": "user", "content": r["posting"]})
        demos.append({"role": "assistant", "content": json.dumps(r["labels"], ensure_ascii=False)})
    return demos


# ---------------------------------------------------------------------------
# Generation + scoring
# ---------------------------------------------------------------------------
def generate(model, tok, posting, demos=None):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            *(demos or []),                       # few-shot turns go BEFORE the real question
            {"role": "user", "content": posting}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=512, do_sample=False,   # greedy = reproducible
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def parse_json(text):
    """Models sometimes wrap JSON in prose/fences. Try hard, then give up."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def score(pred, gold):
    """Returns (json_valid, scalar_correct, scalar_total, list_f1_sum, list_count)."""
    if pred is None:
        return 0, 0, len(FIELDS), 0.0, len(LIST_FIELDS)
    correct = sum(1 for f in FIELDS if pred.get(f) == gold.get(f))
    f1_sum = 0.0
    for f in LIST_FIELDS:
        p = set(x.lower() for x in pred.get(f, []) or [])
        g = set(x.lower() for x in gold.get(f, []) or [])
        if not p and not g:
            f1_sum += 1.0
        elif not p or not g:
            f1_sum += 0.0
        else:
            inter = len(p & g)
            prec, rec = inter / len(p), inter / len(g)
            f1_sum += 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    return 1, correct, len(FIELDS), f1_sum, len(LIST_FIELDS)


def evaluate(model, tok, rows, label, demos=None):
    """Run the model over the test rows, print the three task metrics that
    actually matter for structured extraction (loss is not one of them), and
    return them as a dict so callers (e.g. the ablation) can collect a table."""
    valid = corr = tot = lcount = 0
    lf1 = 0.0
    for r in rows:
        out = generate(model, tok, r["posting"], demos)
        v, c, t, f, lc = score(parse_json(out), r["labels"])
        valid += v; corr += c; tot += t; lf1 += f; lcount += lc
    metrics = {"json_valid": valid / len(rows),
               "field_acc": corr / tot,
               "skills_f1": lf1 / lcount}
    print(f"\n=== {label} ===")
    print(f"JSON valid:      {valid}/{len(rows)}  ({100*metrics['json_valid']:.0f}%)")
    print(f"Field accuracy:  {corr}/{tot}  ({100*metrics['field_acc']:.0f}%)")
    print(f"Skills list F1:  {metrics['skills_f1']:.2f}")
    return metrics


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def load_base_model():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=get_dtype())
    return model.to(get_device())


def load_finetuned_model(base, output_dir=OUTPUT_DIR):
    """Wrap the base model with a trained LoRA adapter (default OUTPUT_DIR)."""
    from peft import PeftModel
    return PeftModel.from_pretrained(base, output_dir)


def train_lora(train_rows, val_rows, output_dir, epochs=EPOCHS):
    """Train a LoRA adapter on the given rows and save it to output_dir.

    This is the single source of truth for the training recipe -- step2 trains
    once with it, the data-size ablation calls it repeatedly with different row
    counts. Heavy libs are imported lazily so eval-only steps stay fast to load.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    train_ds = Dataset.from_list([to_chat(r) for r in train_rows])
    val_ds = Dataset.from_list([to_chat(r) for r in val_rows])

    model_kwargs = {"dtype": get_dtype()}
    if USE_4BIT:   # QLoRA -- CUDA only
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    lora = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # bf16 autocast + gradient checkpointing only pay off on a CUDA GPU; on
    # MPS/CPU they hurt or break, so gate them on the device.
    on_cuda = get_device() == "cuda"

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        max_length=MAX_LEN,
        bf16=on_cuda,
        gradient_checkpointing=on_cuda,
        logging_steps=1,
        eval_strategy="epoch",           # watch eval_loss vs train loss for overfitting
        save_strategy="no",              # we save once at the end -> no per-epoch checkpoint dirs
        report_to="none",                # set "wandb" to see live loss curves
        assistant_only_loss=True,        # loss masking; trainer auto-patches the chat template
        seed=SEED,
        model_init_kwargs=model_kwargs,
    )

    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
