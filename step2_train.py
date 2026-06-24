"""
STEP 2  --  Fine-tune a LoRA adapter on the training split.

LoRA freezes the 0.5B base weights and trains tiny adapter matrices instead,
so this is fast and the saved adapter is only a few MB. While it runs, watch:

  - train `loss` falling step by step
  - `eval_loss` (on the val split) printed each epoch -- if it RISES while train
    loss keeps falling, that's OVERFITTING (memorizing the few training rows).
  - `mean_token_accuracy` -- measured only on the assistant/JSON tokens, which
    is the proof that loss masking (assistant_only_loss) is working.

    python step2_train.py
"""

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from common import (BATCH_SIZE, EPOCHS, GRAD_ACCUM, LEARNING_RATE, LORA_ALPHA,
                    LORA_DROPOUT, LORA_R, MAX_LEN, MODEL_NAME, OUTPUT_DIR, SEED,
                    TARGET_MODULES, USE_4BIT, WARMUP_RATIO, get_device,
                    get_dtype, prepare_splits, to_chat)


def main():
    train, val, _ = prepare_splits()
    print(f"train={len(train)}  val={len(val)}")

    train_ds = Dataset.from_list([to_chat(r) for r in train])
    val_ds = Dataset.from_list([to_chat(r) for r in val])

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

    # These two only pay off on a CUDA GPU; on MPS/CPU they hurt or break, so we
    # gate them on the device. get_dtype() already handles bf16-vs-fp16 loading.
    on_cuda = get_device() == "cuda"

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        max_length=MAX_LEN,
        bf16=on_cuda,                    # bf16 autocast: CUDA only
        gradient_checkpointing=on_cuda,  # memory-vs-compute trade; only worth it on big GPU models
        logging_steps=1,
        eval_strategy="epoch",           # watch eval_loss vs train loss for overfitting
        save_strategy="epoch",
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
    trainer.save_model(OUTPUT_DIR)
    print(f"\nLoRA adapter saved to {OUTPUT_DIR}/  ->  next: python step3_evaluate.py")


if __name__ == "__main__":
    main()
