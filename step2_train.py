"""
STEP 2  --  Fine-tune a LoRA adapter on the training split.

LoRA freezes the 0.5B base weights and trains tiny adapter matrices instead,
so this is fast and the saved adapter is only a few MB. While it runs, watch:

  - train `loss` falling step by step
  - `eval_loss` (on the val split) printed each epoch -- if it RISES while train
    loss keeps falling, that's OVERFITTING (memorizing the few training rows).
  - `mean_token_accuracy` -- measured only on the assistant/JSON tokens, which
    is the proof that loss masking (assistant_only_loss) is working.

The actual training recipe lives in `common.train_lora()` so it can be reused
(the data-size ablation calls the same function). Tune knobs in `common.py`.

    python step2_train.py
"""

from common import OUTPUT_DIR, prepare_splits, train_lora


def main():
    train, val, _ = prepare_splits()
    print(f"train={len(train)}  val={len(val)}")

    train_lora(train, val, OUTPUT_DIR)
    print(f"\nLoRA adapter saved to {OUTPUT_DIR}/  ->  next: python step3_evaluate.py")


if __name__ == "__main__":
    main()
