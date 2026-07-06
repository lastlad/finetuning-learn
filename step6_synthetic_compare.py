"""
STEP 6 (EXPERIMENT, GUIDE §4)  --  Does synthetic data help?

Trains the SAME recipe on three different TRAIN sets and evaluates each on the
SAME hand-labeled gold test set (zero-shot):

  - hand       : the 14 hand-labeled train rows
  - synthetic  : the ~100 reverse-generated rows from step5
  - mixed      : hand + synthetic

The val and test splits are ALWAYS hand-labeled gold -- synthetic data never
touches evaluation, so "did it help?" is measured only on real, human-labeled
examples. That's the honest test of distillation.

Slow on MPS (trains once per source, ~100 rows each). Run in the background;
trim TRAIN_SOURCES (e.g. drop "hand") to shrink the job.

    uv run python step6_synthetic_compare.py
"""

from common import (OUTPUT_DIR, SYNTHETIC_PATH, TEST_SPLIT, evaluate,
                    load_base_model, load_finetuned_model, load_rows,
                    load_tokenizer, prepare_splits, train_lora)

TRAIN_SOURCES = ["hand", "synthetic", "mixed"]
EPOCHS = 3   # far more data than step2's 14 rows -> fewer passes needed (also much faster)


def build_train(source, hand_train, synthetic):
    if source == "hand":
        return hand_train
    if source == "synthetic":
        return synthetic
    return hand_train + synthetic           # mixed


def main():
    hand_train, val, _ = prepare_splits()   # val/test = hand-labeled gold, held fixed
    test = load_rows(TEST_SPLIT)
    synthetic = load_rows(SYNTHETIC_PATH)
    tok = load_tokenizer()
    print(f"hand_train={len(hand_train)}  synthetic={len(synthetic)}  "
          f"val={len(val)}  test={len(test)}")

    results = []
    for source in TRAIN_SOURCES:
        train_rows = build_train(source, hand_train, synthetic)
        out_dir = f"{OUTPUT_DIR}-{source}"
        print(f"\n{'#'*18}  training '{source}' on {len(train_rows)} rows -> {out_dir}  {'#'*18}")
        train_lora(train_rows, val, out_dir, epochs=EPOCHS)

        base = load_base_model()                         # fresh base each time
        tuned = load_finetuned_model(base, out_dir)
        m = evaluate(tuned, tok, test,
                     f"FINE-TUNED | train={source} ({len(train_rows)} rows) | zero-shot")
        results.append((source, len(train_rows), m))

    print("\n\n============== SYNTHETIC DATA COMPARISON ==============")
    print(f"{'source':>10} | {'rows':>5} | {'JSON valid':>10} | {'field acc':>9} | {'skills F1':>9}")
    print("-" * 58)
    for source, n, m in results:
        print(f"{source:>10} | {n:>5} | {m['json_valid']*100:>9.0f}% | "
              f"{m['field_acc']*100:>8.0f}% | {m['skills_f1']:>9.2f}")
    print("\nDid synthetic-only match hand? Did mixed beat both? Read the failures, not just the table.")


if __name__ == "__main__":
    main()
