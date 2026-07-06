"""
EXPERIMENT (GUIDE §3)  --  Data-size ablation.

Train the SAME recipe on progressively more rows and watch how test accuracy
moves. Everything is held fixed EXCEPT the number of training rows -- that's
what makes this a clean ablation (change one variable, measure the effect).

The eye-opener: for a narrow format task, a tiny dataset gets you most of the
way. You'll likely see accuracy climb fast and then flatten -- the point where
more of THIS data stops helping (and you'd need harder/more-varied data instead).

Heads up: this trains once PER size, so it's slow on MPS/CPU (minutes each).
Best run in the background.

    uv run python step4_ablation.py
"""

from common import (OUTPUT_DIR, TEST_SPLIT, evaluate, load_base_model,
                    load_finetuned_model, load_rows, load_tokenizer,
                    prepare_splits, train_lora)

# Subsets of the 14-row train split. The split is seeded, so row order is fixed:
# "first 5" is always the same 5 rows -> reproducible.
TRAIN_SIZES = [5, 10, 14]


def main():
    train, val, _ = prepare_splits()
    test = load_rows(TEST_SPLIT)
    tok = load_tokenizer()

    results = []
    for n in TRAIN_SIZES:
        subset = train[:n]
        out_dir = f"{OUTPUT_DIR}-n{n}"
        print(f"\n{'#'*20}  training on {n} rows -> {out_dir}  {'#'*20}")
        train_lora(subset, val, out_dir)

        # Fresh base each time, then wrap with this size's adapter. Evaluate
        # zero-shot on the SAME held-out test set -> apples to apples.
        base = load_base_model()
        tuned = load_finetuned_model(base, out_dir)
        m = evaluate(tuned, tok, test, f"FINE-TUNED | {n} train rows | zero-shot")
        results.append((n, m))

    print("\n\n================== DATA-SIZE ABLATION ==================")
    print(f"{'rows':>5} | {'JSON valid':>10} | {'field acc':>9} | {'skills F1':>9}")
    print("-" * 47)
    for n, m in results:
        print(f"{n:>5} | {m['json_valid']*100:>9.0f}% | "
              f"{m['field_acc']*100:>8.0f}% | {m['skills_f1']:>9.2f}")
    print("\nRead it as a curve: where does accuracy stop climbing?")


if __name__ == "__main__":
    main()
