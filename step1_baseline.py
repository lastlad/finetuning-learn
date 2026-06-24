"""
STEP 1  --  Baseline: how good is the model BEFORE any fine-tuning?

Run this FIRST. Fine-tuning only "counts" if it later beats these numbers.
We measure two prompting strategies on the held-out test set:

  (a) zero-shot  -- just the schema in the system prompt
  (b) few-shot   -- schema + a few worked examples in the prompt (still NO training)

If few-shot alone is good enough, you might not need to fine-tune at all --
that is a real, money-saving finding, not a failure.

    python step1_baseline.py
"""

from common import (FEWSHOT_N, TEST_SPLIT, evaluate, load_base_model,
                    load_fewshot_demos, load_rows, load_tokenizer, prepare_splits)


def main():
    prepare_splits()                 # writes the split files every step reuses
    test = load_rows(TEST_SPLIT)
    tok = load_tokenizer()
    base = load_base_model()

    # (a) zero-shot
    evaluate(base, tok, test, "BASE | zero-shot")

    # (b) few-shot -- same weights, just richer prompt
    demos = load_fewshot_demos(FEWSHOT_N)
    evaluate(base, tok, test, f"BASE | {FEWSHOT_N}-shot", demos=demos)

    print("\nNext: python step2_train.py")


if __name__ == "__main__":
    main()
