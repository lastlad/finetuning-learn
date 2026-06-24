"""
STEP 3  --  Evaluate the FINE-TUNED model on the same held-out test set.

We test the fine-tuned model with BOTH prompting styles, and reprint the base
numbers alongside, so the full comparison lands in one place:

  base zero-shot  |  base few-shot  |  fine-tuned zero-shot  |  fine-tuned few-shot

The WIN CONDITION for fine-tuning: match (or beat) base few-shot using a SHORT
zero-shot prompt -- i.e. the behavior is now baked into the weights, so you no
longer pay for demo tokens on every single request.

Things to look for:
  - Did fine-tuned zero-shot beat base few-shot? (the core question)
  - Does few-shot STILL help the fine-tuned model, or has it stopped mattering?

    python step3_evaluate.py
"""

from common import (FEWSHOT_N, TEST_SPLIT, evaluate, load_base_model,
                    load_fewshot_demos, load_finetuned_model, load_rows,
                    load_tokenizer, prepare_splits)


def main():
    prepare_splits()
    test = load_rows(TEST_SPLIT)
    tok = load_tokenizer()
    demos = load_fewshot_demos(FEWSHOT_N)

    # Base, for reference -- evaluate BEFORE wrapping with the adapter.
    base = load_base_model()
    evaluate(base, tok, test, "BASE | zero-shot")
    evaluate(base, tok, test, f"BASE | {FEWSHOT_N}-shot", demos=demos)

    # Fine-tuned: the same base weights + the trained LoRA adapter.
    tuned = load_finetuned_model(base)
    evaluate(tuned, tok, test, "FINE-TUNED | zero-shot")
    evaluate(tuned, tok, test, f"FINE-TUNED | {FEWSHOT_N}-shot", demos=demos)


if __name__ == "__main__":
    main()
