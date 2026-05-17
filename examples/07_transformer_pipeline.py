"""
Example 07 — Transformer Pipeline
=================================

Shows the modern way: minimal cleaning, the model's own tokenizer,
and inference via Hugging Face Transformers.

This is the END-TO-END counterpart to the classical pipeline in example 06.

Run:
    python examples/07_transformer_pipeline.py

Note: first run will download ~250 MB of model weights.
"""

import sys


def section(title: str) -> None:
    print("\n" + "─" * 70)
    print(title)
    print("─" * 70)


try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch
except ImportError:
    print("This example needs the 'transformers' and 'torch' packages.")
    print("Install with:  pip install transformers torch")
    sys.exit(1)


MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"


SAMPLES = [
    "This movie is not good.",
    "The movie was AMAZING!!! I really loved it.",
    "I don't love this product.",
    "The food was never bad.",
    "Patient has no chest pain.",
    "Wait... you really liked it?!",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Show what the tokenizer does (this IS the preprocessing for Transformers)
# ─────────────────────────────────────────────────────────────────────────────
def demo_tokenizer():
    section("1. The tokenizer IS the preprocessor")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    text = SAMPLES[1]
    print(f"  Input text : {text!r}\n")

    # tokenize() returns the human-readable subword pieces
    pieces = tokenizer.tokenize(text)
    print(f"  Subword pieces ({len(pieces)}):")
    print(f"    {pieces}\n")

    # __call__ returns model-ready tensors
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    print("  Model-ready encoding:")
    print(f"    input_ids      shape={tuple(enc['input_ids'].shape)}")
    print(f"    attention_mask shape={tuple(enc['attention_mask'].shape)}")
    print(f"    first 12 input_ids: {enc['input_ids'][0][:12].tolist()}")

    print("\n  → That's the entire preprocessing pipeline for a Transformer.")
    print("  → No stop-word removal, no stemming, no lowercasing by you.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Run sentiment classification on the samples
# ─────────────────────────────────────────────────────────────────────────────
def demo_sentiment():
    section("2. Sentiment predictions — preprocessing-free")

    sa = pipeline("sentiment-analysis", model=MODEL_ID)

    print(f"  {'Text':<55s} {'Label':<10s} {'Score'}")
    print("  " + "-" * 80)
    for text in SAMPLES:
        result = sa(text)[0]
        label, score = result["label"], result["score"]
        print(f"  {text:<55s} {label:<10s} {score:.3f}")

    print("""
  Look at the second example: 'I don't love this product.'
  → Correctly NEGATIVE, because the model sees 'don't love' together.
  → A classical pipeline that removed 'don't' would call this POSITIVE.""")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Counter-example: what happens if you pre-clean?
# ─────────────────────────────────────────────────────────────────────────────
def demo_overcleaning_hurts():
    section("3. What if we 'helpfully' preprocess before BERT?")

    import re
    from nltk.corpus import stopwords
    import nltk
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    stops = set(stopwords.words("english"))

    def over_clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return " ".join(w for w in text.split() if w not in stops)

    sa = pipeline("sentiment-analysis", model=MODEL_ID)

    print(f"  {'Original / Cleaned':<55s} {'Label':<10s} {'Score'}")
    print("  " + "-" * 80)
    for text in [
        "This movie is not good.",
        "I don't love this product.",
        "The food was never bad.",
    ]:
        cleaned = over_clean(text)
        r_orig    = sa(text)[0]
        r_cleaned = sa(cleaned)[0]
        print(f"  ORIG   : {text:<46s} {r_orig['label']:<10s} {r_orig['score']:.3f}")
        print(f"  CLEAN  : {cleaned:<46s} {r_cleaned['label']:<10s} {r_cleaned['score']:.3f}")
        print()

    print("  → Notice how aggressive cleaning frequently flips BERT's prediction.")
    print("  → Lesson: for Transformers, leave the text alone.")


def demo_summary():
    section("4. Summary")
    print("""
  The Transformer-style preprocessing recipe is almost laughably short:

      1. Strip obvious noise (HTML, control chars, weird encodings).
      2. Pass the text to the MODEL'S OWN tokenizer.
      3. Done.

  Why does this work?
      - The model was trained on natural text. Cleaning makes inputs
        look different from training data, hurting performance.
      - The tokenizer's subword vocabulary already handles rare words,
        morphology, and contractions correctly.
      - Negations, punctuation, and casing are signal for these models,
        not noise.
""")


if __name__ == "__main__":
    print("=" * 70)
    print("NLP PREPROCESSING — STEP 07: TRANSFORMER PIPELINE")
    print("=" * 70)
    print(f"\nModel: {MODEL_ID}")
    print("(First run will download model weights, ~250 MB.)")

    demo_tokenizer()
    demo_sentiment()
    demo_overcleaning_hurts()
    demo_summary()
