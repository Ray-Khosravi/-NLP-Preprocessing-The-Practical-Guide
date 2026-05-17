"""
Example 02 — Tokenization
=========================

Word, sentence, and subword tokenization side by side.
Shows why you must use a Transformer's own tokenizer for Transformer models.

Dependencies: nltk, transformers (optional — the subword section is skipped if missing).

Run:
    python examples/02_tokenization.py
"""

import sys


SAMPLE = (
    "I love unbelievable NLP. It's powerful! "
    "Transformers like BERT and GPT are state-of-the-art."
)


def section(title: str) -> None:
    print("\n" + "─" * 70)
    print(title)
    print("─" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Whitespace tokenization (naive)
# ─────────────────────────────────────────────────────────────────────────────
def demo_naive_split():
    section("1. Naive .split() — the wrong way")
    tokens = SAMPLE.split()
    print(f"  Tokens ({len(tokens)}): {tokens}")
    print("\n  Notice: 'powerful!' and 'NLP.' include punctuation glued to words.")
    print("  This is almost never what you want.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Word tokenization (NLTK)
# ─────────────────────────────────────────────────────────────────────────────
def demo_word_tokenize():
    section("2. NLTK word_tokenize — proper word tokenization")
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            print("  Downloading 'punkt'...")
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(SAMPLE)
        print(f"  Tokens ({len(tokens)}): {tokens}")
        print("\n  Notice: punctuation is now its own token. Contractions are split.")
    except ImportError:
        print("  ⚠️  nltk not installed — `pip install nltk`")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sentence tokenization
# ─────────────────────────────────────────────────────────────────────────────
def demo_sent_tokenize():
    section("3. NLTK sent_tokenize — splitting into sentences")
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(SAMPLE)
        for i, s in enumerate(sentences, 1):
            print(f"  Sentence {i}: {s}")
    except ImportError:
        print("  ⚠️  nltk not installed")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Subword tokenization with a Transformer tokenizer
# ─────────────────────────────────────────────────────────────────────────────
def demo_subword():
    section("4. Subword tokenization (BERT) — how Transformers see text")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("  ⚠️  transformers not installed — skipping. (pip install transformers)")
        return

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(SAMPLE)
    ids = tokenizer.encode(SAMPLE)

    print(f"  Subword tokens ({len(tokens)}):")
    print(f"    {tokens}")
    print(f"\n  Token IDs ({len(ids)}): {ids[:15]}...  [showing first 15]")
    print("\n  Notice 'unbelievable' becomes: ['un', '##bel', '##ie', '##va', '##ble']")
    print("  The '##' prefix means 'this is a continuation of the previous token'.")
    print("  This is how BERT handles rare/unknown words — by composing them from pieces.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Different models, different tokenizers
# ─────────────────────────────────────────────────────────────────────────────
def demo_tokenizer_comparison():
    section("5. Same text, different models — different tokenizations")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return

    text = "Tokenization isn't always intuitive."
    models = [
        ("bert-base-uncased",   "BERT"),
        ("roberta-base",        "RoBERTa"),
        ("gpt2",                "GPT-2"),
    ]

    print(f"  Input: {text!r}\n")
    for model_id, name in models:
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            out = tok.tokenize(text)
            print(f"  {name:<10s} ({len(out)} tokens): {out}")
        except Exception as e:
            print(f"  {name:<10s}: skipped ({e.__class__.__name__})")

    print("\n  → Each model expects its own tokenizer. Never mix them.")


def demo_recommendation():
    section("6. Recommendation")
    print("""
  ✅ Word tokenization (NLTK / spaCy):
       Use for classical NLP — TF-IDF, Bag of Words, etc.

  ✅ Sentence tokenization:
       Use for summarization, sentence-level classification,
       or splitting docs before feeding chunks to a model.

  ✅ Subword tokenization (HuggingFace tokenizers):
       Always use the MODEL'S OWN tokenizer for Transformers.
       Never feed pre-split tokens to BERT/GPT/etc.

  ❌ .split():
       Almost never appropriate for serious NLP.
""")


if __name__ == "__main__":
    print("=" * 70)
    print("NLP PREPROCESSING — STEP 02: TOKENIZATION")
    print("=" * 70)
    print(f"\nInput text: {SAMPLE!r}")

    demo_naive_split()
    demo_word_tokenize()
    demo_sent_tokenize()
    demo_subword()
    demo_tokenizer_comparison()
    demo_recommendation()
