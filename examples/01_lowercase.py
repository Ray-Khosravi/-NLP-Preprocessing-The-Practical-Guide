"""
Example 01 — Lowercasing
========================

Demonstrates what lowercasing does, when it helps, and when it hurts.

Run:
    python examples/01_lowercase.py
"""

SAMPLES = [
    "The Movie was AMAZING! I love this MOVIE.",
    "Apple opened a new office in Toronto.",
    "I work in IT and live in the US.",
    "FREE MONEY!!! Click here NOW.",
]


def lowercase(text: str) -> str:
    """The entire transformation — that's it."""
    return text.lower()


def section(title: str) -> None:
    print("\n" + "─" * 70)
    print(title)
    print("─" * 70)


def demo_basic():
    section("1. Basic lowercasing")
    for text in SAMPLES:
        print(f"  Before : {text}")
        print(f"  After  : {lowercase(text)}\n")


def demo_vocabulary_collapse():
    """Show how lowercasing shrinks vocabulary for classical models."""
    section("2. Why it helps classical ML: vocabulary collapse")
    text = "Movie movie MOVIE Movies movies"
    raw_vocab = set(text.split())
    low_vocab = set(text.lower().split())
    print(f"  Text                  : {text}")
    print(f"  Raw vocabulary    ({len(raw_vocab)}) : {sorted(raw_vocab)}")
    print(f"  Lowered vocabulary ({len(low_vocab)}) : {sorted(low_vocab)}")
    print("\n  → 5 distinct tokens become 2. Smaller vocab = faster, simpler features.")


def demo_when_it_hurts():
    """Show how lowercasing destroys signal for NER-style tasks."""
    section("3. When lowercasing HURTS: NER, acronyms, proper nouns")

    cases = [
        ("Apple",   "company (Apple Inc.)"),
        ("apple",   "fruit"),
        ("US",      "country (United States)"),
        ("us",      "pronoun"),
        ("IT",      "Information Technology department"),
        ("it",      "pronoun"),
        ("Bush",    "person (e.g., George Bush)"),
        ("bush",    "plant"),
    ]
    print("  Word    | Meaning")
    print("  --------|---------------------------------------")
    for word, meaning in cases:
        print(f"  {word:<7s} | {meaning}")
    print("\n  After lowercasing, every row above collapses into its lowercase form")
    print("  and the model loses the ability to distinguish these meanings.")


def demo_recommendation():
    section("4. Recommendation")
    print("""
  ✅ Lowercase for:
       - TF-IDF / Bag of Words pipelines
       - Topic modeling (LDA, NMF)
       - Classical spam / sentiment classifiers
       - Naive search & keyword matching

  ❌ Avoid lowercasing for:
       - Named Entity Recognition (NER)
       - Machine translation
       - Anything with meaningful acronyms (medical, legal, finance)
       - LLM / GPT inputs — they expect natural text
       - BERT models WHOSE NAME DOES NOT END IN '-uncased'
""")


if __name__ == "__main__":
    print("=" * 70)
    print("NLP PREPROCESSING — STEP 01: LOWERCASING")
    print("=" * 70)
    demo_basic()
    demo_vocabulary_collapse()
    demo_when_it_hurts()
    demo_recommendation()
