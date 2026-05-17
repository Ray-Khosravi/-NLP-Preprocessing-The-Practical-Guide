"""
Example 04 — Removing Stop Words
================================

Demonstrates standard stop-word removal AND the critical importance of
preserving negation words for sentiment, medical, and legal NLP.

Run:
    python examples/04_stopwords.py
"""

import nltk


# Make sure the stopwords corpus is available
for resource in ("corpora/stopwords", "tokenizers/punkt", "tokenizers/punkt_tab"):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1], quiet=True)


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ─────────────────────────────────────────────────────────────────────────────
# Stop-word sets
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_STOPS = set(stopwords.words("english"))

NEGATIONS = {
    "not", "no", "nor", "never", "without", "against",
    "isn't", "aren't", "wasn't", "weren't",
    "don't", "doesn't", "didn't",
    "won't", "wouldn't", "shan't", "shouldn't",
    "can't", "cannot", "couldn't",
    "hasn't", "haven't", "hadn't",
}

SAFE_STOPS = DEFAULT_STOPS - NEGATIONS


def section(title: str) -> None:
    print("\n" + "─" * 70)
    print(title)
    print("─" * 70)


def remove_stops(text: str, stops: set[str]) -> list[str]:
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t not in stops and t.isalpha()]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Show the default stop word list
# ─────────────────────────────────────────────────────────────────────────────
def demo_what_is_in_the_list():
    section("1. What's in NLTK's default English stop-word list?")
    print(f"  Total entries: {len(DEFAULT_STOPS)}")
    print(f"  First 30: {sorted(DEFAULT_STOPS)[:30]}")
    print(f"\n  ⚠️  These are in the default list:")
    for word in sorted(NEGATIONS):
        if word in DEFAULT_STOPS:
            print(f"     - {word!r}")
    print("\n  → Without customizing, NLTK removes 'not', 'no', 'never', etc.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Topic modeling case — full removal is good
# ─────────────────────────────────────────────────────────────────────────────
def demo_topic_modeling():
    section("2. ✅ Topic modeling — full removal works well")
    text = "The government announced a new climate policy for renewable energy."
    cleaned = remove_stops(text, DEFAULT_STOPS)
    print(f"  Input  : {text}")
    print(f"  Output : {cleaned}")
    print("\n  → The topic ('climate policy, renewable energy') is now obvious.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sentiment analysis — DANGER ZONE
# ─────────────────────────────────────────────────────────────────────────────
def demo_sentiment_disaster():
    section("3. ❌ Sentiment — naive removal flips meaning")

    examples = [
        ("This movie is not good.",                "NEGATIVE"),
        ("I don't love this product.",             "NEGATIVE"),
        ("The food was never bad.",                "POSITIVE"),
        ("This isn't bad at all.",                 "POSITIVE"),
    ]

    print(f"  {'Sentence':<35s} | {'True':<9s} | Naive removal → reads as")
    print("  " + "-" * 78)
    for text, true_label in examples:
        naive = remove_stops(text, DEFAULT_STOPS)
        # A toy sentiment heuristic, just for illustration
        pos_words = {"good", "love", "great"}
        neg_words = {"bad", "terrible", "awful"}
        if any(w in pos_words for w in naive):
            inferred = "POSITIVE"
        elif any(w in neg_words for w in naive):
            inferred = "NEGATIVE"
        else:
            inferred = "UNCLEAR"
        flag = "🚨" if inferred != true_label else "  "
        print(f"  {flag} {text:<33s} | {true_label:<9s} | {naive} → {inferred}")

    print("\n  → Naive stop-word removal FLIPS the polarity of every example above.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. The fix: custom stop-word list
# ─────────────────────────────────────────────────────────────────────────────
def demo_fix():
    section("4. ✅ The fix: customize your stop-word list")

    text = "This movie is not good."
    naive = remove_stops(text, DEFAULT_STOPS)
    smart = remove_stops(text, SAFE_STOPS)

    print(f"  Input  : {text}")
    print(f"  Naive  (default stops)  : {naive}   ← meaning destroyed")
    print(f"  Smart  (negations kept) : {smart}   ← meaning preserved")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Medical-NLP horror story
# ─────────────────────────────────────────────────────────────────────────────
def demo_medical_horror():
    section("5. ❌ Medical NLP — stop-word removal can be dangerous")

    notes = [
        "Patient has no chest pain.",
        "Patient denies any shortness of breath.",
        "No history of cardiac arrest.",
    ]
    print("  Original clinical notes vs. naive cleaning:\n")
    for note in notes:
        naive = remove_stops(note, DEFAULT_STOPS)
        print(f"  Original : {note}")
        print(f"  Cleaned  : {naive}")
        print(f"  → Now reads as the OPPOSITE clinical finding 🚨\n")


def demo_recommendation():
    section("6. Recommendation")
    print("""
  ✅ Remove stop words for:
       - Topic modeling (LDA, NMF)
       - Keyword extraction
       - TF-IDF for document classification
       - Search-engine indexing
       - Text clustering by topic

  ⚠️ Customize the list first for:
       - Sentiment analysis  (keep 'not', 'no', 'never', etc.)
       - Question answering  (keep wh-words, modals)

  ❌ Don't remove stop words for:
       - Medical NLP  ('no fever' ≠ 'fever')
       - Legal NLP    ('shall not' ≠ 'shall')
       - Transformer / LLM inputs
       - Translation
       - Named Entity Recognition
""")


if __name__ == "__main__":
    print("=" * 70)
    print("NLP PREPROCESSING — STEP 04: STOP-WORD REMOVAL")
    print("=" * 70)
    demo_what_is_in_the_list()
    demo_topic_modeling()
    demo_sentiment_disaster()
    demo_fix()
    demo_medical_horror()
    demo_recommendation()
