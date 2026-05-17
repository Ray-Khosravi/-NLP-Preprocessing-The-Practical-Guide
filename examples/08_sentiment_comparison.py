"""
Example 08 — Sentiment Analysis: Three Pipelines Compared
=========================================================

The "money shot" example. Same sentences, three pipelines:

    A. Naive classical pipeline   (removes negations — BREAKS sentiment)
    B. Smart classical pipeline   (keeps negations — works)
    C. Transformer pipeline       (no cleaning at all — works best)

You can literally watch the naive pipeline get the answers wrong.

Run:
    python examples/08_sentiment_comparison.py
"""

import re
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────────────────────────────────────
# 0. Resources
# ─────────────────────────────────────────────────────────────────────────────
for resource, package in [
    ("corpora/stopwords",    "stopwords"),
    ("corpora/wordnet",      "wordnet"),
    ("corpora/omw-1.4",      "omw-1.4"),
    ("tokenizers/punkt",     "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(package, quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Two cleaning functions — naive vs smart
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_STOPS = set(stopwords.words("english"))
NEGATIONS = {"not", "no", "nor", "never", "without", "against",
             "isn't", "aren't", "wasn't", "weren't",
             "don't", "doesn't", "didn't", "won't",
             "can't", "cannot", "couldn't"}
SAFE_STOPS = DEFAULT_STOPS - NEGATIONS

lemmatizer = WordNetLemmatizer()


def naive_clean(text: str) -> str:
    """The 'follow every preprocessing tutorial blindly' pipeline."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in DEFAULT_STOPS and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t, pos="v") for t in tokens]
    return " ".join(tokens)


def smart_clean(text: str) -> str:
    """The 'I actually thought about my task' pipeline."""
    text = text.lower()
    # Keep ! and ? as features — they're sentiment signal
    text = re.sub(r"[^a-z\s!?]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in SAFE_STOPS and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t, pos="v") for t in tokens]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Training data — short and explicit for the demo
# ─────────────────────────────────────────────────────────────────────────────
TRAIN = [
    ("I love this movie.",             1),
    ("This product is amazing.",       1),
    ("Best purchase I ever made.",     1),
    ("The food was excellent.",        1),
    ("Truly wonderful experience.",    1),
    ("Great service and great food.",  1),

    ("I hate this movie.",             0),
    ("This product is terrible.",      0),
    ("Worst purchase I ever made.",    0),
    ("The food was awful.",            0),
    ("Truly disappointing.",           0),
    ("Bad service and bad food.",      0),

    # Including negations in training so the model can learn them
    ("Not good at all.",               0),
    ("This isn't great.",              0),
    ("I don't love this.",             0),
    ("It was not bad.",                1),
    ("Never had a better time.",       1),
    ("Without question, the best.",    1),
]
X_train = [t for t, _ in TRAIN]
y_train = [y for _, y in TRAIN]

TEST = [
    ("This movie is not good.",        0),
    ("I don't love this product.",     0),
    ("The food was never bad.",        1),
    ("This isn't terrible.",           1),
    ("I really loved it!",             1),
    ("Worst experience ever.",         0),
]


def build_pipeline(cleaner):
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=cleaner, ngram_range=(1, 2), min_df=1)),
        ("clf",   LogisticRegression(max_iter=1000)),
    ])


def section(title: str) -> None:
    print("\n" + "─" * 76)
    print(title)
    print("─" * 76)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Show what each cleaner does
# ─────────────────────────────────────────────────────────────────────────────
def demo_cleaning_difference():
    section("1. Naive vs smart cleaning — what survives?")
    for text, _ in TEST:
        print(f"  Input  : {text}")
        print(f"  Naive  : {naive_clean(text)}")
        print(f"  Smart  : {smart_clean(text)}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Train two classical pipelines and compare on the test sentences
# ─────────────────────────────────────────────────────────────────────────────
def demo_classical():
    section("2. Classical pipelines — naive vs smart")
    naive  = build_pipeline(naive_clean).fit(X_train, y_train)
    smart  = build_pipeline(smart_clean).fit(X_train, y_train)

    print(f"  {'Sentence':<35s} {'Truth':<5s} {'Naive':<10s} {'Smart':<10s}")
    print("  " + "-" * 70)
    naive_right = smart_right = 0
    for text, truth in TEST:
        n = naive.predict([text])[0]
        s = smart.predict([text])[0]
        naive_right += int(n == truth)
        smart_right += int(s == truth)
        n_tag = "POS" if n == 1 else "NEG"
        s_tag = "POS" if s == 1 else "NEG"
        t_tag = "POS" if truth == 1 else "NEG"
        n_mark = "✓" if n == truth else "✗"
        s_mark = "✓" if s == truth else "✗"
        print(f"  {text:<35s} {t_tag:<5s} {n_tag} {n_mark:<6s} {s_tag} {s_mark}")

    print()
    print(f"  Naive  accuracy: {naive_right}/{len(TEST)}")
    print(f"  Smart  accuracy: {smart_right}/{len(TEST)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. (Optional) Transformer pipeline
# ─────────────────────────────────────────────────────────────────────────────
def demo_transformer():
    section("3. Transformer pipeline — no cleaning at all")
    try:
        from transformers import pipeline
    except ImportError:
        print("  ⚠️  transformers not installed — skipping. (pip install transformers torch)")
        return

    sa = pipeline("sentiment-analysis",
                  model="distilbert-base-uncased-finetuned-sst-2-english")

    print(f"  {'Sentence':<35s} {'Truth':<5s} {'BERT':<12s} {'Score'}")
    print("  " + "-" * 72)
    right = 0
    for text, truth in TEST:
        out = sa(text)[0]
        pred = 1 if out["label"] == "POSITIVE" else 0
        right += int(pred == truth)
        t_tag = "POS" if truth == 1 else "NEG"
        mark = "✓" if pred == truth else "✗"
        print(f"  {text:<35s} {t_tag:<5s} {out['label']:<10s} {mark}  {out['score']:.3f}")
    print()
    print(f"  Transformer accuracy: {right}/{len(TEST)}")


def demo_summary():
    section("4. Take-aways")
    print("""
  The same six test sentences ran through three pipelines:

    • Naive classical   → trips on every negation
    • Smart classical   → handles negations because we kept them in the vocab
    • Transformer       → handles negations because it was trained on real text

  The lesson is NOT "Transformers always win". The lesson is:

      You only need heavy preprocessing for classical models — and even
      then, it must be tailored to your task. Default pipelines lose
      sentiment signal by default.
""")


if __name__ == "__main__":
    print("=" * 76)
    print("NLP PREPROCESSING — STEP 08: SENTIMENT — THREE PIPELINES")
    print("=" * 76)
    demo_cleaning_difference()
    demo_classical()
    demo_transformer()
    demo_summary()
