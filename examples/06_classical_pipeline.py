"""
Example 06 — Full Classical Pipeline
====================================

A complete classical NLP pipeline:

    raw text  →  cleaning  →  TF-IDF  →  Logistic Regression  →  prediction

We train a tiny binary classifier on a toy corpus so you can see the
whole flow end to end in one file.

Run:
    python examples/06_classical_pipeline.py
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ─────────────────────────────────────────────────────────────────────────────
# 0. Ensure NLTK resources are present
# ─────────────────────────────────────────────────────────────────────────────
for resource, package in [
    ("corpora/stopwords",  "stopwords"),
    ("corpora/wordnet",    "wordnet"),
    ("corpora/omw-1.4",    "omw-1.4"),
    ("tokenizers/punkt",   "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(package, quiet=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Preprocessing function (the actual "preprocessing" lives here)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_STOPS = set(stopwords.words("english"))
# Keep negations — important even in a spam-style classifier
NEGATIONS = {"not", "no", "nor", "never", "without", "against"}
STOPS = DEFAULT_STOPS - NEGATIONS

lemmatizer = WordNetLemmatizer()


def preprocess(text: str) -> str:
    """The full classical cleaning recipe."""
    text = text.lower()                                       # 1. lowercase
    text = re.sub(r"http\S+|www\.\S+", " ", text)             # 2. strip URLs
    text = re.sub(r"\d+", " ", text)                          # 3. strip digits
    text = re.sub(r"[^a-z\s!?]", " ", text)                   # 4. keep letters + ! ?
    tokens = word_tokenize(text)                              # 5. tokenize
    tokens = [t for t in tokens if t not in STOPS and len(t) > 1]   # 6. stopwords
    tokens = [lemmatizer.lemmatize(t, pos="v") for t in tokens]      # 7. lemmatize
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Toy dataset — spam vs ham
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_TEXTS = [
    # spam
    "FREE money!!! Click here NOW to claim your $1000 prize.",
    "Congratulations! You have won a free iPhone. Click the link!",
    "Lowest price guaranteed! Order today and save 70 percent.",
    "Earn $5000 a week working from home. No experience required!",
    "URGENT: Your account has been compromised. Verify now: http://bit.ly/xx",
    "Claim your free gift card now! Limited time offer.",
    # ham
    "Hey, can we move the meeting from 3pm to 4pm tomorrow?",
    "I really enjoyed the book you lent me. Thanks!",
    "The project deadline is next Friday. Please send your updates.",
    "Mom is making dinner at 7 — let me know if you can make it.",
    "Quick question about the report you sent last week.",
    "Don't forget to bring the documents for the meeting.",
]
TRAIN_LABELS = [1] * 6 + [0] * 6   # 1 = spam, 0 = ham

TEST_TEXTS = [
    "Free vacation prize! Click to claim now!",
    "Are you free for lunch tomorrow?",
    "WINNER!!! You have been selected. Reply YES to claim.",
    "Could you review the draft I sent yesterday?",
]


def section(title: str) -> None:
    print("\n" + "─" * 70)
    print(title)
    print("─" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Show the preprocessing in action
# ─────────────────────────────────────────────────────────────────────────────
def demo_preprocessing():
    section("1. Preprocessing examples")
    for raw in TRAIN_TEXTS[:3] + TRAIN_TEXTS[6:9]:
        print(f"  Raw     : {raw}")
        print(f"  Cleaned : {preprocess(raw)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Train the model
# ─────────────────────────────────────────────────────────────────────────────
def train_and_evaluate():
    section("2. Train TF-IDF + Logistic Regression")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocess,
            ngram_range=(1, 2),
            min_df=1,
        )),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(TRAIN_TEXTS, TRAIN_LABELS)
    train_acc = pipeline.score(TRAIN_TEXTS, TRAIN_LABELS)
    print(f"  Training accuracy: {train_acc:.2%} (tiny dataset — expect ~100%)")

    section("3. Predictions on unseen test sentences")
    labels = ["ham", "spam"]
    preds = pipeline.predict(TEST_TEXTS)
    probs = pipeline.predict_proba(TEST_TEXTS)
    for text, pred, prob in zip(TEST_TEXTS, preds, probs):
        print(f"  Input : {text}")
        print(f"  → {labels[pred].upper():<5s}  (p_spam = {prob[1]:.3f})\n")

    # Inspect the top features the model considers spammy
    section("4. Top 'spam-indicating' features the model learned")
    vec, clf = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    feature_names = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    # 10 most-positive coefficients = most predictive of spam
    top = sorted(zip(coefs, feature_names), reverse=True)[:10]
    for c, name in top:
        print(f"  weight = {c:+.3f}    {name}")


def demo_summary():
    section("5. Summary")
    print("""
  This is the canonical classical NLP pipeline:

      preprocess  →  TfidfVectorizer  →  LogisticRegression

  Notes:
    • Notice how preprocess() is plugged into TfidfVectorizer via
      the `preprocessor=` argument — sklearn calls it on each document.
    • We KEPT negations in the stop-word list so the model can learn
      from phrases like "not free" or "no prize".
    • For real spam detection, you'd use a real dataset (e.g., the
      SMS Spam Collection) and tune hyperparameters with cross-validation.

  For the MODERN equivalent — see example 07 (Transformer pipeline).
""")


if __name__ == "__main__":
    print("=" * 70)
    print("NLP PREPROCESSING — STEP 06: FULL CLASSICAL PIPELINE")
    print("=" * 70)
    demo_preprocessing()
    train_and_evaluate()
    demo_summary()
