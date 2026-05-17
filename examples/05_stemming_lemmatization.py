"""
Example 05 — Stemming vs Lemmatization
======================================

A side-by-side comparison of three stemmers (Porter, Snowball, Lancaster)
and the WordNet lemmatizer, including POS-aware lemmatization.

Run:
    python examples/05_stemming_lemmatization.py
"""

import nltk


# Make sure required corpora are present
for resource, package in [
    ("corpora/wordnet",                          "wordnet"),
    ("corpora/omw-1.4",                          "omw-1.4"),
    ("taggers/averaged_perceptron_tagger",       "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng",   "averaged_perceptron_tagger_eng"),
    ("tokenizers/punkt",                         "punkt"),
    ("tokenizers/punkt_tab",                     "punkt_tab"),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(package, quiet=True)


from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag


porter    = PorterStemmer()
snowball  = SnowballStemmer("english")
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()


def section(title: str) -> None:
    print("\n" + "─" * 78)
    print(title)
    print("─" * 78)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Side-by-side comparison
# ─────────────────────────────────────────────────────────────────────────────
def demo_side_by_side():
    section("1. Three stemmers vs. WordNet lemmatizer")

    words = [
        "playing", "played", "plays",
        "studies", "studying", "studied",
        "running", "ran", "runs",
        "better", "best",
        "was", "were", "is",
        "mice", "geese", "feet",
        "connection", "connections", "connecting", "connected",
        "caring", "cared", "cares",
    ]

    header = f"  {'Word':<14s} {'Porter':<12s} {'Snowball':<12s} {'Lancaster':<12s} {'Lemma':<12s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for w in words:
        row = (
            f"  {w:<14s}"
            f" {porter.stem(w):<12s}"
            f" {snowball.stem(w):<12s}"
            f" {lancaster.stem(w):<12s}"
            f" {lemmatizer.lemmatize(w):<12s}"
        )
        print(row)

    print("\n  → Notice 'studi' (from stemming) is not a real English word.")
    print("  → Notice the default lemmatizer doesn't reduce verbs like 'running'.")
    print("  → Why? It needs Part-of-Speech information. See the next demo.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. POS-aware lemmatization
# ─────────────────────────────────────────────────────────────────────────────
def to_wordnet_pos(tag: str) -> str:
    """Map Penn-Treebank POS tags to WordNet POS tags."""
    if tag.startswith("V"):
        return "v"   # verb
    if tag.startswith("J"):
        return "a"   # adjective
    if tag.startswith("R"):
        return "r"   # adverb
    return "n"       # noun (default)


def lemmatize_sentence(text: str) -> list[str]:
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    return [lemmatizer.lemmatize(tok, to_wordnet_pos(tag)) for tok, tag in tagged]


def demo_pos_aware():
    section("2. POS-aware lemmatization — the right way")

    sentences = [
        "She was running through the forest.",
        "The mice were squeaking loudly.",
        "Better solutions need careful thinking.",
    ]
    for s in sentences:
        print(f"  Input  : {s}")
        print(f"  Naive  : {[lemmatizer.lemmatize(t) for t in word_tokenize(s)]}")
        print(f"  POS    : {lemmatize_sentence(s)}")
        print()
    print("  → POS-aware lemmatization correctly reduces 'running' → 'run', "
          "'were' → 'be', etc.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Where stemming wins and where it loses
# ─────────────────────────────────────────────────────────────────────────────
def demo_when_to_use_what():
    section("3. Stemming vs Lemmatization — when to pick which")

    table = [
        ("Speed",                    "FAST",        "Slower"),
        ("Output is a real word",    "Often no",    "Yes"),
        ("Accuracy",                 "Low",         "High"),
        ("Needs POS tagging",        "No",          "For best results"),
        ("Disk / runtime cost",      "Tiny",        "Loads WordNet"),
        ("Good for",                 "Search, IR",  "Classification, IE"),
    ]

    print(f"  {'Property':<28s} {'Stemming':<18s} {'Lemmatization':<18s}")
    print("  " + "-" * 64)
    for prop, s, l in table:
        print(f"  {prop:<28s} {s:<18s} {l:<18s}")


def demo_recommendation():
    section("4. Recommendation")
    print("""
  ✅ Use STEMMING when:
       - You need speed at scale (search engines, IR)
       - You don't care if the output is a real word
       - You're processing huge corpora

  ✅ Use LEMMATIZATION when:
       - You want real, readable output
       - You're doing classification, topic modeling, or NER preprocessing
       - You have POS tags available

  ❌ Use NEITHER when:
       - You're using a Transformer (BERT/GPT) — its tokenizer is enough
       - You're feeding text to an LLM
       - You're doing translation or text generation
""")


if __name__ == "__main__":
    print("=" * 78)
    print("NLP PREPROCESSING — STEP 05: STEMMING vs LEMMATIZATION")
    print("=" * 78)
    demo_side_by_side()
    demo_pos_aware()
    demo_when_to_use_what()
    demo_recommendation()
