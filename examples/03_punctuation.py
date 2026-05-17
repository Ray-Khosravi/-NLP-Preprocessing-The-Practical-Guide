"""
Example 03 — Removing Punctuation
=================================

Aggressive removal vs selective removal vs leaving it alone.
Shows how punctuation carries real signal for some tasks.

Run:
    python examples/03_punctuation.py
"""

import re
import string


SAMPLES = [
    "Wait... you really liked it?!",
    "This is amazing!!! I can't believe it.",
    "Really?",
    "Really.",
    "print(\"Hello, world!\")",
    "Email me at john.doe@example.com.",
]


def section(title: str) -> None:
    print("\n" + "─" * 70)
    print(title)
    print("─" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Aggressive removal — strip everything that isn't a letter or space
# ─────────────────────────────────────────────────────────────────────────────
def aggressive_remove(text: str) -> str:
    return re.sub(r"[^a-zA-Z\s]", "", text)


# ─────────────────────────────────────────────────────────────────────────────
# 2. string.punctuation — the textbook way
# ─────────────────────────────────────────────────────────────────────────────
def textbook_remove(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Smart removal — keep ! ? . because they carry meaning
# ─────────────────────────────────────────────────────────────────────────────
def smart_remove(text: str) -> str:
    # Keep word chars, whitespace, and the three semantic punctuation marks.
    return re.sub(r"[^\w\s!?.]", " ", text).strip()


def demo_three_strategies():
    section("1. Three strategies, side by side")
    for text in SAMPLES:
        print(f"  Original   : {text}")
        print(f"  Aggressive : {aggressive_remove(text)}")
        print(f"  Textbook   : {textbook_remove(text)}")
        print(f"  Smart      : {smart_remove(text)}")
        print()


def demo_signal_loss():
    section("2. Where aggressive removal destroys meaning")

    pairs = [
        ("Really?",         "Really."),
        ("Good!!!",         "Good"),
        ("Wait, what?!",    "Wait what"),
        ("You did WHAT?!",  "You did WHAT"),
    ]

    print("  Pair                          | After aggressive removal")
    print("  ------------------------------|------------------------------")
    for a, b in pairs:
        ra, rb = aggressive_remove(a), aggressive_remove(b)
        marker = "  ← same string!" if ra == rb else ""
        print(f"  {a:<29s} | {ra}{marker}")
        print(f"  {b:<29s} | {rb}")
        print("  " + "-" * 60)

    print("\n  → Sentiment & intent signal vanishes when ! and ? are removed.")


def demo_code_text():
    section("3. The code case — never strip punctuation from code")
    code = 'def greet(name): print(f"Hello, {name}!")'
    print(f"  Original   : {code}")
    print(f"  Aggressive : {aggressive_remove(code)}")
    print("\n  → Removing punctuation turns valid code into nonsense.")
    print("  → If you process code-related text, keep punctuation.")


def demo_recommendation():
    section("4. Recommendation")
    print("""
  ✅ Aggressive removal:
       - Topic modeling on long documents
       - Keyword extraction
       - Cleaning text for word-frequency analysis

  ⚠️  Smart removal (keep ! ? .):
       - Sentiment analysis
       - Tweet / social-media classification
       - Anywhere intensity or sentence type matters

  ❌ Don't remove punctuation:
       - Question Answering (the '?' is the signal)
       - Code-related NLP
       - Transformer / LLM inputs
       - Machine translation
""")


if __name__ == "__main__":
    print("=" * 70)
    print("NLP PREPROCESSING — STEP 03: PUNCTUATION REMOVAL")
    print("=" * 70)
    demo_three_strategies()
    demo_signal_loss()
    demo_code_text()
    demo_recommendation()
