#!/usr/bin/env bash
# One-shot setup for the NLP Preprocessing Guide.
# Creates a virtual environment, installs dependencies, and pulls down the
# required NLTK corpora and spaCy model.

set -e

echo "==> Creating virtual environment in .venv/"
python3 -m venv .venv

echo "==> Activating environment"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip --quiet

echo "==> Installing Python packages from requirements.txt"
pip install -r requirements.txt --quiet

echo "==> Downloading NLTK data"
python -m nltk.downloader \
    stopwords punkt punkt_tab wordnet omw-1.4 \
    averaged_perceptron_tagger averaged_perceptron_tagger_eng

echo "==> Downloading spaCy small English model (optional)"
python -m spacy download en_core_web_sm || \
    echo "   (spaCy model download failed — examples don't strictly need it.)"

echo ""
echo "✅ Setup complete. Try:"
echo "     source .venv/bin/activate"
echo "     python examples/01_lowercase.py"
