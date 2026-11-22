"""
preprocessing.py: the text cleaning used by the TF-IDF intent model, in training AND in the API.

What it does: lowercase -> split into words -> stem each word (NLTK Porter stemmer).
    "I was charged twice for my top-ups" -> "i wa charg twice for my top up"

WHY stemming (decided in Step 2, with data):
    - lowercase only        -> validation macro-F1 0.887
    - spaCy lemmatization   -> 0.905
    - NLTK Porter stemming  -> 0.904, about 8x faster, and needs no 12 MB spaCy language model
    - removing stopwords    -> 0.840 (much worse: "didn't go through" loses its meaning)
  Stemming gives almost the same gain as lemmatization while keeping the deployed app light.

WHY this file lives in app/ (not src/):
    The deployed model must clean text EXACTLY like training did. If training used stemming
    but the API did not, the model would see different words in production and silently get
    worse ("training-serving skew"). So this function is placed INSIDE the scikit-learn
    Pipeline (Step 3): the saved model file remembers to call it, in training and in the API.
"""

import re

from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()


def stem_text(text: str) -> str:
    """Clean ONE message: lowercase, keep only words/numbers, stem every word."""
    # [a-z0-9']+ = runs of letters, digits or apostrophes. Punctuation is dropped.
    words = re.findall(r"[a-z0-9']+", text.lower())
    stemmed_words = [_stemmer.stem(word) for word in words]
    return " ".join(stemmed_words)


def stem_texts(texts) -> list[str]:
    """Clean MANY messages. This is the function the Pipeline calls (via FunctionTransformer)."""
    return [stem_text(text) for text in texts]
