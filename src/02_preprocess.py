"""
Step 2: Text preprocessing with spaCy (and NLTK for comparison)
===============================================================

Goal: learn the classic text-cleaning steps, and find out WITH DATA which of them
actually help our TF-IDF model (Step 3). Spoiler: more cleaning is not always better.

What this script does:
    1. Shows what each preprocessing step does to real customer messages.
    2. Compares stemming (NLTK) vs lemmatization (spaCy).
    3. Checks whether removing stopwords destroys meaning ("not", "can't" ...).
    4. Runs a quick experiment: 5 preprocessing versions -> same simple model ->
       which version gives the best validation macro-F1, and how fast is each one?

Run:  python src/02_preprocess.py
(First time only: python -m spacy download en_core_web_sm)

Definitions:
    - Token: one piece of text, usually a word or punctuation mark.
      "I can't pay" -> ["I", "ca", "n't", "pay"]   (spaCy splits "can't" into 2 tokens)
    - Tokenization: cutting text into tokens.
    - Lowercasing: "Card" and "card" become the same word.
    - Stopwords: very common words ("the", "is", "my", "not") that are often removed
      because they appear everywhere. Danger: some carry meaning ("not", "no").
    - Stemming: chop word endings with simple rules: "charges" -> "charg".
      Fast, but the result is often not a real word.
    - Lemmatization: turn a word into its dictionary form using grammar knowledge:
      "charged" -> "charge", "was" -> "be". Slower, but the result is a real word.
    - N-gram: a group of N words in a row. "card arrived" is a 2-gram (bigram).
      Bigrams keep a little word order, which helps with "not received".
    - When NOT to preprocess: transformer models (Step 5) were trained on raw text and
      have their own tokenizer, so we give them the raw message.
"""

import sys
import time

import spacy
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from common import REPORTS_DIR, ROOT_DIR, evaluate_predictions, load_splits, print_header, save_json

sys.path.append(str(ROOT_DIR))   # lets Python find the app/ folder
from app.preprocessing import stem_texts   # the stemming function the deployed model uses

# Load spaCy's small English model.
# disable=["parser", "ner"]: we do not need sentence structure or named entities here,
# and switching them off makes processing much faster.
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Words that flip or change meaning. Removing them can turn
# "I did NOT get my card" into "get card", which means something else.
NEGATION_WORDS = ["not", "no", "n't", "never", "nothing", "cannot", "without", "none"]


def preprocess_many(texts: list[str], remove_stopwords: bool, lemmatize: bool) -> list[str]:
    """Clean many texts with spaCy and return them as plain strings.

    Always: lowercase + drop punctuation and spaces.
    Optional: remove stopwords, lemmatize.

    nlp.pipe(...) processes texts in batches, which is much faster than calling
    nlp(text) one message at a time.
    """
    cleaned_texts = []
    for doc in nlp.pipe(texts, batch_size=256):
        words = []
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            if remove_stopwords and token.is_stop:
                continue
            if lemmatize:
                words.append(token.lemma_.lower())
            else:
                words.append(token.text.lower())
        cleaned_texts.append(" ".join(words))
    return cleaned_texts


train_df, val_df, _ = load_splits()   # "_" = we don't need the test set here

# ---------------------------------------------------------------------------
# 1. What does each step do?
# ---------------------------------------------------------------------------
print_header("1. RAW vs PROCESSED (5 examples)")
examples = [
    "I still haven't received my new card, it was ordered 2 weeks ago!",
    "Why was I charged twice for the same payment?",
    "My top-up didn't go through.",
    "I can't remember my PIN.",
    "The exchange rate you charged was not correct.",
]
for text in examples:
    tokens = [token.text for token in nlp(text)]
    print(f"\nRAW              : {text}")
    print(f"tokens           : {tokens}")
    print(f"lowercase        : {preprocess_many([text], remove_stopwords=False, lemmatize=False)[0]}")
    print(f"- stopwords      : {preprocess_many([text], remove_stopwords=True, lemmatize=False)[0]}")
    print(f"lemmatized       : {preprocess_many([text], remove_stopwords=False, lemmatize=True)[0]}")

# ---------------------------------------------------------------------------
# 2. Stemming vs lemmatization
# ---------------------------------------------------------------------------
print_header("2. STEMMING (NLTK Porter) vs LEMMATIZATION (spaCy)")
stemmer = PorterStemmer()
words = ["charged", "charges", "charging", "transferred", "verification", "was", "better", "cards"]
print(f"{'word':<14}{'stem':<14}{'lemma':<14}")
for word in words:
    lemma = nlp(word)[0].lemma_
    print(f"{word:<14}{stemmer.stem(word):<14}{lemma:<14}")
print("\nStems are often not real words ('verif'); lemmas are dictionary words ('be').")

# ---------------------------------------------------------------------------
# 3. Does stopword removal destroy meaning?
# ---------------------------------------------------------------------------
print_header("3. STOPWORDS vs NEGATION")
negations_in_stoplist = [word for word in NEGATION_WORDS if word in nlp.Defaults.stop_words]
print(f"Negation words that spaCy treats as STOPWORDS: {negations_in_stoplist}")

# How many training messages contain a negation?
has_negation = train_df["text"].str.lower().str.contains(r"\bnot\b|n't|\bno\b|\bnever\b", regex=True)
print(f"Training messages with a negation: {has_negation.sum():,} of {len(train_df):,} "
      f"({has_negation.mean() * 100:.1f}%)")

print("\nExamples (raw -> stopwords removed):")
for text in train_df[has_negation]["text"].head(4):
    cleaned = preprocess_many([text], remove_stopwords=True, lemmatize=False)[0]
    print(f"  {text}")
    print(f"    -> {cleaned}")

# ---------------------------------------------------------------------------
# 4. Experiment: which preprocessing helps TF-IDF + Logistic Regression?
# ---------------------------------------------------------------------------
print_header("4. EXPERIMENT: 5 preprocessing versions, same model, validation macro-F1")
print("(TF-IDF + Logistic Regression is explained fully in Step 3. Here it is just a judge.)")
print("Processing all texts, this takes about two minutes...")

# The spaCy versions we compare: name -> (remove_stopwords, lemmatize)
spacy_versions = {
    "lowercase_only": (False, False),
    "lemmatize_spacy": (False, True),
    "remove_stopwords": (True, False),
    "lemmatize_and_remove_stopwords": (True, True),
}

results = {}
all_version_names = list(spacy_versions) + ["stem_nltk"]
for version_name in all_version_names:
    # Time how long the cleaning of ALL texts takes (speed matters in production too).
    start = time.perf_counter()
    if version_name == "stem_nltk":
        train_texts = stem_texts(train_df["text"].tolist())
        val_texts = stem_texts(val_df["text"].tolist())
    else:
        remove_stopwords, lemmatize = spacy_versions[version_name]
        train_texts = preprocess_many(train_df["text"].tolist(), remove_stopwords, lemmatize)
        val_texts = preprocess_many(val_df["text"].tolist(), remove_stopwords, lemmatize)
    cleaning_seconds = time.perf_counter() - start

    # Same simple model for every version, so only the preprocessing changes.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(C=10, max_iter=2000)),
    ])
    model.fit(train_texts, train_df["intent"])
    predictions = model.predict(val_texts)

    scores = evaluate_predictions(val_df["intent"], predictions)
    scores["cleaning_seconds"] = round(cleaning_seconds, 1)
    results[version_name] = scores
    print(f"  {version_name:<32} accuracy={scores['accuracy']:.4f}   macro_f1={scores['macro_f1']:.4f}"
          f"   cleaning time={cleaning_seconds:.1f}s")

best_version = max(results, key=lambda name: results[name]["macro_f1"])
print(f"\nHighest macro-F1: {best_version}")

# DECISION (Claude's default, review it): use NLTK stemming for the TF-IDF model.
# If stemming is within ~0.5 points of the best, its speed and the fact that it needs no
# spaCy language model in production make it the better engineering choice.
chosen_version = "stem_nltk"
gap = results[best_version]["macro_f1"] - results[chosen_version]["macro_f1"]
print(f"Chosen for production: {chosen_version} "
      f"({gap * 100:.2f} points below the best, much faster, no spaCy model needed on the server)")

save_json({"results": results, "best_version": best_version, "chosen_version": chosen_version},
          REPORTS_DIR / "step2_preprocessing.json")
print("Saved: reports/step2_preprocessing.json")
