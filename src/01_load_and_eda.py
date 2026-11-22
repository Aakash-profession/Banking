"""
Step 1: Load Banking77 + Exploratory Data Analysis (EDA)
========================================================

Goal: understand the data BEFORE training any model.

What this script does:
    1. Downloads Banking77 (the original CSV files from PolyAI's GitHub) once, and caches it.
    2. Checks class balance: how many examples does each intent have?
    3. Checks text length: how long are customer messages?
    4. Checks duplicates and train/test overlap (a leakage check).
    5. Shows examples of intents that are easy to confuse.
    6. Creates a stratified validation split from the training data and saves all 3 splits.

Run:  python src/01_load_and_eda.py

Definitions:
    - EDA (Exploratory Data Analysis): looking at the data with counts and charts to find
      problems and surprises before modelling.
    - Intent: what the customer wants, e.g. "card_arrival" = "where is my card?".
    - Class balance: whether every intent has roughly the same number of examples.
      If one intent had 1,000 examples and another 10, a model could ignore the small one.
    - Train / validation / test split:
        train      -> the model learns from this
        validation -> we compare models and tune settings on this
        test       -> touched ONCE at the very end (Step 9) for the honest final score
      Everyday example: practice questions (train), mock exam (validation),
      the real exam you only sit once (test).
    - Stratified split: the split keeps the SAME share of each intent in both parts.
      If card_arrival is 1.5% of train, it is also ~1.5% of validation.
"""

import matplotlib
matplotlib.use("Agg")   # "Agg" = save charts to files without opening a window
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from common import PROCESSED_DIR, RANDOM_SEED, RAW_DIR, REPORTS_DIR, print_header, save_json

# The original Banking77 files published by PolyAI (columns: text, category).
TRAIN_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
TEST_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

# DECISION (Claude's default, review it): 10% of train becomes validation.
# Train has ~10,000 rows, so validation gets ~1,000 rows (~13 per intent).
# More validation = more reliable scores but less training data. 10% is a common balance.
VALIDATION_SIZE = 0.10

# Pairs of intents that mean almost the same thing. A model will find these hard.
CONFUSING_PAIRS = [
    ("card_arrival", "card_delivery_estimate"),
    ("lost_or_stolen_card", "compromised_card"),
    ("top_up_failed", "top_up_reverted"),
    ("pending_transfer", "transfer_not_received_by_recipient"),
    ("card_payment_wrong_exchange_rate", "wrong_exchange_rate_for_cash_withdrawal"),
]


def download_if_missing(url: str, file_name: str) -> pd.DataFrame:
    """Download a CSV the first time; after that, read the saved copy.

    WHY cache: downloading every run is slow and breaks when you are offline.
    """
    local_path = RAW_DIR / file_name
    if local_path.exists():
        print(f"Reading cached file: {local_path}")
        return pd.read_csv(local_path)

    print(f"Downloading {url}")
    df = pd.read_csv(url)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(local_path, index=False)
    return df


# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
print_header("1. LOAD BANKING77")
full_train_df = download_if_missing(TRAIN_URL, "train.csv")
test_df = download_if_missing(TEST_URL, "test.csv")

# Rename "category" to "intent" because that is the word used in IVR systems.
full_train_df = full_train_df.rename(columns={"category": "intent"})
test_df = test_df.rename(columns={"category": "intent"})

print(f"Train rows : {len(full_train_df):,}")
print(f"Test rows  : {len(test_df):,}")
print(f"Intents    : {full_train_df['intent'].nunique()}")
print("\nFirst 5 rows:")
print(full_train_df.head().to_string())

# ---------------------------------------------------------------------------
# 2. Class balance
# ---------------------------------------------------------------------------
print_header("2. CLASS BALANCE (examples per intent in train)")
counts = full_train_df["intent"].value_counts()   # intent -> number of rows, biggest first

print(f"Smallest intent : {counts.idxmin()} ({counts.min()} examples)")
print(f"Largest intent  : {counts.idxmax()} ({counts.max()} examples)")
print(f"Average         : {counts.mean():.1f} examples per intent")
# Imbalance ratio = largest / smallest. Close to 1 = balanced. Above ~10 = a real problem.
print(f"Imbalance ratio : {counts.max() / counts.min():.1f}x")
print("\n5 smallest intents:")
print(counts.tail(5).to_string())

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
plt.figure(figsize=(14, 5))
counts.plot(kind="bar")
plt.title("Banking77: training examples per intent")
plt.ylabel("Number of examples")
plt.xticks(fontsize=6)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "step1_class_counts.png", dpi=120)
plt.close()
print("\nSaved chart: reports/step1_class_counts.png")

# Test set: PolyAI made it perfectly balanced (40 per intent). Let's confirm.
test_counts = test_df["intent"].value_counts()
print(f"Test set: min {test_counts.min()}, max {test_counts.max()} examples per intent")

# ---------------------------------------------------------------------------
# 3. Text length
# ---------------------------------------------------------------------------
print_header("3. TEXT LENGTH (number of words per message)")
# str.split() cuts a sentence at spaces: "my card is lost" -> 4 words.
word_counts = full_train_df["text"].str.split().str.len()

print(f"Average words : {word_counts.mean():.1f}")
print(f"Median words  : {word_counts.median():.0f}")
print(f"95% of messages have <= {word_counts.quantile(0.95):.0f} words")
print(f"Longest       : {word_counts.max()} words")
print(f"Shortest      : {word_counts.min()} words")
print(f"\nShortest message example : {full_train_df.loc[word_counts.idxmin(), 'text']!r}")
# WHY this matters: short texts mean little context, and it tells us Step 5 can use a
# small max_length for DistilBERT (faster training).

plt.figure(figsize=(8, 4))
plt.hist(word_counts, bins=40)
plt.title("Banking77: words per customer message")
plt.xlabel("Number of words")
plt.ylabel("Number of messages")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "step1_text_lengths.png", dpi=120)
plt.close()
print("Saved chart: reports/step1_text_lengths.png")

# ---------------------------------------------------------------------------
# 4. Duplicates and train/test overlap (leakage check)
# ---------------------------------------------------------------------------
print_header("4. DUPLICATES + TRAIN/TEST OVERLAP")
duplicates_in_train = full_train_df["text"].duplicated().sum()
# Data leakage = the model sees test information during training, so its test score
# looks better than it really is. Identical sentences in train AND test are one form.
overlap_with_test = test_df["text"].isin(full_train_df["text"]).sum()

print(f"Duplicate texts inside train   : {duplicates_in_train}")
print(f"Test texts that also are in train: {overlap_with_test}")

# ---------------------------------------------------------------------------
# 5. Confusing intents
# ---------------------------------------------------------------------------
print_header("5. CONFUSING INTENT PAIRS (2 examples each)")
for intent_a, intent_b in CONFUSING_PAIRS:
    print(f"\n--- {intent_a}  vs  {intent_b} ---")
    for intent in (intent_a, intent_b):
        examples = full_train_df[full_train_df["intent"] == intent]["text"].head(2)
        for text in examples:
            print(f"  [{intent}] {text}")

# ---------------------------------------------------------------------------
# 6. Create the validation split (stratified) and save all splits
# ---------------------------------------------------------------------------
print_header("6. TRAIN / VALIDATION / TEST SPLIT")
# train_test_split shuffles the rows and cuts them in two.
#   test_size=0.10            -> 10% goes to the second part (our validation set)
#   stratify=intent column    -> keep the same share of each intent in both parts
#   random_state=RANDOM_SEED  -> the same split every time we run
train_df, val_df = train_test_split(
    full_train_df,
    test_size=VALIDATION_SIZE,
    stratify=full_train_df["intent"],
    random_state=RANDOM_SEED,
)

print(f"Train      : {len(train_df):,} rows")
print(f"Validation : {len(val_df):,} rows")
print(f"Test       : {len(test_df):,} rows (untouched until Step 9)")

# Check that stratification worked: compare each intent's share in train vs validation.
train_share = train_df["intent"].value_counts(normalize=True)   # normalize=True -> fractions
val_share = val_df["intent"].value_counts(normalize=True)
biggest_gap = (train_share - val_share).abs().max()
print(f"Largest difference in an intent's share (train vs val): {biggest_gap * 100:.2f} percentage points")
print(f"Every intent present in validation? {val_df['intent'].nunique() == 77}")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
print("\nSaved: data/processed/train.csv, val.csv, test.csv")

save_json(
    {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "num_intents": int(full_train_df["intent"].nunique()),
        "min_examples_per_intent": int(counts.min()),
        "max_examples_per_intent": int(counts.max()),
        "avg_words": round(float(word_counts.mean()), 2),
        "p95_words": float(word_counts.quantile(0.95)),
        "max_words": int(word_counts.max()),
        "duplicates_in_train": int(duplicates_in_train),
        "test_texts_also_in_train": int(overlap_with_test),
    },
    REPORTS_DIR / "step1_eda_summary.json",
)
print("Saved: reports/step1_eda_summary.json")
