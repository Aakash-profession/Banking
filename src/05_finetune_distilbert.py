"""
Step 5: Fine-tune DistilBERT (Hugging Face Transformers + PyTorch)
==================================================================

Goal: train a real transformer model for our 77 intents, and decide whether its extra
accuracy is worth its extra size and latency compared with Steps 3-4.

What this script does:
    1. Tokenizes the RAW text with DistilBERT's own tokenizer (no spaCy cleaning, see Step 2).
    2. Loads pre-trained DistilBERT and adds a new 77-intent classification "head".
    3. Fine-tunes it with the Hugging Face Trainer for a few epochs.
    4. Prints train loss vs validation loss per epoch (to spot overfitting).
    5. Measures validation accuracy / macro-F1, model size and latency, logs to MLflow.

Run:  python src/05_finetune_distilbert.py
Time: on a laptop CPU (no GPU) about 30-60 minutes for 3 epochs.
      Faster option: Google Colab (free GPU): upload this repo's src/ folder and the
      data/processed/ CSVs, then run the same script there (a few minutes on a GPU).

Definitions:
    - Transformer: a neural network architecture that reads all words of a sentence at once
      and uses "attention" to decide which other words matter for each word.
    - Attention (intuition): in "the card I ordered last week still hasn't arrived",
      the word "arrived" pays attention to "card" to understand WHAT has not arrived.
    - BERT / DistilBERT: transformer models pre-trained on lots of English text to
      understand language. DistilBERT is a smaller, faster copy of BERT ("distilled"):
      ~40% smaller, ~60% faster, keeps ~97% of BERT's quality.
    - BERT vs GPT: BERT reads the whole sentence in both directions -> good at
      UNDERSTANDING (classification). GPT predicts the next word left-to-right -> good at
      GENERATING text.
    - Tokenizer: turns text into the number IDs the model was trained with. It splits rare
      words into pieces: "verification" -> "verification", "unrecognised" -> "un", "##rec", ...
    - Fine-tuning: start from the pre-trained model and continue training ALL its weights
      on our labelled data. Feature extraction (Step 4) instead keeps the model frozen and
      only trains a small classifier on its output.
    - Epoch: one full pass over the training data.
    - Learning rate: how big each weight update is. Too high -> training is unstable;
      too low -> learning is very slow. 5e-5 (0.00005) is a standard value for BERT models.
    - Loss: a number that measures how wrong the model is (lower = better).
    - Overfitting: train loss keeps going down while validation loss goes UP:
      the model is memorising the training data instead of learning general patterns.
"""

import time

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from common import (
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    evaluate_predictions,
    load_splits,
    log_mlflow_run,
    measure_latency_ms,
    print_header,
    save_json,
)

BASE_MODEL = "distilbert-base-uncased"   # "uncased" = it lowercases text itself
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
BATCH_SIZE = 32                          # how many messages per weight update
MAX_LENGTH = 64                          # Step 1: 95% of messages have far fewer words than this
OUTPUT_DIR = MODELS_DIR / "distilbert"
MODEL_NAME = "distilbert"

train_df, val_df, _ = load_splits()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}  (cuda = NVIDIA GPU, cpu = processor only)")

# ---------------------------------------------------------------------------
# 1. Labels -> numbers
# ---------------------------------------------------------------------------
# The model outputs 77 numbers; we need a fixed mapping between positions and intents.
intent_names = sorted(train_df["intent"].unique())
label2id = {name: index for index, name in enumerate(intent_names)}   # "activate_my_card" -> 0
id2label = {index: name for name, index in label2id.items()}           # 0 -> "activate_my_card"

# ---------------------------------------------------------------------------
# 2. Tokenize
# ---------------------------------------------------------------------------
print_header("1. TOKENIZE")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

example = "My card hasn't arrived yet"
print(f"Example text   : {example}")
print(f"Tokens         : {tokenizer.tokenize(example)}")
print(f"Token IDs      : {tokenizer(example)['input_ids']}")
print("([CLS] and [SEP] are special start/end tokens added automatically.)")


def make_dataset(df) -> Dataset:
    """Convert a pandas DataFrame into a tokenized Hugging Face Dataset."""
    dataset = Dataset.from_dict({
        "text": df["text"].tolist(),
        "label": [label2id[name] for name in df["intent"]],
    })

    def tokenize(batch: dict) -> dict:
        # truncation=True: cut messages longer than MAX_LENGTH tokens.
        # No padding here: the data collator pads each batch only as much as needed
        # ("dynamic padding"), which is much faster than padding everything to 64.
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    return dataset.map(tokenize, batched=True)


train_dataset = make_dataset(train_df)
val_dataset = make_dataset(val_df)
print(f"Train examples: {len(train_dataset):,}   Validation examples: {len(val_dataset):,}")

# ---------------------------------------------------------------------------
# 3. Load the pre-trained model with a new 77-class head
# ---------------------------------------------------------------------------
print_header("2. LOAD PRE-TRAINED DISTILBERT")
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(intent_names),
    id2label=id2label,
    label2id=label2id,
)
# A warning that some weights are "newly initialized" is EXPECTED: that is the new
# classification head, which starts random and is learned during fine-tuning.
num_parameters = sum(parameter.numel() for parameter in model.parameters())
print(f"Parameters: {num_parameters / 1e6:.1f} million")


def compute_metrics(eval_prediction) -> dict:
    """Called by the Trainer after each epoch on the validation set."""
    logits, labels = eval_prediction            # logits = raw scores, shape (rows, 77)
    predicted_ids = np.argmax(logits, axis=1)   # highest score = predicted intent
    predicted = [id2label[i] for i in predicted_ids]
    actual = [id2label[i] for i in labels]
    return evaluate_predictions(actual, predicted)


# ---------------------------------------------------------------------------
# 4. Train with the Hugging Face Trainer
# ---------------------------------------------------------------------------
print_header(f"3. FINE-TUNE ({NUM_EPOCHS} epochs)")
training_args = TrainingArguments(
    output_dir=str(MODELS_DIR / "distilbert_checkpoints"),
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=64,
    weight_decay=0.01,          # a little regularization against overfitting
    eval_strategy="epoch",      # evaluate on validation after every epoch
    save_strategy="no",         # don't save checkpoints during training (we save once at the end)
    logging_strategy="epoch",   # record the training loss once per epoch
    report_to="none",           # we log to MLflow ourselves, below
    seed=RANDOM_SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,                         # the tokenizer is saved with the model
    data_collator=DataCollatorWithPadding(tokenizer),   # pads each batch (dynamic padding)
    compute_metrics=compute_metrics,
)

start = time.perf_counter()
trainer.train()
train_time_s = time.perf_counter() - start
print(f"Training time: {train_time_s / 60:.1f} minutes")

# ---------------------------------------------------------------------------
# 5. Train loss vs validation loss per epoch
# ---------------------------------------------------------------------------
print_header("4. LOSS PER EPOCH (overfitting check)")
# log_history is a list of dicts the Trainer recorded: some have "loss" (training),
# some have "eval_loss" (validation).
train_losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
eval_entries = [entry for entry in trainer.state.log_history if "eval_loss" in entry]
print(f"{'epoch':>6}{'train_loss':>12}{'val_loss':>10}{'val_macro_f1':>14}")
for epoch_number, (train_loss, eval_entry) in enumerate(zip(train_losses, eval_entries), start=1):
    print(f"{epoch_number:>6}{train_loss:>12.4f}{eval_entry['eval_loss']:>10.4f}{eval_entry['eval_macro_f1']:>14.4f}")
print("If val_loss rises while train_loss keeps falling, the model starts to overfit.")

# ---------------------------------------------------------------------------
# 6. Final validation scores, size, latency
# ---------------------------------------------------------------------------
print_header("5. FINAL VALIDATION RESULTS + SIZE + LATENCY")
final_eval = trainer.evaluate()
metrics = {
    "accuracy": round(final_eval["eval_accuracy"], 4),
    "macro_f1": round(final_eval["eval_macro_f1"], 4),
    "weighted_f1": round(final_eval["eval_weighted_f1"], 4),
}
print(metrics)

trainer.save_model(str(OUTPUT_DIR))   # saves the model weights + config + tokenizer
model_size_mb = sum(path.stat().st_size for path in OUTPUT_DIR.rglob("*") if path.is_file()) / (1024 * 1024)

model.eval()            # evaluation mode (switches off dropout)
model.to("cpu")         # latency is measured on CPU, like a cheap server would run it


def predict_one(text: str) -> str:
    inputs = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():   # no gradients needed for prediction -> faster
        logits = model(**inputs).logits
    return id2label[int(logits.argmax(dim=1))]


latency = measure_latency_ms(predict_one, val_df["text"].tolist()[:200])
print(f"Model size      : {model_size_mb:.0f} MB")
print(f"Latency (1 msg) : avg {latency['latency_avg_ms']:.1f} ms, p95 {latency['latency_p95_ms']:.1f} ms (CPU)")

results = {
    "model": MODEL_NAME,
    "split": "validation",
    **metrics,
    "train_time_s": round(train_time_s, 1),
    **latency,
    "model_size_mb": round(model_size_mb, 1),
    "epochs": NUM_EPOCHS,
    "device": device,
}
save_json(results, METRICS_DIR / f"{MODEL_NAME}.json")

log_mlflow_run(
    run_name=MODEL_NAME,
    params={
        "model": MODEL_NAME,
        "base_model": BASE_MODEL,
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "device": device,
    },
    metrics={
        "val_accuracy": metrics["accuracy"],
        "val_macro_f1": metrics["macro_f1"],
        "val_weighted_f1": metrics["weighted_f1"],
        "train_time_s": train_time_s,
        "latency_avg_ms": latency["latency_avg_ms"],
        "model_size_mb": model_size_mb,
    },
    # One value per epoch -> MLflow draws the loss curves (train vs validation).
    metrics_per_epoch={
        "train_loss": train_losses,
        "val_loss": [entry["eval_loss"] for entry in eval_entries],
    },
)

print(f"\nSaved model   : models/{MODEL_NAME}/")
print(f"Saved metrics : reports/metrics/{MODEL_NAME}.json  (+ MLflow run)")
