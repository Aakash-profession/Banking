"""
Step 7: Clustering for intent discovery
=======================================

Goal: find NEW intents the system does not know yet, using the calls it is unsure about.

The experiment (my own twist):
    1. HIDE 5 intents: train a TF-IDF model WITHOUT them (it only knows 72 intents).
    2. Send it "incoming traffic" that includes those 5 hidden intents.
    3. Low-confidence messages are collected (in production: the calls sent to humans).
    4. Cluster those messages by meaning (MiniLM embeddings + K-means / HDBSCAN).
    5. Check: did the clusters rediscover the 5 hidden intents? (purity, ARI)
    6. Name each cluster: top keywords, plus an LLM-suggested name if an LLM is configured.

Run:  python src/07_intent_discovery.py   (needs Step 6's threshold)

Definitions:
    - Clustering: grouping items so that similar items end up together, WITHOUT labels.
      (Unsupervised learning: there is no "correct answer" given to the algorithm.)
    - K-means: choose K, place K centre points, give every item to its nearest centre,
      move each centre to the middle of its items, repeat until nothing changes.
      You must choose K yourself. Every item is forced into some cluster.
    - HDBSCAN: finds dense areas of points. It chooses the number of clusters itself
      and marks lonely points as "noise" (label -1) instead of forcing them into a group.
    - Silhouette score (-1 to 1): for each item, "how close am I to my own cluster vs the
      nearest other cluster?" Higher = clearer, better-separated clusters. Needs NO labels,
      so we can use it to choose K.
    - Purity (0 to 1): for each cluster, take its most common TRUE intent; purity = share of
      items that match their cluster's main intent. Needs labels (we have them here because
      we hid the intents ourselves).
    - ARI (Adjusted Rand Index): how much two groupings agree, corrected for luck.
      1 = identical groupings, 0 = no better than random.
    - PCA: squashes 384 numbers down to 2 so we can draw the points on a chart.
"""

import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from common import RANDOM_SEED, REPORTS_DIR, ROOT_DIR, load_json, load_splits, print_header, save_json

sys.path.append(str(ROOT_DIR))   # lets Python find the app/ folder
from app.llm import ask_llm, is_llm_configured
from app.preprocessing import stem_texts

# The 5 intents we pretend the bank has never seen before.
HIDDEN_INTENTS = [
    "apple_pay_or_google_pay",
    "age_limit",
    "lost_or_stolen_phone",
    "terminate_account",
    "card_swallowed",
]
K_VALUES_TO_TRY = range(3, 16)       # K-means: try K = 3 ... 15, keep the best silhouette
HDBSCAN_MIN_CLUSTER_SIZE = 15        # HDBSCAN: a group needs at least this many messages

train_df, val_df, _ = load_splits()
threshold = load_json(REPORTS_DIR / "threshold.json")["chosen_threshold"]

# ---------------------------------------------------------------------------
# 1. Train a model that does NOT know the 5 hidden intents
# ---------------------------------------------------------------------------
print_header("1. TRAIN WITHOUT THE 5 HIDDEN INTENTS")
is_hidden_in_train = train_df["intent"].isin(HIDDEN_INTENTS)
known_train_df = train_df[~is_hidden_in_train]
print(f"Hidden intents : {HIDDEN_INTENTS}")
print(f"Training on    : {len(known_train_df):,} messages, {known_train_df['intent'].nunique()} intents")

# Same recipe as the deployed model (Step 3), just trained on fewer intents.
model = Pipeline([
    ("stem", FunctionTransformer(stem_texts)),
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    ("clf", LogisticRegression(C=10.0, max_iter=2000)),
])
model.fit(known_train_df["text"], known_train_df["intent"])

# ---------------------------------------------------------------------------
# 2. Incoming traffic -> which messages get LOW confidence?
# ---------------------------------------------------------------------------
print_header("2. INCOMING TRAFFIC + LOW-CONFIDENCE DETECTION")
# Traffic = the whole validation set (all 77 intents)
#         + the TRAIN messages of the 5 hidden intents (the model never saw them).
# WHY not the test set: we keep the test set untouched for Step 9.
traffic_df = pd.concat([val_df, train_df[is_hidden_in_train]], ignore_index=True)
traffic_df["is_hidden"] = traffic_df["intent"].isin(HIDDEN_INTENTS)

traffic_confidence = model.predict_proba(traffic_df["text"]).max(axis=1)
traffic_df["flagged"] = traffic_confidence < threshold   # would go to a human

hidden_rows = traffic_df[traffic_df["is_hidden"]]
flagged_df = traffic_df[traffic_df["flagged"]].reset_index(drop=True)

print(f"Traffic size                        : {len(traffic_df):,} messages "
      f"({traffic_df['is_hidden'].sum()} from hidden intents)")
print(f"Threshold (from Step 6)             : {threshold}")
print(f"Hidden-intent messages flagged      : {hidden_rows['flagged'].mean() * 100:.1f}%  "
      "(recall: how many new-intent calls we catch)")
print(f"Flagged messages that ARE hidden    : {flagged_df['is_hidden'].mean() * 100:.1f}%  "
      "(the rest are hard calls from known intents)")
print(f"Flagged pool to cluster             : {len(flagged_df):,} messages")

# ---------------------------------------------------------------------------
# 3. Embed the flagged messages
# ---------------------------------------------------------------------------
print_header("3. EMBED THE FLAGGED MESSAGES (MiniLM)")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
embeddings = encoder.encode(flagged_df["text"].tolist(), batch_size=64, normalize_embeddings=True)
print(f"Embeddings: {embeddings.shape}")

# ---------------------------------------------------------------------------
# 4a. K-means: choose K with the silhouette score
# ---------------------------------------------------------------------------
print_header("4a. K-MEANS: choose K by silhouette score")
best_k = None
best_silhouette = -1.0
for k in K_VALUES_TO_TRY:
    # n_init=10: run K-means 10 times from different random starts and keep the best,
    # because a bad random start can give poor clusters.
    labels = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    print(f"  K = {k:>2}   silhouette = {score:.4f}")
    if score > best_silhouette:
        best_k = k
        best_silhouette = score
print(f"Best K = {best_k} (silhouette {best_silhouette:.4f})")

kmeans_model = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_SEED).fit(embeddings)
kmeans_labels = kmeans_model.labels_                 # cluster number of every message
cluster_centres = kmeans_model.cluster_centers_     # the "middle point" of every cluster

# ---------------------------------------------------------------------------
# 4b. HDBSCAN: finds the number of clusters itself
# ---------------------------------------------------------------------------
print_header("4b. HDBSCAN")
# copy=True: HDBSCAN works on a copy, so our embeddings array is never changed.
hdbscan_labels = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, copy=True).fit_predict(embeddings)
n_hdbscan_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
noise_share = (hdbscan_labels == -1).mean()
print(f"Clusters found : {n_hdbscan_clusters}")
print(f"Noise points   : {noise_share * 100:.1f}% (not put in any cluster)")


# ---------------------------------------------------------------------------
# 5. Evaluate both clusterings against the true intents
# ---------------------------------------------------------------------------
def purity(true_labels: list[str], cluster_labels: np.ndarray) -> float:
    """Share of items whose true intent equals their cluster's most common true intent.
    Noise points (-1) are left out."""
    matched = 0
    counted = 0
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue
        intents_in_cluster = [true_labels[i] for i in range(len(true_labels)) if cluster_labels[i] == cluster_id]
        most_common_count = Counter(intents_in_cluster).most_common(1)[0][1]
        matched += most_common_count
        counted += len(intents_in_cluster)
    return matched / counted if counted else 0.0


print_header("5. HOW GOOD ARE THE CLUSTERS? (vs the true intents)")
true_intents = flagged_df["intent"].tolist()
kmeans_purity = purity(true_intents, kmeans_labels)
kmeans_ari = adjusted_rand_score(true_intents, kmeans_labels)
hdbscan_purity = purity(true_intents, hdbscan_labels)
hdbscan_ari = adjusted_rand_score(true_intents, hdbscan_labels)   # noise counts as one group
print(f"{'method':<10}{'clusters':>9}{'purity':>9}{'ARI':>8}")
print(f"{'K-means':<10}{best_k:>9}{kmeans_purity:>9.3f}{kmeans_ari:>8.3f}")
print(f"{'HDBSCAN':<10}{n_hdbscan_clusters:>9}{hdbscan_purity:>9.3f}{hdbscan_ari:>8.3f}")

# Did each hidden intent end up together in one cluster?
print("\nHidden intent recovery (K-means):")
print(f"  {'hidden intent':<26}{'flagged':>8}{'best cluster':>13}{'caught there':>13}{'cluster purity':>16}")
recovery = []
for intent in HIDDEN_INTENTS:
    rows_of_intent = [i for i in range(len(true_intents)) if true_intents[i] == intent]
    if not rows_of_intent:
        continue
    cluster_counts = Counter(kmeans_labels[i] for i in rows_of_intent)
    best_cluster, count_in_best = cluster_counts.most_common(1)[0]
    cluster_size = int((kmeans_labels == best_cluster).sum())
    share_of_intent = count_in_best / len(rows_of_intent)   # how much of the intent is in that cluster
    share_of_cluster = count_in_best / cluster_size         # how much of the cluster is that intent
    recovery.append({
        "intent": intent,
        "flagged": len(rows_of_intent),
        "best_cluster": int(best_cluster),
        "share_of_intent_in_cluster": round(share_of_intent, 3),
        "cluster_purity_for_intent": round(share_of_cluster, 3),
    })
    print(f"  {intent:<26}{len(rows_of_intent):>8}{best_cluster:>13}"
          f"{share_of_intent * 100:>12.0f}%{share_of_cluster * 100:>15.0f}%")

# ---------------------------------------------------------------------------
# 6. Describe each K-means cluster: keywords + examples + (optional) LLM name
# ---------------------------------------------------------------------------
print_header("6. CLUSTER DESCRIPTIONS (K-means)")
use_llm = is_llm_configured()
print(f"LLM naming: {'ON' if use_llm else 'OFF (no LLM configured, keywords only)'}")

# Keywords: the words with the highest average TF-IDF score inside each cluster.
keyword_vectorizer = TfidfVectorizer(stop_words="english")
keyword_matrix = keyword_vectorizer.fit_transform(flagged_df["text"])
vocabulary = keyword_vectorizer.get_feature_names_out()

cluster_descriptions = []
for cluster_id in range(best_k):
    in_cluster = kmeans_labels == cluster_id
    cluster_texts = flagged_df["text"][in_cluster].tolist()
    mean_scores = np.asarray(keyword_matrix[in_cluster].mean(axis=0)).ravel()
    keywords = [vocabulary[i] for i in mean_scores.argsort()[::-1][:5]]   # top 5, highest first
    main_intent, main_count = Counter(flagged_df["intent"][in_cluster]).most_common(1)[0]

    # The most TYPICAL messages of a cluster are the ones closest to its centre.
    # (Taking the "first 10" could pick odd, untypical messages and mislead the LLM.)
    # Embeddings are normalized, so a dot product with the centre measures closeness.
    closeness_to_centre = embeddings[in_cluster] @ cluster_centres[cluster_id]
    most_typical_first = closeness_to_centre.argsort()[::-1]
    typical_texts = [cluster_texts[i] for i in most_typical_first[:10]]

    llm_name = ""
    if use_llm:
        examples_text = "\n".join(f"- {text}" for text in typical_texts)
        try:
            reply = ask_llm(
                system_prompt="You name groups of bank customer messages. Reply with ONLY a short "
                              "snake_case intent name (2 to 4 words), nothing else.",
                user_prompt=f"Customer messages:\n{examples_text}\n\nIntent name:",
                max_tokens=20,
            )
            llm_name = reply["text"].splitlines()[0].strip().strip("`'\"")
        except Exception as error:   # the LLM is optional: never let it crash the script
            llm_name = f"(LLM failed: {type(error).__name__})"

    cluster_descriptions.append({
        "cluster": cluster_id,
        "size": len(cluster_texts),
        "keywords": keywords,
        "llm_name": llm_name,
        "main_true_intent": main_intent,
        "main_intent_share": round(main_count / len(cluster_texts), 3),
        "typical_examples": typical_texts[:3],
    })
    print(f"\nCluster {cluster_id}  ({len(cluster_texts)} messages)")
    print(f"  keywords        : {', '.join(keywords)}")
    if use_llm:
        print(f"  LLM name        : {llm_name}")
    print(f"  true main intent: {main_intent} ({main_count / len(cluster_texts) * 100:.0f}%)"
          f"{'  <- HIDDEN INTENT' if main_intent in HIDDEN_INTENTS else ''}")
    for text in typical_texts[:2]:
        print(f"  typical example : {text}")

# ---------------------------------------------------------------------------
# 7. Picture: PCA down to 2 dimensions
# ---------------------------------------------------------------------------
points_2d = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(embeddings)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(points_2d[:, 0], points_2d[:, 1], c=kmeans_labels, cmap="tab20", s=10)
plt.colorbar(scatter, label="K-means cluster")
plt.title(f"Low-confidence messages, K-means clusters (K={best_k}), PCA 2D")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "step7_clusters_pca.png", dpi=120)
plt.close()
print("\nSaved chart: reports/step7_clusters_pca.png")

save_json(
    {
        "hidden_intents": HIDDEN_INTENTS,
        "threshold": threshold,
        "hidden_flagged_rate": round(float(hidden_rows["flagged"].mean()), 4),
        "flagged_pool_size": len(flagged_df),
        "flagged_share_hidden": round(float(flagged_df["is_hidden"].mean()), 4),
        "kmeans": {"k": best_k, "silhouette": round(best_silhouette, 4),
                   "purity": round(kmeans_purity, 4), "ari": round(kmeans_ari, 4)},
        "hdbscan": {"clusters": n_hdbscan_clusters, "noise_share": round(float(noise_share), 4),
                    "purity": round(hdbscan_purity, 4), "ari": round(hdbscan_ari, 4)},
        "hidden_intent_recovery": recovery,
        "clusters": cluster_descriptions,
    },
    REPORTS_DIR / "step7_discovery.json",
)
print("Saved: reports/step7_discovery.json")
