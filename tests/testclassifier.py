import os
import sqlite3
import sys
import time

import pandas as pd
from sentence_transformers import SentenceTransformer

# Add the directory to sys.path to import gauss_math and gauss_data
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tribblefis.gauss_math import (
    calculate_gaussian_correlation,
    create_gaussian_membership_dict,
    take_top_features,
)
from tribblefis.gauss_plot import report_figures_of_merit
from sklearn.model_selection import train_test_split

start_time = time.time()

# EMBEDDINGS_MODEL = "jinaai/jina-embeddings-v5-text-small-classification"
EMBEDDINGS_MODEL = "jinaai/jina-embeddings-v2-small-en"
EMBEDDINGS_MODEL = "thenlper/gte-small"
# EMBEDDINGS_MODEL = "jinaai/jina-embeddings-v5-text-nano"
# EMBEDDINGS_MODEL = "jinaai/jina-embeddings-v5-text-nano-classification"
TICKET_DATA = r"C:\work\ai-research\experiments\cw_ticket_classifier\ticket_data2-20k.db"
OUTPUT_CLASS = "ticket_type"

N_GAUSSIANS = -1

# Download from Hugging Face
model = SentenceTransformer(model_name_or_path=EMBEDDINGS_MODEL, truncate_dim=8, trust_remote_code=True)

print(f"Loading data from {TICKET_DATA}...")
conn = sqlite3.connect(TICKET_DATA)
df = pd.read_sql_query("SELECT * FROM tickets", conn)
conn.close()

print("Combining summary and description...")
df["text"] = df["ticket_summary"].fillna("") + " " + df["ticket_description"].fillna("")

df = df[["text", OUTPUT_CLASS]]

# Filter blanks
df = df[df[OUTPUT_CLASS] != ""]

print(f"Encoding {len(df)} documents (this might take a while)...")
# Using 5000 samples for better coverage and more data for GMM
# df_sample = df.sample(n=min(5000, len(df)), random_state=42)
df_sample = df
embeddings = model.encode(df_sample["text"].tolist(), show_progress_bar=True)

print("Creating dataframe with embeddings...")
emb_df = pd.DataFrame(embeddings)
emb_df.columns = [f"dim_{i}" for i in range(emb_df.shape[1])]
emb_df[OUTPUT_CLASS] = df_sample[OUTPUT_CLASS].values

# Filter out classes with very few samples (e.g. less than 10)
class_counts = emb_df[OUTPUT_CLASS].value_counts()
to_keep = class_counts[class_counts >= 10].index
emb_df = emb_df[emb_df[OUTPUT_CLASS].isin(to_keep)]

print("Performing Mixture of Gaussians classification...")

# Split data
X = emb_df.drop(columns=[OUTPUT_CLASS])
y = emb_df[OUTPUT_CLASS]

n_unique = y.nunique()
print(f"Number of unique values in y: {n_unique}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Dataset split: Train={len(X_train)}, Test={len(X_test)}")

# Calculate correlation coefficient between Gaussian distributions using training data
feature_differentiators = calculate_gaussian_correlation(X_train, y_train)

# Take the top-n variables so that the normalized differentiation value encompasses 90-95%
top_n, top_n_todo = take_top_features(feature_differentiators)

print(f"Selected Top-{top_n} Variables ({top_n/len(feature_differentiators):.2%} coverage):")

print("Fitting Gaussian Mixture Model...")
gaussian_memberships = create_gaussian_membership_dict(
    X_train, y_train, top_n_var_names=top_n_todo, n_gaussians=N_GAUSSIANS
)
# Iteratively refine the model until no improvement
max_iterations = 5  # Prevent infinite loops
prev_accuracy = 0.0
iteration = 0
saved_gaussian_memberships = gaussian_memberships

while iteration < max_iterations:
    cm_train, top_confusion_train, confused_data_train = report_figures_of_merit(
        X_train, y_train, gaussian_memberships, n_unique, start_time, top_n_todo, label=f"train_iter{iteration}"
    )

    # Calculate current accuracy
    current_accuracy = cm_train.diagonal().sum() / cm_train.sum()
    print(f"\nIteration {iteration}: Train Accuracy = {current_accuracy:.4f}")

    # Check for improvement
    if iteration > 0 and current_accuracy <= prev_accuracy:
        print(f"No improvement from {prev_accuracy:.4f} to {current_accuracy:.4f}. Stopping.")
        break

    saved_gaussian_memberships = gaussian_memberships
    # If there are no confusions to address, stop
    if not confused_data_train:
        print("No confused pairs to refine. Stopping.")
        break

    # Augment the model with focused training on confused pairs
    for (true_class, confused_class), confusion_data in confused_data_train.items():
        X_local_train, y_local_train = confusion_data["X"], confusion_data["y"]
        new_gaussian_memberships = create_gaussian_membership_dict(
            X_local_train, y_local_train, top_n_var_names=top_n_todo, n_gaussians=N_GAUSSIANS
        )
        # Now, we need to augment the existing gaussian memberships
        gaussian_memberships = gaussian_memberships.augment(new_gaussian_memberships)

    prev_accuracy = current_accuracy
    iteration += 1

print(f"\nFinal model after {iteration} iteration(s) with accuracy {prev_accuracy:.4f}")
report_figures_of_merit(X_test, y_test, saved_gaussian_memberships, n_unique, start_time, top_n_todo, label="test")
