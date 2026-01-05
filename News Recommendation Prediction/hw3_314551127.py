"""
Method 1 (Keras): TF-IDF similarity + sentence embedding features + shallow MLP.

Idea:
- Use HashingVectorizer to get sparse TF-IDF; compute cosine similarity (scalar) for user history vs candidate.
- Use sentence-transformers to get dense embeddings; build user embedding by mean of clicked news.
- Feature vector = [tfidf_cosine, dense_cosine] + user_dense + cand_dense + (user*cand).
- Train a small Keras MLP (binary cross-entropy) and predict test impressions.

Usage example:
python tfidf_logreg.py \
  --train_news dataset/train/train_news.tsv \
  --test_news dataset/test/test_news.tsv \
  --train_behaviors dataset/train/train_behaviors.tsv \
  --test_behaviors dataset/test/test_behaviors.tsv \
  --max_train_rows 50000 \
  --output tfidf_keras_submission.csv

Requires: numpy, scipy, scikit-learn, sentence-transformers, tensorflow.
"""

from __future__ import annotations

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import csv
import math
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import tensorflow as tf


def load_news_texts(path: str) -> Dict[str, str]:
    news_text: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            text = (row.get("title", "") + " " + row.get("abstract", "")).strip()
            news_text[row["news_id"]] = text
    return news_text


def build_vectorizer(n_features: int) -> HashingVectorizer:
    return HashingVectorizer(
        n_features=n_features,
        alternate_sign=False,
        lowercase=True,
        norm="l2",
        stop_words="english",
    )


def encode_news(vectorizer: HashingVectorizer, news_text: Dict[str, str]) -> Tuple[Dict[str, int], sparse.csr_matrix]:
    ids: List[str] = list(news_text.keys())
    corpus: List[str] = [news_text[nid] for nid in ids]
    matrix = vectorizer.transform(corpus)
    id_to_row = {nid: i for i, nid in enumerate(ids)}
    return id_to_row, matrix.tocsr()


def build_sentence_embeddings(
    news_text: Dict[str, str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 512,
) -> Tuple[Dict[str, np.ndarray], int]:
    model = SentenceTransformer(model_name)
    ids = list(news_text.keys())
    vectors: Dict[str, np.ndarray] = {}
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size]
        texts = [news_text[i] for i in batch_ids]
        embs = model.encode(texts, batch_size=min(batch_size, 64), show_progress_bar=False, normalize_embeddings=True)
        for nid, emb in zip(batch_ids, embs):
            vectors[nid] = emb.astype(np.float32)
    dim = next(iter(vectors.values())).shape[0] if vectors else 0
    return vectors, dim


def parse_clicked(raw: str) -> List[str]:
    if not raw or raw == "nan":
        return []
    return [tok for tok in raw.strip().split(" ") if tok]


def parse_impressions(raw: str) -> List[Tuple[str, int]]:
    if not raw:
        return []
    result: List[Tuple[str, int]] = []
    for tok in raw.strip().split(" "):
        if not tok:
            continue
        if "-" in tok:
            news_id, label = tok.split("-")
            result.append((news_id, int(label)))
        else:
            result.append((tok, -1))  # unlabeled (test)
    return result


def build_user_vector(clicked: List[str], id_to_row: Dict[str, int], news_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    rows = [id_to_row[nid] for nid in clicked if nid in id_to_row]
    if not rows:
        return sparse.csr_matrix((1, news_matrix.shape[1]))
    sub = news_matrix[rows]
    mean_vec = sub.mean(axis=0)
    return sparse.csr_matrix(mean_vec)


def build_user_dense(clicked: List[str], dense_lookup: Dict[str, np.ndarray], dim: int) -> np.ndarray:
    vecs = [dense_lookup[nid] for nid in clicked if nid in dense_lookup]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(np.stack(vecs), axis=0)


def build_feature_and_label_arrays(
    behaviors_path: str,
    id_to_row: Dict[str, int],
    news_matrix: sparse.csr_matrix,
    dense_lookup: Dict[str, np.ndarray],
    dense_dim: int,
    limit: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    feats: List[np.ndarray] = []
    labels: List[int] = []
    with open(behaviors_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            clicked = parse_clicked(row["clicked_news"])
            user_sparse = build_user_vector(clicked, id_to_row, news_matrix)
            user_dense = build_user_dense(clicked, dense_lookup, dense_dim)
            impressions = parse_impressions(row["impressions"])
            for news_id, label in impressions:
                if label < 0:
                    continue  # skip unlabeled in training
                if news_id not in id_to_row:
                    continue
                cand_sparse = news_matrix[id_to_row[news_id]]
                tfidf_sim = cosine_similarity(user_sparse, cand_sparse).ravel()[0] if user_sparse.nnz and cand_sparse.nnz else 0.0
                cand_dense = dense_lookup.get(news_id, np.zeros(dense_dim, dtype=np.float32))
                dense_sim = float(np.dot(user_dense, cand_dense)) if dense_dim > 0 else 0.0
                feat = np.concatenate(
                    [
                        np.array([tfidf_sim, dense_sim], dtype=np.float32),
                        user_dense,
                        cand_dense,
                        user_dense * cand_dense,
                    ]
                )
                feats.append(feat)
                labels.append(label)
    if not feats:
        raise ValueError("No training samples were built; check input paths.")
    X = np.stack(feats)
    y = np.array(labels, dtype=np.float32)
    return X, y


def build_keras_mlp(input_dim: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(input_dim,), name="features")
    x = tf.keras.layers.Dense(256, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["AUC"])
    return model


def predict(
    behaviors_path: str,
    id_to_row: Dict[str, int],
    news_matrix: sparse.csr_matrix,
    dense_lookup: Dict[str, np.ndarray],
    dense_dim: int,
    model: tf.keras.Model,
    output_path: str,
) -> None:
    header = ["id"] + [f"p{i}" for i in range(1, 16)]
    with open(behaviors_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        writer = csv.writer(fout)
        writer.writerow(header)

        for row in reader:
            clicked = parse_clicked(row["clicked_news"])
            user_sparse = build_user_vector(clicked, id_to_row, news_matrix)
            user_dense = build_user_dense(clicked, dense_lookup, dense_dim)
            impressions = parse_impressions(row["impressions"])
            scores: List[float] = []
            for news_id, _ in impressions:
                cand_sparse = news_matrix[id_to_row.get(news_id, -1)] if news_id in id_to_row else sparse.csr_matrix((1, news_matrix.shape[1]))
                tfidf_sim = cosine_similarity(user_sparse, cand_sparse).ravel()[0] if user_sparse.nnz and cand_sparse.nnz else 0.0
                cand_dense = dense_lookup.get(news_id, np.zeros(dense_dim, dtype=np.float32))
                dense_sim = float(np.dot(user_dense, cand_dense)) if dense_dim > 0 else 0.0
                feat = np.concatenate(
                    [
                        np.array([tfidf_sim, dense_sim], dtype=np.float32),
                        user_dense,
                        cand_dense,
                        user_dense * cand_dense,
                    ]
                )
                prob = float(model.predict(feat.reshape(1, -1), verbose=0)[0, 0])
                scores.append(0.0 if math.isnan(prob) else prob)
            while len(scores) < 15:
                scores.append(0.0)
            writer.writerow([row["id"]] + scores[:15])


def main() -> None:
    parser = argparse.ArgumentParser(description="TF-IDF + Logistic Regression click predictor.")
    parser.add_argument("--train_news", type=str, default="dataset/train/train_news.tsv")
    parser.add_argument("--test_news", type=str, default="dataset/test/test_news.tsv")
    parser.add_argument("--train_behaviors", type=str, default="dataset/train/train_behaviors.tsv")
    parser.add_argument("--test_behaviors", type=str, default="dataset/test/test_behaviors.tsv")
    parser.add_argument("--n_features", type=int, default=2**18)
    parser.add_argument("--max_train_rows", type=int, default=30000, help="Number of behavior rows to consume for training (None for all).")
    parser.add_argument("--sentence_model", type=str, default="sentence-transformers/all-MiniLM-L12-v2")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--output", type=str, default="tfidf_keras_submission.csv")
    args = parser.parse_args()

    print('Built with CUDA:', tf.test.is_built_with_cuda())
    print('Physical GPUs:', tf.config.list_physical_devices('GPU'))
    print('Logical GPUs:', tf.config.list_logical_devices('GPU'))

    print("Loading news text...")
    news_text = {}
    news_text.update(load_news_texts(args.train_news))
    news_text.update(load_news_texts(args.test_news))

    print("Vectorizing news...")
    vectorizer = build_vectorizer(args.n_features)
    id_to_row, news_matrix = encode_news(vectorizer, news_text)
    print(f"Encoded {len(id_to_row)} news items into {news_matrix.shape[1]}-D vectors.")

    print(f"Building sentence embeddings with {args.sentence_model} ...")
    dense_lookup, dense_dim = build_sentence_embeddings(news_text, model_name=args.sentence_model)
    print(f"Built dense embeddings for {len(dense_lookup)} news items, dim={dense_dim}")

    print("Building training data...")
    X, y = build_feature_and_label_arrays(
        behaviors_path=args.train_behaviors,
        id_to_row=id_to_row,
        news_matrix=news_matrix,
        dense_lookup=dense_lookup,
        dense_dim=dense_dim,
        limit=args.max_train_rows if args.max_train_rows > 0 else None,
    )
    print(f"Training samples: {X.shape[0]}, feature dim: {X.shape[1]}")

    print("Training Keras MLP...")
    model = build_keras_mlp(input_dim=X.shape[1])
    model.fit(X, y, batch_size=args.batch_size, epochs=args.epochs, verbose=1, validation_split=0.05)

    print("Predicting on test set...")
    predict(args.test_behaviors, id_to_row, news_matrix, dense_lookup, dense_dim, model, args.output)
    print(f"Saved submission to {args.output}")


if __name__ == "__main__":
    main()
