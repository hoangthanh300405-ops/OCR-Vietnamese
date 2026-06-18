from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import faiss
import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.features import create_vector, load_pickle, l2_normalize


def voting_from_results(distances, indices, labels):
    vote = Counter()
    for idx, score in zip(indices, distances):
        vote[labels[idx]] += float(score)
    top5 = vote.most_common(5)
    best_label, best_score = top5[0]
    return best_label, best_score, top5, distances, indices


def voting_word(vector, index, labels, k=7):
    distances, indices = index.search(vector, k)
    return voting_from_results(distances[0], indices[0], labels)


def load_query_models(
    codebook_path="./models/codebook.pkl",
    pca_path="./models/pca_model.pkl",
):
    codebook = load_pickle(codebook_path)
    pca_model = load_pickle(pca_path)
    return codebook, pca_model


def create_query_vector(image_path, codebook, pca_model):
    vector = create_vector(image_path, codebook=codebook, pca_model=pca_model)
    vector = np.asarray([vector], dtype=np.float32)
    return l2_normalize(vector, axis=1).astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize one cropped word with HOG+LBP+dense-SIFT PCA retrieval")
    parser.add_argument("--image", default="./Vietnam/test_word/20151208_0146_7105_1_tg_0_2_13.png")
    parser.add_argument("--index", default="./models/vector_database.index")
    parser.add_argument("--labels", default="./models/word_labels.npy")
    parser.add_argument("--names", default="./models/file_names.npy")
    parser.add_argument("--codebook", default="./models/codebook.pkl")
    parser.add_argument("--pca", default="./models/pca_model.pkl")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        sys.exit(1)

    index = faiss.read_index(args.index)
    labels = np.load(args.labels, allow_pickle=True)
    names = np.load(args.names, allow_pickle=True)
    codebook, pca_model = load_query_models(args.codebook, args.pca)

    vector = create_query_vector(image_path, codebook, pca_model)
    if vector.shape[1] != index.d:
        raise ValueError(f"Query dim {vector.shape[1]} does not match index dim {index.d}. Rebuild the index.")

    best_label, best_score, top5, distances, indices = voting_word(vector, index, labels, args.k)

    print(f"Query image     : {image_path}")
    print(f"Best prediction : {best_label} (score: {best_score:.4f})")
    print()
    print(f"{'Rank':<4} {'Vote':<10} {'Distance':<10} {'Label':<20} {'Filename'}")
    print("-" * 80)
    for rank, ((lbl, vote), dist, idx) in enumerate(zip(top5, distances, indices), 1):
        print(f"{rank:<4} {vote:.4f}     {dist:.4f}     {lbl:<20} {names[idx]}")
