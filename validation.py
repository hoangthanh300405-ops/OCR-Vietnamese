from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.database import load_labels
from src.features import create_vector, load_pickle, l2_normalize
from src.recognition import voting_from_results


def validate(
    index_path="./models/vector_database.index",
    labels_path="./models/word_labels.npy",
    names_path="./models/file_names.npy",
    codebook_path="./models/codebook.pkl",
    pca_path="./models/pca_model.pkl",
    validation_dir="./Vietnam/validation_word",
    csv_path="./Vietnam/validation_word.csv",
    k_values=(1, 5, 10),
    batch_size=256,
    max_samples: int | None = None,
):
    index = faiss.read_index(index_path)
    labels = np.load(labels_path, allow_pickle=True)
    _names = np.load(names_path, allow_pickle=True)
    labels_dict = load_labels(csv_path)
    codebook = load_pickle(codebook_path)
    pca_model = load_pickle(pca_path)

    total = 0
    correct = 0
    precision = {k: 0.0 for k in k_values}
    recall = {k: 0.0 for k in k_values}
    label_counts = Counter(labels)

    file_list = sorted(f for f in os.listdir(validation_dir) if f.lower().endswith(".png"))
    if max_samples is not None:
        file_list = file_list[:max_samples]
    max_k = max(k_values)

    for start in tqdm(range(0, len(file_list), batch_size), desc="Validating"):
        batch_files = file_list[start:start + batch_size]
        vectors = []
        true_labels = []
        for fname in batch_files:
            img_id = Path(fname).stem
            true_label = labels_dict.get(img_id)
            if true_label is None:
                continue
            path = Path(validation_dir) / fname
            vectors.append(create_vector(path, codebook=codebook, pca_model=pca_model))
            true_labels.append(true_label)

        if not vectors:
            continue

        queries = l2_normalize(np.asarray(vectors, dtype=np.float32), axis=1).astype(np.float32)
        if queries.shape[1] != index.d:
            raise ValueError(f"Query dim {queries.shape[1]} does not match index dim {index.d}. Rebuild the index.")

        distances, indices = index.search(queries, max_k)
        for row_idx, true_label in enumerate(true_labels):
            total += 1
            pred_label, _, _, _, _ = voting_from_results(distances[row_idx], indices[row_idx], labels)
            if pred_label == true_label:
                correct += 1

            preds = [labels[i] for i in indices[row_idx]]
            n_total = label_counts.get(true_label, 0)
            for k in k_values:
                topk = preds[:k]
                hits = sum(1 for p in topk if p == true_label)
                precision[k] += hits / k
                if n_total > 0:
                    recall[k] += hits / min(n_total, k)

    if total == 0:
        raise ValueError("No validation samples were evaluated.")

    print("=" * 60)
    print("VALIDATION REPORT - HOG + LBP + dense SIFT + PCA")
    print("=" * 60)
    print(f"Index: {index_path}")
    print(f"Total validation queries: {total}")
    print()
    print(f"{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'Top-1 Accuracy':<20} {correct / total * 100:<10.2f}%")
    for k in k_values:
        print(f"{'Precision@' + str(k):<20} {precision[k] / total * 100:<10.2f}%")
    for k in k_values:
        print(f"{'Recall@' + str(k):<20} {recall[k] / total * 100:<10.2f}%")

    return correct / total, precision, recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate HOG+LBP+dense-SIFT PCA OCR retrieval")
    parser.add_argument("--index", default="./models/vector_database.index")
    parser.add_argument("--labels", default="./models/word_labels.npy")
    parser.add_argument("--names", default="./models/file_names.npy")
    parser.add_argument("--codebook", default="./models/codebook.pkl")
    parser.add_argument("--pca", default="./models/pca_model.pkl")
    parser.add_argument("--validation-dir", default="./Vietnam/validation_word")
    parser.add_argument("--csv-path", default="./Vietnam/validation_word.csv")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    validate(
        index_path=args.index,
        labels_path=args.labels,
        names_path=args.names,
        codebook_path=args.codebook,
        pca_path=args.pca,
        validation_dir=args.validation_dir,
        csv_path=args.csv_path,
        k_values=tuple(args.k),
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
