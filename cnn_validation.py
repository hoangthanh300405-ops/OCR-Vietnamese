from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.cnn_database import _InferDataset
from src.cnn_features import EMBED_DIM, WordEmbeddingNet
from src.database import load_labels


def validate_cnn(
    index_path: str = "./models/cnn_vector_database.index",
    labels_path: str = "./models/cnn_word_labels.npy",
    names_path: str = "./models/cnn_file_names.npy",
    model_path: str = "./models/cnn_embedding.pt",
    validation_dir: str = "./Vietnam/validation_word",
    csv_path: str = "./Vietnam/validation_word.csv",
    k_values: tuple[int, ...] = (1, 5, 10),
    batch_size: int = 256,
    num_workers: int = 4,
    max_samples: int | None = None,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load artefacts ────────────────────────────────────────────────────────
    index = faiss.read_index(index_path)
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = 64

    labels = np.load(labels_path, allow_pickle=True)
    labels_dict = load_labels(csv_path)
    label_counts = Counter(labels.tolist())

    checkpoint = torch.load(model_path, map_location=device)
    embed_dim = checkpoint.get("embed_dim", EMBED_DIM)
    model = WordEmbeddingNet(embed_dim=embed_dim, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Loaded CNN model  : {model_path}  (embed_dim={embed_dim})")
    print(f"Index vectors     : {index.ntotal}  (dim={index.d})")
    print(f"Validation dir    : {validation_dir}")

    # ── Build file list ───────────────────────────────────────────────────────
    file_list = sorted(f for f in os.listdir(validation_dir) if f.lower().endswith(".png"))
    if max_samples is not None:
        file_list = file_list[:max_samples]

    # Filter out files without ground-truth labels
    valid_files = [f for f in file_list if labels_dict.get(Path(f).stem) is not None]
    true_labels_list = [labels_dict[Path(f).stem] for f in valid_files]

    dataset = _InferDataset(Path(validation_dir), valid_files)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )

    max_k = max(k_values)
    total = 0
    correct = 0
    precision = {k: 0.0 for k in k_values}
    recall = {k: 0.0 for k in k_values}

    label_offset = 0
    with torch.no_grad():
        for batch_tensors in tqdm(loader, desc="Validating"):
            batch_tensors = batch_tensors.to(device)
            embeddings = model(batch_tensors).cpu().numpy().astype(np.float32)
            distances, indices = index.search(embeddings, max_k)

            for row_idx in range(len(embeddings)):
                true_label = true_labels_list[label_offset + row_idx]
                total += 1

                # Weighted vote for top-1 (matches vote() logic in app)
                sim = 2.0 - np.clip(distances[row_idx], 0, 4.0)
                counter: Counter = Counter()
                for idx_val, s in zip(indices[row_idx], sim):
                    counter[str(labels[idx_val])] += float(s)
                pred_label = counter.most_common(1)[0][0]
                if pred_label == true_label:
                    correct += 1

                preds = [str(labels[i]) for i in indices[row_idx]]
                n_total = label_counts.get(true_label, 0)
                for k in k_values:
                    topk = preds[:k]
                    hits = sum(1 for p in topk if p == true_label)
                    precision[k] += hits / k
                    if n_total > 0:
                        recall[k] += hits / min(n_total, k)

            label_offset += len(embeddings)

    if total == 0:
        raise ValueError("No validation samples were evaluated.")

    print()
    print("=" * 60)
    print("VALIDATION REPORT - ResNet18 ArcFace CNN embedding")
    print("=" * 60)
    print(f"Total queries : {total}")
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
    parser = argparse.ArgumentParser(description="Validate CNN ResNet18 ArcFace OCR retrieval")
    parser.add_argument("--index", default="./models/cnn_vector_database.index")
    parser.add_argument("--labels", default="./models/cnn_word_labels.npy")
    parser.add_argument("--names", default="./models/cnn_file_names.npy")
    parser.add_argument("--model", default="./models/cnn_embedding.pt")
    parser.add_argument("--validation-dir", default="./Vietnam/validation_word")
    parser.add_argument("--csv-path", default="./Vietnam/validation_word.csv")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    validate_cnn(
        index_path=args.index,
        labels_path=args.labels,
        names_path=args.names,
        model_path=args.model,
        validation_dir=args.validation_dir,
        csv_path=args.csv_path,
        k_values=tuple(args.k),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        device=args.device,
    )
