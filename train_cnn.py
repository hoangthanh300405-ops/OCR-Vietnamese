from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.cnn_features import ArcMarginProduct, EMBED_DIM, WordEmbeddingNet, load_word_image
from src.database import _word_files, load_labels


class WordDataset(Dataset):
    def __init__(self, image_dir: str | Path, files: list[str], label_ids: list[int]):
        self.image_dir = Path(image_dir)
        self.files = files
        self.label_ids = label_ids

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        tensor = load_word_image(self.image_dir / self.files[idx])
        return tensor, self.label_ids[idx]


def build_label_index(files: list[str], labels_dict: dict[str, str]):
    classes = sorted({labels_dict.get(Path(f).stem, "unknown") for f in files})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    label_ids = [class_to_idx[labels_dict.get(Path(f).stem, "unknown")] for f in files]
    return label_ids, classes


def train(
    image_dir: str | Path = "./Vietnam/train_word",
    csv_path: str | Path = "./Vietnam/train_word.csv",
    output_model: str | Path = "./models/cnn_embedding.pt",
    embed_dim: int = EMBED_DIM,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_workers: int = 4,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    files = _word_files(image_dir)
    if not files:
        raise FileNotFoundError(f"No word images found in {image_dir}")
    labels_dict = load_labels(csv_path)
    label_ids, classes = build_label_index(files, labels_dict)
    num_classes = len(classes)
    print(f"Training on {len(files)} images, {num_classes} word classes, device={device}")

    dataset = WordDataset(image_dir, files, label_ids)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )

    model = WordEmbeddingNet(embed_dim=embed_dim, pretrained=True).to(device)
    head = ArcMarginProduct(embed_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        head.train()
        total_loss = 0.0
        correct = 0
        for images, labels in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            embeddings = model(images)
            logits = head(embeddings, labels)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()

        scheduler.step()
        print(f"  epoch {epoch + 1}: loss={total_loss / len(dataset):.4f} "
              f"train_acc={correct / len(dataset):.4f}")

    output_model = Path(output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "embed_dim": embed_dim, "classes": classes}, output_model)
    print(f"Saved model to {output_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN word-embedding model (ResNet18 + ArcFace)")
    parser.add_argument("--image-dir", default="./Vietnam/train_word")
    parser.add_argument("--csv-path", default="./Vietnam/train_word.csv")
    parser.add_argument("--output-model", default="./models/cnn_embedding.pt")
    parser.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    train(
        image_dir=args.image_dir,
        csv_path=args.csv_path,
        output_model=args.output_model,
        embed_dim=args.embed_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
    )
