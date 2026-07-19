from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.preprocessing import pre_processing

EMBED_DIM = 256
# pre_processing() resizes to cv2 dsize=(128, 256) -> array shape (H=256, W=128).
_TENSOR_SHAPE = (256, 128)


class WordEmbeddingNet(nn.Module):
    """ImageNet-pretrained ResNet18 adapted to 1-channel handwritten-word
    crops, projected to a compact L2-normalized embedding for FAISS search.

    This replaces the hand-engineered HOG+LBP+dense-SIFT pipeline in
    src/features.py. The FAISS index, voting logic, and IndexFlatIP cosine
    search are unchanged -- only the vector that goes into the index differs.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)

        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
            stride=old_conv.stride, padding=old_conv.padding, bias=False,
        )
        if pretrained:
            # Average the pretrained 3-channel filters into 1 channel instead
            # of discarding the learned low-level edge/texture detectors.
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv

        # children(): conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        # Drop only the final fc; keep the adaptive avgpool.
        self.trunk = nn.Sequential(*list(backbone.children())[:-1])
        self.embed = nn.Linear(512, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)
        feat = torch.flatten(feat, 1)
        emb = self.embed(feat)
        return F.normalize(emb, p=2, dim=1)


class ArcMarginProduct(nn.Module):
    """ArcFace margin head, used only during training. At inference time we
    discard this and read embeddings straight out of WordEmbeddingNet --
    ArcFace's job is just to shape the embedding space so that same-word
    crops end up close together and different words far apart, which is
    exactly what FAISS cosine retrieval needs.
    """

    def __init__(self, embed_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.30):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        weight = F.normalize(self.weight, dim=1)
        cosine = embeddings @ weight.t()
        if labels is None:
            return cosine * self.scale

        theta = torch.acos(cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=weight.shape[0]).float()
        logits = one_hot * target_logits + (1.0 - one_hot) * cosine
        return logits * self.scale


def load_word_image(path: str | Path) -> torch.Tensor:
    """Load + preprocess one crop exactly like the classical pipeline does
    (same deskew/binarize/distance-transform), then convert to a model
    input tensor. Reusing pre_processing keeps both pipelines comparable."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    processed = pre_processing(image)
    if processed.shape != _TENSOR_SHAPE:
        processed = cv2.resize(processed, (_TENSOR_SHAPE[1], _TENSOR_SHAPE[0]))
    tensor = torch.from_numpy(processed).float().unsqueeze(0) / 255.0
    return (tensor - 0.5) / 0.5


def create_cnn_vector(path: str | Path, model: WordEmbeddingNet, device: str = "cpu") -> np.ndarray:
    """Single-image convenience wrapper around WordEmbeddingNet, mirroring
    create_vector() in src/features.py so call sites stay symmetrical."""
    model.eval()
    with torch.no_grad():
        tensor = load_word_image(path).unsqueeze(0).to(device)
        embedding = model(tensor)
    return embedding.cpu().numpy()[0].astype(np.float32)
