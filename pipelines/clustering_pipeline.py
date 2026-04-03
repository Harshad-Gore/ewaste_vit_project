from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, models, transforms
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.hardware_utils import detect_runtime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="clustering pipeline from trained classifier embeddings")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--classification-dir", type=str, default=str(PROJECT_ROOT / "models" / "classification"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "models" / "clustering"))
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tsne-sample", type=int, default=6000)
    return parser.parse_args()


def build_resnet50_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    in_dim = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


@torch.no_grad()
def extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten(1)).to(device).eval()
    all_embeddings = []

    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            feats = extractor(images)
        all_embeddings.append(feats.cpu().float().numpy())

    return np.vstack(all_embeddings)


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    cls_dir = Path(args.classification_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = detect_runtime()
    device = runtime.device

    best_info_path = cls_dir / "best_model.json"
    if not best_info_path.exists():
        raise FileNotFoundError(f"best model file not found: {best_info_path}")

    with best_info_path.open("r", encoding="utf-8") as fp:
        best_info = json.load(fp)

    best_arch = best_info.get("best_arch", "resnet50")
    if best_arch != "resnet50":
        print(f"best model is {best_arch}; clustering pipeline currently uses resnet50 extractor")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(data_dir / "train", transform=transform)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=transform)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=transform)

    class_names = train_ds.classes
    num_classes = len(class_names)

    checkpoint_path = cls_dir / "resnet50" / "resnet50_best.pth"
    if not checkpoint_path.exists():
        fallback = cls_dir / best_arch / f"{best_arch}_best.pth"
        if not fallback.exists():
            raise FileNotFoundError("no suitable checkpoint found for clustering feature extraction")
        checkpoint_path = fallback

    model = build_resnet50_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    full_ds = ConcatDataset([train_ds, val_ds, test_ds])
    full_loader = DataLoader(
        full_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    labels = np.array(
        [s[1] for s in train_ds.samples]
        + [s[1] for s in val_ds.samples]
        + [s[1] for s in test_ds.samples]
    )

    embeddings = extract_embeddings(model, full_loader, device)
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    pca = PCA(n_components=0.95, random_state=42)
    emb_pca = pca.fit_transform(emb_scaled)

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(emb_pca)

    metrics = {
        "n_clusters": args.n_clusters,
        "total_samples": int(len(labels)),
        "silhouette_score": float(silhouette_score(emb_pca, cluster_labels, sample_size=min(2000, len(labels)))),
        "davies_bouldin_index": float(davies_bouldin_score(emb_pca, cluster_labels)),
        "calinski_harabasz_index": float(calinski_harabasz_score(emb_pca, cluster_labels)),
        "adjusted_rand_index": float(adjusted_rand_score(labels, cluster_labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(labels, cluster_labels)),
        "cluster_sizes": {
            str(i): int((cluster_labels == i).sum()) for i in range(args.n_clusters)
        },
    }

    # t-sne on sample for visualization artifact compatibility
    n = len(emb_pca)
    sample_n = min(args.tsne_sample, n)
    idx = np.random.RandomState(42).choice(n, size=sample_n, replace=False)
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,
        learning_rate="auto",
        init="pca",
    )
    tsne_result = tsne.fit_transform(emb_pca[idx])

    np.save(output_dir / "embeddings.npy", embeddings)
    np.save(output_dir / "embeddings_scaled.npy", emb_scaled)
    np.save(output_dir / "embeddings_pca.npy", emb_pca)
    np.save(output_dir / "labels.npy", labels)
    np.save(output_dir / "cluster_labels.npy", cluster_labels)
    np.save(output_dir / "tsne_idx.npy", idx)
    np.save(output_dir / "tsne_result.npy", tsne_result)
    np.save(output_dir / "tsne_labels.npy", labels[idx])
    np.save(output_dir / "tsne_clusters.npy", cluster_labels[idx])

    with (output_dir / "clustering_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    with (output_dir / "clustering_results.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "n_clusters": args.n_clusters,
                "total_samples": int(len(labels)),
                "class_names": class_names,
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "pca_components": int(emb_pca.shape[1]),
                "pca_variance_kept": float(pca.explained_variance_ratio_.sum()),
            },
            fp,
            indent=2,
        )

    print("clustering pipeline complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
