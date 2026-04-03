from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (  # noqa: E402
    build_classification_artifacts,
    compute_classification_metrics,
    write_metrics_json,
)
from training.hardware_utils import detect_runtime  # noqa: E402


SUPPORTED_ARCHES = [
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b3",
    "convnext_tiny",
    "swin_tiny",
    "vit_b16",
]


HAZARD_MAP = {
    "Battery": "HIGH",
    "PCB": "HIGH",
    "Mobile": "HIGH",
    "Television": "HIGH",
    "Laptop": "HIGH",
    "light bulbs": "HIGH",
    "Refrigerator": "HIGH",
    "Air-Conditioner": "HIGH",
    "Microwave": "MEDIUM",
    "Washing Machine": "MEDIUM",
    "Printer": "MEDIUM",
    "Microchip-IC": "MEDIUM",
    "Keyboard": "LOW",
    "Mouse": "LOW",
    "Resistor": "LOW",
    "transistor": "LOW",
    "heat-sink": "LOW",
    "Passive-Component": "LOW",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="deep + traditional ml competition runner")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data"))
    parser.add_argument(
        "--classification-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "classification"),
        help="directory containing deep model checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "competition"),
    )
    parser.add_argument(
        "--embedding-arch",
        type=str,
        default=None,
        help="arch for embedding extraction for traditional ml (defaults to best deep model)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def build_eval_datasets(data_dir: Path):
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

    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        raise ValueError("class mismatch between train/val/test")

    return train_ds, val_ds, test_ds


def build_model_for_inference(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=None)
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
    elif arch == "resnet50":
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
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_dim = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    elif arch == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        in_dim = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        in_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_dim, num_classes)
    elif arch == "swin_tiny":
        model = models.swin_t(weights=None)
        in_dim = model.head.in_features
        model.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_dim, num_classes))
    elif arch == "vit_b16":
        model = models.vit_b_16(weights=None)
        in_dim = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    else:
        raise ValueError(f"unsupported arch: {arch}")
    return model


@torch.no_grad()
def evaluate_deep_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    labels_all = []

    for images, labels in tqdm(loader, desc="deep-eval", leave=False):
        images = images.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels_all.append(labels.numpy())

    return np.concatenate(preds), np.concatenate(labels_all)


def get_embedding_extractor(model: nn.Module, arch: str) -> nn.Module:
    if arch in {"resnet18", "resnet50"}:
        return nn.Sequential(*list(model.children())[:-1], nn.Flatten(1))
    if arch in {"efficientnet_b0", "efficientnet_b3"}:
        return nn.Sequential(model.features, model.avgpool, nn.Flatten(1))
    if arch == "convnext_tiny":
        return nn.Sequential(model.features, model.avgpool, nn.Flatten(1))
    if arch == "swin_tiny":
        class SwinExtractor(nn.Module):
            def __init__(self, net: nn.Module):
                super().__init__()
                self.net = net

            def forward(self, x):
                x = self.net.features(x)
                x = self.net.norm(x)
                x = self.net.permute(x)
                x = self.net.avgpool(x)
                x = torch.flatten(x, 1)
                return x

        return SwinExtractor(model)
    if arch == "vit_b16":
        class ViTExtractor(nn.Module):
            def __init__(self, net: nn.Module):
                super().__init__()
                self.net = net

            def forward(self, x):
                x = self.net._process_input(x)
                cls = self.net.class_token.expand(x.shape[0], -1, -1)
                x = torch.cat([cls, x], dim=1)
                x = self.net.encoder(x)
                return x[:, 0]

        return ViTExtractor(model)
    raise ValueError(f"unsupported embedding extractor for arch: {arch}")


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    arch: str,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    extractor = get_embedding_extractor(model, arch).to(device).eval()
    embeddings = []

    for images, _ in tqdm(loader, desc=f"embed-{arch}", leave=False):
        images = images.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            feats = extractor(images)
        embeddings.append(feats.detach().cpu().float().numpy())

    return np.vstack(embeddings)


def discover_checkpoints(classification_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for arch in SUPPORTED_ARCHES:
        p = classification_dir / arch / f"{arch}_best.pth"
        if p.exists():
            out[arch] = p
    return out


def run_hierarchical_svm(
    X_trainval: np.ndarray,
    y_trainval: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict:
    hazard_int = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    inv_hazard_int = {v: k for k, v in hazard_int.items()}

    hz_train = np.array([hazard_int[HAZARD_MAP[class_names[label]]] for label in y_trainval])
    hz_test = np.array([hazard_int[HAZARD_MAP[class_names[label]]] for label in y_test])

    stage1 = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
    stage1.fit(X_trainval, hz_train)
    stage1_pred = stage1.predict(X_test)
    stage1_acc = float(accuracy_score(hz_test, stage1_pred))

    groups: dict[str, list[int]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for idx, cls in enumerate(class_names):
        groups[HAZARD_MAP[cls]].append(idx)

    stage2_models = {}
    stage2_maps = {}

    for hz_name, cls_indices in groups.items():
        mask = np.isin(y_trainval, cls_indices)
        if mask.sum() < 10:
            continue

        unique = sorted(np.unique(y_trainval[mask]))
        global_to_local = {g: i for i, g in enumerate(unique)}
        local_to_global = {i: g for g, i in global_to_local.items()}

        y_local = np.array([global_to_local[val] for val in y_trainval[mask]])

        clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
        clf.fit(X_trainval[mask], y_local)

        stage2_models[hz_name] = clf
        stage2_maps[hz_name] = local_to_global

    final_preds = []
    for i in range(len(X_test)):
        hz_name = inv_hazard_int[int(stage1.predict(X_test[i : i + 1])[0])]
        if hz_name not in stage2_models:
            final_preds.append(0)
            continue
        local = int(stage2_models[hz_name].predict(X_test[i : i + 1])[0])
        final_preds.append(stage2_maps[hz_name][local])

    final_preds = np.array(final_preds)
    metrics = compute_classification_metrics(y_test, final_preds)

    return {
        "metrics": asdict(metrics),
        "stage1_hazard_accuracy": round(stage1_acc, 4),
        "preds": final_preds.tolist(),
        "labels": y_test.tolist(),
    }


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    classification_dir = Path(args.classification_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = detect_runtime()
    device = runtime.device

    print(f"device: {device}")
    if runtime.gpu_name:
        print(f"gpu_name: {runtime.gpu_name}")
        print(f"vram_gb: {runtime.vram_gb:.2f}")

    train_ds, val_ds, test_ds = build_eval_datasets(data_dir)
    class_names = train_ds.classes
    num_classes = len(class_names)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    ckpt_map = discover_checkpoints(classification_dir)
    if not ckpt_map:
        raise FileNotFoundError(
            f"no checkpoints found under {classification_dir}. run training/research_benchmark.py first"
        )

    deep_results = {}

    for arch, ckpt_path in ckpt_map.items():
        model = build_model_for_inference(arch, num_classes=num_classes).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

        start = time.time()
        preds, labels = evaluate_deep_model(model, test_loader, device)
        elapsed = time.time() - start

        metrics = compute_classification_metrics(labels, preds)
        artifacts = build_classification_artifacts(labels, preds, class_names)

        deep_results[arch] = {
            "model_type": "deep_learning",
            "metrics": {k: round(v, 4) for k, v in asdict(metrics).items()},
            "elapsed_seconds": round(elapsed, 2),
            "checkpoint": str(ckpt_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            **artifacts,
            "preds": preds.tolist(),
            "labels": labels.tolist(),
        }

    # choose embedding source model
    if args.embedding_arch:
        embedding_arch = args.embedding_arch
    else:
        best_info_path = classification_dir / "best_model.json"
        if best_info_path.exists():
            with best_info_path.open("r", encoding="utf-8") as fp:
                embedding_arch = json.load(fp).get("best_arch", "")
        else:
            embedding_arch = ""

    if embedding_arch not in ckpt_map:
        embedding_arch = max(
            deep_results,
            key=lambda a: deep_results[a]["metrics"]["macro_f1"],
        )

    emb_model = build_model_for_inference(embedding_arch, num_classes=num_classes).to(device)
    emb_model.load_state_dict(torch.load(ckpt_map[embedding_arch], map_location=device))

    embed_loader = DataLoader(
        ConcatDataset([train_ds, val_ds, test_ds]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    embeddings = extract_embeddings(emb_model, embedding_arch, embed_loader, device)

    n_train = len(train_ds)
    n_val = len(val_ds)

    X_train_raw = embeddings[:n_train]
    X_val_raw = embeddings[n_train : n_train + n_val]
    X_test_raw = embeddings[n_train + n_val :]

    y_train = np.array([sample[1] for sample in train_ds.samples])
    y_val = np.array([sample[1] for sample in val_ds.samples])
    y_test = np.array([sample[1] for sample in test_ds.samples])

    y_trainval = np.concatenate([y_train, y_val])
    X_trainval_raw = np.vstack([X_train_raw, X_val_raw])

    pca_dim = min(256, X_trainval_raw.shape[1])
    pca = PCA(n_components=pca_dim, random_state=42)
    scaler = StandardScaler()

    X_trainval = scaler.fit_transform(pca.fit_transform(X_trainval_raw))
    X_test = scaler.transform(pca.transform(X_test_raw))

    traditional_models = {
        "knn_k7": KNeighborsClassifier(n_neighbors=7, metric="cosine", n_jobs=-1),
        "svm_rbf": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42),
        "svm_linear": SVC(kernel="linear", C=1.0, probability=True, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=350, random_state=42, n_jobs=-1),
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            n_jobs=-1,
            random_state=42,
        ),
        "naive_bayes": GaussianNB(),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=3,
            random_state=42,
        ),
    }

    traditional_results = {}

    for model_name, clf in traditional_models.items():
        start = time.time()
        clf.fit(X_trainval, y_trainval)
        preds = clf.predict(X_test)
        elapsed = time.time() - start

        metrics = compute_classification_metrics(y_test, preds)
        artifacts = build_classification_artifacts(y_test, preds, class_names)

        traditional_results[model_name] = {
            "model_type": "traditional_ml",
            "embedding_arch": embedding_arch,
            "metrics": {k: round(v, 4) for k, v in asdict(metrics).items()},
            "elapsed_seconds": round(elapsed, 2),
            **artifacts,
            "preds": preds.tolist(),
            "labels": y_test.tolist(),
        }

    hierarchical = run_hierarchical_svm(
        X_trainval=X_trainval,
        y_trainval=y_trainval,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
    )

    all_players = {}
    for name, payload in deep_results.items():
        all_players[name] = payload
    for name, payload in traditional_results.items():
        all_players[name] = payload
    all_players["hierarchical_svm"] = {
        "model_type": "hierarchical",
        "metrics": {k: round(v, 4) for k, v in hierarchical["metrics"].items()},
        "stage1_hazard_accuracy": hierarchical["stage1_hazard_accuracy"],
        "preds": hierarchical["preds"],
        "labels": hierarchical["labels"],
    }

    ranked = sorted(
        all_players.items(),
        key=lambda item: (item[1]["metrics"]["macro_f1"], item[1]["metrics"]["accuracy"]),
        reverse=True,
    )

    winner_name, winner_payload = ranked[0]

    leaderboard = {
        "winner": winner_name,
        "winner_metrics": winner_payload["metrics"],
        "ranking": [
            {
                "rank": i + 1,
                "model": name,
                "model_type": payload["model_type"],
                "accuracy": payload["metrics"]["accuracy"],
                "macro_f1": payload["metrics"]["macro_f1"],
            }
            for i, (name, payload) in enumerate(ranked)
        ],
        "embedding_arch_for_traditional_ml": embedding_arch,
        "class_names": class_names,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    write_metrics_json(output_dir / "deep_results.json", deep_results)
    write_metrics_json(output_dir / "traditional_ml_results.json", traditional_results)
    write_metrics_json(output_dir / "all_players_results.json", all_players)
    write_metrics_json(output_dir / "leaderboard.json", leaderboard)

    print("\nmodel competition leaderboard")
    print("=" * 78)
    for row in leaderboard["ranking"]:
        print(
            f"rank {row['rank']:>2}: {row['model']:<22} "
            f"type={row['model_type']:<14} acc={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f}"
        )
    print("-" * 78)
    print(f"winner: {winner_name}")


if __name__ == "__main__":
    main()
