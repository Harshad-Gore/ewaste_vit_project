from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import (  # noqa: E402
    build_classification_artifacts,
    compute_classification_metrics,
    write_metrics_json,
)
from training.hardware_utils import (  # noqa: E402
    detect_runtime,
    suggest_batch_size,
    suggest_num_workers,
)


SUPPORTED_ARCHES = [
    "resnet18",
    "resnet50",
    "efficientnet_b0",
    "efficientnet_b3",
    "convnext_tiny",
    "swin_tiny",
    "vit_b16",
]


@dataclass
class TrainConfig:
    img_size: int = 224
    batch_size: int = 32
    num_epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    patience: int = 6
    unfreeze_epoch: int = 6
    seed: int = 42
    label_smoothing: float = 0.1
    use_amp: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_data_dir(data_dir: str | None) -> Path:
    if data_dir:
        candidate = Path(data_dir).resolve()
    else:
        candidate = PROJECT_ROOT / "data"

    required = [candidate / "train", candidate / "val", candidate / "test"]
    if not all(path.exists() for path in required):
        missing = [str(path) for path in required if not path.exists()]
        raise FileNotFoundError(
            "dataset folders not found. expected train/val/test under data dir. "
            f"missing: {missing}"
        )
    return candidate


def build_transforms(config: TrainConfig) -> tuple[transforms.Compose, transforms.Compose]:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((config.img_size + 32, config.img_size + 32)),
            transforms.RandomCrop(config.img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_transform, eval_transform


def build_dataloaders(data_dir: Path, config: TrainConfig):
    train_tf, eval_tf = build_transforms(config)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)

    class_names = train_ds.classes
    if val_ds.classes != class_names or test_ds.classes != class_names:
        raise ValueError("class order mismatch between train/val/test folders")

    targets = np.array(train_ds.targets)
    class_counts = np.bincount(targets)
    sample_weights = 1.0 / class_counts[targets]

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names, class_counts


def _replace_resnet_head(model: nn.Module, num_classes: int) -> None:
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


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        _replace_resnet_head(model, num_classes)
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        _replace_resnet_head(model, num_classes)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_dim = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    elif arch == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        in_dim = model.classifier[-1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    elif arch == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_dim, num_classes)
    elif arch == "swin_tiny":
        model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        in_dim = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_dim, num_classes),
        )
    elif arch == "vit_b16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        in_dim = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    else:
        raise ValueError(f"unsupported architecture: {arch}")

    return model


def freeze_backbone(model: nn.Module, arch: str) -> None:
    head_keys = {
        "resnet18": ["fc"],
        "resnet50": ["fc"],
        "efficientnet_b0": ["classifier"],
        "efficientnet_b3": ["classifier"],
        "convnext_tiny": ["classifier"],
        "swin_tiny": ["head"],
        "vit_b16": ["heads"],
    }

    keys = head_keys[arch]
    for name, param in model.named_parameters():
        param.requires_grad = any(key in name for key in keys)


def unfreeze_for_finetune(model: nn.Module, arch: str) -> None:
    unfreeze_keys = {
        "resnet18": ["layer4", "fc"],
        "resnet50": ["layer4", "fc"],
        "efficientnet_b0": ["features.7", "features.8", "classifier"],
        "efficientnet_b3": ["features.7", "features.8", "classifier"],
        "convnext_tiny": ["features.6", "features.7", "classifier"],
        "swin_tiny": ["features.6", "features.7", "head"],
        "vit_b16": ["encoder.layers.encoder_layer_10", "encoder.layers.encoder_layer_11", "heads"],
    }

    keys = unfreeze_keys[arch]
    for name, param in model.named_parameters():
        if any(key in name for key in keys):
            param.requires_grad = True


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_sum += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss_sum = 0.0
    total = 0
    preds: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []

    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)

        loss_sum += loss.item() * images.size(0)
        total += labels.size(0)

        preds.append(logits.argmax(dim=1).cpu().numpy())
        labels_all.append(labels.cpu().numpy())

    return loss_sum / total, np.concatenate(preds), np.concatenate(labels_all)


def run_single_arch(
    arch: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    class_names: list[str],
    class_counts: np.ndarray,
    config: TrainConfig,
    output_dir: Path,
    device: torch.device,
) -> dict:
    print(f"\ntraining: {arch}")
    print("-" * 60)

    model = build_model(arch, num_classes=len(class_names)).to(device)
    freeze_backbone(model, arch)

    weight_tensor = torch.tensor(class_counts.max() / class_counts, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=config.label_smoothing)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)

    amp_enabled = config.use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

    best_state = None
    best_val = -1.0
    patience_ctr = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": [],
    }

    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        if epoch == config.unfreeze_epoch:
            unfreeze_for_finetune(model, arch)
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.lr * 0.2,
                weight_decay=config.weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, config.num_epochs - epoch + 1),
                eta_min=1e-6,
            )
            print(f"epoch {epoch}: fine-tuning layers unfrozen")

        train_loss, train_acc = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
        )

        val_loss, val_preds, val_labels = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
        )

        val_metrics = compute_classification_metrics(val_labels, val_preds)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics.accuracy)
        history["val_macro_f1"].append(val_metrics.macro_f1)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"epoch {epoch:02d}/{config.num_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_metrics.accuracy:.4f} "
            f"val_macro_f1={val_metrics.macro_f1:.4f}"
        )

        if val_metrics.macro_f1 > best_val:
            best_val = val_metrics.macro_f1
            best_state = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= config.patience:
                print(f"early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError(f"no best checkpoint captured for {arch}")

    model.load_state_dict(best_state)

    test_loss, test_preds, test_labels = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
    )
    test_metrics = compute_classification_metrics(test_labels, test_preds)
    artifacts = build_classification_artifacts(test_labels, test_preds, class_names)

    elapsed = time.time() - start_time
    arch_dir = output_dir / arch
    arch_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = arch_dir / f"{arch}_best.pth"
    torch.save(model.state_dict(), checkpoint_path)

    payload = {
        "arch": arch,
        "history": history,
        "test_loss": round(test_loss, 4),
        "metrics": {k: round(v, 4) for k, v in asdict(test_metrics).items()},
        "elapsed_seconds": round(elapsed, 2),
        "class_names": class_names,
        **artifacts,
    }

    write_metrics_json(arch_dir / "results.json", payload)

    report_txt_path = arch_dir / "classification_report.txt"
    with report_txt_path.open("w", encoding="utf-8") as fp:
        fp.write(f"model: {arch}\n\n")
        fp.write(json.dumps(artifacts["classification_report"], indent=2))
        fp.write("\n")

    print(
        f"test summary ({arch}) | "
        f"accuracy={test_metrics.accuracy:.4f}, macro_f1={test_metrics.macro_f1:.4f}"
    )

    return {
        "test_accuracy": round(test_metrics.accuracy, 4),
        "test_loss": round(test_loss, 4),
        "macro_f1": round(test_metrics.macro_f1, 4),
        "weighted_f1": round(test_metrics.weighted_f1, 4),
        "macro_precision": round(test_metrics.macro_precision, 4),
        "macro_recall": round(test_metrics.macro_recall, 4),
        "elapsed_seconds": round(elapsed, 2),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="research-grade ewaste benchmark trainer")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="dataset directory with train/val/test folders (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "classification"),
        help="directory to save model checkpoints and results",
    )
    parser.add_argument(
        "--arches",
        type=str,
        default="resnet50,vit_b16,convnext_tiny,swin_tiny,efficientnet_b3",
        help="comma-separated model list",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--unfreeze-epoch", type=int, default=6)
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    arches = [a.strip() for a in args.arches.split(",") if a.strip()]
    invalid = [a for a in arches if a not in SUPPORTED_ARCHES]
    if invalid:
        raise ValueError(f"unsupported arches: {invalid}. supported: {SUPPORTED_ARCHES}")

    runtime = detect_runtime()

    # auto-pick performant defaults when not explicitly provided.
    auto_batch_size = suggest_batch_size(runtime.vram_gb, arches)
    auto_num_workers = suggest_num_workers(
        cpu_count=runtime.cpu_count,
        on_gpu=runtime.device.type == "cuda",
    )

    batch_size = args.batch_size if args.batch_size is not None else auto_batch_size
    num_workers = args.num_workers if args.num_workers is not None else auto_num_workers

    config = TrainConfig(
        img_size=args.img_size,
        batch_size=batch_size,
        num_epochs=args.epochs,
        num_workers=num_workers,
        seed=args.seed,
        patience=args.patience,
        unfreeze_epoch=args.unfreeze_epoch,
        use_amp=not args.no_amp,
    )

    set_seed(config.seed)

    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = runtime.device
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"device: {device}")
    if runtime.gpu_name:
        print(f"gpu_name: {runtime.gpu_name}")
        print(f"vram_gb: {runtime.vram_gb:.2f}")
    print(f"cpu_count: {runtime.cpu_count}")
    print(f"data_dir: {data_dir}")
    print(f"output_dir: {output_dir}")
    print(f"arches: {arches}")
    print(f"batch_size: {config.batch_size}")
    print(f"num_workers: {config.num_workers}")

    train_loader, val_loader, test_loader, class_names, class_counts = build_dataloaders(
        data_dir=data_dir,
        config=config,
    )

    print(f"num_classes: {len(class_names)}")
    print(f"train_samples: {len(train_loader.dataset)}")
    print(f"val_samples: {len(val_loader.dataset)}")
    print(f"test_samples: {len(test_loader.dataset)}")

    results: dict[str, dict] = {}

    for arch in arches:
        results[arch] = run_single_arch(
            arch=arch,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            class_names=class_names,
            class_counts=class_counts,
            config=config,
            output_dir=output_dir,
            device=device,
        )

    best_arch, best_info = max(results.items(), key=lambda item: item[1]["test_accuracy"])

    summary_payload = {
        "best_arch": best_arch,
        "results": results,
        "best_metrics": best_info,
        "class_names": class_names,
        "config": asdict(config),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    write_metrics_json(output_dir / "test_results.json", results)
    write_metrics_json(
        output_dir / "best_model.json",
        {
            "best_arch": best_arch,
            "results": {
                "test_accuracy": best_info["test_accuracy"],
                "test_loss": best_info["test_loss"],
                "macro_f1": best_info["macro_f1"],
                "weighted_f1": best_info["weighted_f1"],
                "macro_precision": best_info["macro_precision"],
                "macro_recall": best_info["macro_recall"],
            },
            "checkpoint": best_info["checkpoint"],
        },
    )
    write_metrics_json(output_dir / "benchmark_summary.json", summary_payload)

    print("\nbenchmark complete")
    print(f"best_arch: {best_arch}")
    print(f"best_accuracy: {best_info['test_accuracy']:.4f}")


if __name__ == "__main__":
    main()
