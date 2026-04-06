from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import math

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from torchvision import transforms

try:
	from pytorch_grad_cam import GradCAM
	from pytorch_grad_cam.utils.image import show_cam_on_image
	from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except Exception:  # pragma: no cover - optional dependency at runtime
	GradCAM = None
	show_cam_on_image = None
	ClassifierOutputTarget = None

from training.image_preprocessing import (  # noqa: E402
	IMAGENET_MEAN,
	IMAGENET_STD,
	SquarePadResize,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
GRADCAM_SUPPORTED_ARCHES = {"resnet18", "resnet50", "efficientnet_b0", "efficientnet_b3"}


def _ensure_parent(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, path: Path) -> None:
	_ensure_parent(path)
	fig.tight_layout()
	fig.savefig(path, dpi=180, bbox_inches="tight")
	plt.close(fig)


def build_per_class_f1_scores(
	detailed_results: Mapping[str, Mapping],
	class_names: list[str],
) -> dict[str, dict[str, float]]:
	per_class_f1: dict[str, dict[str, float]] = {}
	for arch, payload in detailed_results.items():
		report = payload.get("classification_report", {})
		per_class_f1[arch] = {
			class_name: float(report.get(class_name, {}).get("f1-score", 0.0))
			for class_name in class_names
		}
	return per_class_f1


def save_per_class_f1_scores(
	per_class_f1: Mapping[str, Mapping[str, float]],
	output_path: Path,
) -> None:
	_ensure_parent(output_path)
	with output_path.open("w", encoding="utf-8") as fp:
		json.dump(per_class_f1, fp, indent=2)


def plot_training_curves(
	detailed_results: Mapping[str, Mapping],
	output_path: Path,
) -> None:
	arches = list(detailed_results.keys())
	if not arches:
		return

	ncols = 2
	nrows = math.ceil(len(arches) / ncols)
	fig, axes = plt.subplots(nrows, ncols, figsize=(14, max(4, 4.2 * nrows)))
	axes = np.atleast_1d(axes).ravel()

	for ax, arch in zip(axes, arches):
		history = detailed_results[arch].get("history", {})
		epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
		if len(epochs) == 0:
			ax.axis("off")
			continue

		ax.plot(epochs, history.get("train_loss", []), label="train loss", linewidth=2)
		ax.plot(epochs, history.get("val_loss", []), label="val loss", linewidth=2)
		ax.set_title(arch)
		ax.set_xlabel("epoch")
		ax.set_ylabel("loss")
		ax.grid(alpha=0.18)

		ax2 = ax.twinx()
		ax2.plot(epochs, history.get("val_acc", []), label="val acc", linewidth=2, linestyle="--")
		ax2.plot(epochs, history.get("val_macro_f1", []), label="val macro-f1", linewidth=2, linestyle=":")
		ax2.set_ylabel("score")

		lines = ax.get_lines() + ax2.get_lines()
		labels = [line.get_label() for line in lines]
		ax.legend(lines, labels, loc="lower center", fontsize=8)

	for ax in axes[len(arches) :]:
		ax.axis("off")

	_save_fig(fig, output_path)


def plot_confusion_matrices(
	detailed_results: Mapping[str, Mapping],
	class_names: list[str],
	output_path: Path,
) -> None:
	arches = list(detailed_results.keys())
	if not arches:
		return

	ncols = min(2, len(arches))
	nrows = math.ceil(len(arches) / ncols)
	fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6.8 * nrows))
	axes = np.atleast_1d(axes).ravel()

	for ax, arch in zip(axes, arches):
		cm = np.array(detailed_results[arch].get("confusion_matrix_normalized", []), dtype=float)
		if cm.size == 0:
			ax.axis("off")
			continue
		sns.heatmap(
			cm,
			ax=ax,
			cmap="Blues",
			vmin=0.0,
			vmax=1.0,
			cbar=False,
			xticklabels=class_names,
			yticklabels=class_names,
		)
		metrics = detailed_results[arch].get("metrics", {})
		ax.set_title(
			f"{arch}\nacc={float(metrics.get('accuracy', 0.0)):.4f}  f1={float(metrics.get('macro_f1', 0.0)):.4f}",
			fontweight="bold",
		)
		ax.set_xlabel("predicted label")
		ax.set_ylabel("true label")
		ax.tick_params(axis="x", labelrotation=45, labelsize=7)
		ax.tick_params(axis="y", labelrotation=0, labelsize=7)

	for ax in axes[len(arches) :]:
		ax.axis("off")

	_save_fig(fig, output_path)


def plot_per_class_f1_comparison(
	per_class_f1: Mapping[str, Mapping[str, float]],
	class_names: list[str],
	output_path: Path,
) -> None:
	if not per_class_f1:
		return

	arches = list(per_class_f1.keys())
	matrix = np.array([[per_class_f1[arch][cls] for arch in arches] for cls in class_names], dtype=float)

	fig, ax = plt.subplots(figsize=(1.35 * len(arches) + 6, 0.42 * len(class_names) + 4))
	sns.heatmap(
		matrix,
		ax=ax,
		cmap="YlGnBu",
		vmin=0.0,
		vmax=1.0,
		annot=True,
		fmt=".2f",
		xticklabels=arches,
		yticklabels=class_names,
		cbar_kws={"label": "f1-score"},
	)
	ax.set_title("Per-class F1 comparison", fontweight="bold")
	ax.set_xlabel("architecture")
	ax.set_ylabel("class")
	ax.tick_params(axis="x", rotation=0)
	ax.tick_params(axis="y", rotation=0)
	_save_fig(fig, output_path)


def _resolve_gradcam_target_layers(model: torch.nn.Module, arch: str) -> list[torch.nn.Module]:
	if arch in {"resnet18", "resnet50"}:
		return [model.layer4[-1]]
	if arch.startswith("efficientnet_"):
		return [model.features[-1]]
	raise ValueError(f"Grad-CAM target layer not configured for architecture: {arch}")


def select_gradcam_arch(
	results_summary: Mapping[str, Mapping[str, float]],
) -> str | None:
	candidates = [
		(arch, metrics)
		for arch, metrics in results_summary.items()
		if arch in GRADCAM_SUPPORTED_ARCHES
	]
	if not candidates:
		return None
	best_arch, _ = max(
		candidates,
		key=lambda item: (
			float(item[1].get("test_accuracy", 0.0)),
			float(item[1].get("macro_f1", 0.0)),
		),
	)
	return best_arch


def _collect_gradcam_samples(data_dir: Path, class_names: list[str]) -> dict[str, Path]:
	samples: dict[str, Path] = {}
	for class_name in class_names:
		class_dir = data_dir / "test" / class_name
		if not class_dir.exists():
			continue
		candidates = sorted(
			[path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
		)
		if candidates:
			samples[class_name] = candidates[0]
	return samples


def save_gradcam_gallery(
	*,
	arch: str,
	class_names: list[str],
	data_dir: Path,
	checkpoint_path: Path,
	build_model_fn: Callable[[str, int], torch.nn.Module],
	extract_state_dict_fn: Callable[[object], Mapping[str, torch.Tensor]],
	device: torch.device,
	img_size: int,
	output_path: Path,
) -> bool:
	if GradCAM is None or show_cam_on_image is None or ClassifierOutputTarget is None:
		return False
	if arch not in GRADCAM_SUPPORTED_ARCHES:
		return False

	samples = _collect_gradcam_samples(data_dir, class_names)
	if not samples:
		return False

	model = build_model_fn(arch, len(class_names)).to(device)
	raw_state = torch.load(checkpoint_path, map_location=device)
	state_dict = extract_state_dict_fn(raw_state)
	model.load_state_dict(state_dict)
	model.eval()

	letterbox = SquarePadResize(img_size)
	transform = transforms.Compose(
		[
			letterbox,
			transforms.ToTensor(),
			transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
		]
	)

	target_layers = _resolve_gradcam_target_layers(model, arch)
	cam = GradCAM(model=model, target_layers=target_layers)

	ncols = 3
	nrows = math.ceil(len(samples) / ncols)
	fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
	axes = np.atleast_1d(axes).ravel()

	for ax, class_name in zip(axes, class_names):
		sample_path = samples.get(class_name)
		if sample_path is None:
			ax.axis("off")
			continue

		image = Image.open(sample_path).convert("RGB")
		display_image = letterbox(image)
		display_array = np.asarray(display_image, dtype=np.float32) / 255.0
		input_tensor = transform(image).unsqueeze(0).to(device)

		with torch.no_grad():
			logits = model(input_tensor)
			probs = torch.softmax(logits, dim=1)
			pred_idx = int(torch.argmax(probs, dim=1).item())
			pred_conf = float(probs[0, pred_idx].item())

		targets = [ClassifierOutputTarget(pred_idx)]
		grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
		overlay = show_cam_on_image(display_array, grayscale_cam, use_rgb=True)

		ax.imshow(overlay)
		ax.set_title(
			f"{class_name}\npred={class_names[pred_idx]} ({pred_conf:.2%})",
			fontsize=9,
			fontweight="bold",
		)
		ax.axis("off")

	for ax in axes[len(class_names) :]:
		ax.axis("off")

	fig.suptitle(f"Grad-CAM gallery ({arch})", fontsize=14, fontweight="bold")
	_save_fig(fig, output_path)
	return True
