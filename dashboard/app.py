from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json
import os
import random
import sys
import time
from urllib import request as urllib_request

import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent import AgentInput, EwasteDecisionAgent  # noqa: E402
from agent.tools import DISPOSAL_MAP, HAZARD_MAP, MATERIAL_MAP  # noqa: E402
from training.checkpoint_utils import (  # noqa: E402
	discover_classification_checkpoints,
	extract_model_state_dict,
	read_best_model_info,
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

EVAL_TRANSFORM = transforms.Compose(
	[
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
	]
)

HAZARD_COLOR = {
	"HIGH": "#f97316",
	"MEDIUM": "#facc15",
	"LOW": "#34d399",
	"UNKNOWN": "#94a3b8",
}

TONE_CLASS = {
	"neutral": "tone-neutral",
	"success": "tone-success",
	"warning": "tone-warning",
	"danger": "tone-danger",
}

WORKSPACES = [
	"Operations",
	"Policy",
	"Benchmarks",
	"Analytics",
	"Registry",
]


def inject_styles() -> None:
	st.markdown(
		"""
		<style>
		@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

		:root {
			--bg: #09111f;
			--panel: rgba(10, 20, 36, 0.82);
			--line: rgba(148, 163, 184, 0.18);
			--line-strong: rgba(148, 163, 184, 0.28);
			--text: #e2e8f0;
			--muted: #94a3b8;
			--accent: #2dd4bf;
			--accent-2: #38bdf8;
		}

		html, body, [data-testid="stAppViewContainer"] {
			font-family: 'IBM Plex Sans', sans-serif;
			color: var(--text);
			background:
				radial-gradient(1200px 520px at -10% -10%, rgba(45, 212, 191, 0.16), transparent 48%),
				radial-gradient(980px 420px at 112% 0%, rgba(56, 189, 248, 0.16), transparent 44%),
				linear-gradient(180deg, #08111d 0%, #0a1322 55%, #09101d 100%);
		}

		[data-testid="stHeader"] {
			background: transparent;
		}

		h1, h2, h3 {
			font-family: 'Space Grotesk', sans-serif;
			letter-spacing: 0.01em;
			color: #f8fafc;
		}

		.hero {
			position: relative;
			overflow: hidden;
			border: 1px solid var(--line);
			border-radius: 24px;
			padding: 26px 28px;
			background: linear-gradient(135deg, rgba(8, 17, 31, 0.96) 0%, rgba(15, 23, 42, 0.94) 56%, rgba(10, 18, 32, 0.98) 100%);
			box-shadow: 0 24px 80px rgba(2, 8, 23, 0.38);
			margin-bottom: 12px;
		}

		.hero::after {
			content: "";
			position: absolute;
			inset: auto -10% -38% 38%;
			height: 240px;
			background: radial-gradient(circle, rgba(45, 212, 191, 0.16) 0%, transparent 62%);
			pointer-events: none;
		}

		.hero-kicker {
			text-transform: uppercase;
			font-size: 0.74rem;
			letter-spacing: 0.16em;
			font-weight: 700;
			color: var(--accent);
			margin-bottom: 10px;
		}

		.hero h1 {
			margin: 0;
			font-size: 2.28rem;
			line-height: 1.05;
		}

		.hero p {
			margin-top: 12px;
			margin-bottom: 0;
			color: #cbd5e1;
			font-size: 1rem;
			line-height: 1.7;
			max-width: 880px;
		}

		.glass-card {
			border: 1px solid var(--line);
			border-radius: 18px;
			padding: 16px;
			background: linear-gradient(180deg, rgba(15, 23, 42, 0.84) 0%, rgba(8, 17, 31, 0.96) 100%);
			box-shadow: 0 16px 48px rgba(2, 8, 23, 0.24);
			min-height: 132px;
		}

		.kpi-value {
			font-size: 1.72rem;
			font-weight: 800;
			color: #f8fafc;
		}

		.kpi-label {
			color: var(--muted);
			font-size: 0.74rem;
			text-transform: uppercase;
			letter-spacing: 0.14em;
			font-weight: 700;
		}

		.kpi-detail {
			margin-top: 10px;
			font-size: 0.88rem;
			color: #cbd5e1;
			line-height: 1.45;
		}

		.hazard-pill {
			display: inline-block;
			font-weight: 800;
			color: #f8fafc;
			border-radius: 999px;
			padding: 0.34rem 0.9rem;
			font-size: 0.78rem;
			letter-spacing: 0.08em;
			text-transform: uppercase;
		}

		[data-testid="stSidebar"] {
			border-right: 1px solid var(--line);
			background: linear-gradient(180deg, rgba(8, 17, 31, 0.98) 0%, rgba(10, 20, 36, 0.98) 100%);
		}

		.panel-card, .signal-banner, .prob-card, .sidebar-card {
			border: 1px solid var(--line);
			border-radius: 20px;
			background: linear-gradient(180deg, rgba(12, 22, 38, 0.86) 0%, rgba(8, 17, 31, 0.94) 100%);
			box-shadow: 0 16px 48px rgba(2, 8, 23, 0.2);
		}

		.panel-card, .prob-card {
			padding: 18px;
		}

		.sidebar-card {
			padding: 14px;
			margin-bottom: 12px;
		}

		.panel-label, .sidebar-label {
			font-size: 0.76rem;
			text-transform: uppercase;
			letter-spacing: 0.14em;
			color: var(--muted);
			font-weight: 700;
			margin-bottom: 6px;
		}

		.panel-value {
			font-size: 1.26rem;
			font-weight: 700;
			color: #f8fafc;
			margin-bottom: 8px;
		}

		.panel-copy, .sidebar-copy, .section-copy {
			font-size: 0.94rem;
			line-height: 1.65;
			color: #cbd5e1;
		}

		.signal-banner {
			padding: 16px 18px;
			margin: 4px 0 14px;
		}

		.signal-banner.tone-warning {
			background: linear-gradient(180deg, rgba(120, 53, 15, 0.22) 0%, rgba(8, 17, 31, 0.92) 100%);
		}

		.signal-banner.tone-success {
			background: linear-gradient(180deg, rgba(6, 95, 70, 0.22) 0%, rgba(8, 17, 31, 0.92) 100%);
		}

		.signal-banner.tone-danger {
			background: linear-gradient(180deg, rgba(127, 29, 29, 0.26) 0%, rgba(8, 17, 31, 0.92) 100%);
		}

		.signal-banner.tone-neutral {
			background: linear-gradient(180deg, rgba(14, 116, 144, 0.18) 0%, rgba(8, 17, 31, 0.92) 100%);
		}

		.glass-card.tone-success {
			box-shadow: inset 0 1px 0 rgba(52, 211, 153, 0.22), 0 16px 48px rgba(2, 8, 23, 0.24);
		}

		.glass-card.tone-warning {
			box-shadow: inset 0 1px 0 rgba(245, 158, 11, 0.22), 0 16px 48px rgba(2, 8, 23, 0.24);
		}

		.glass-card.tone-danger {
			box-shadow: inset 0 1px 0 rgba(251, 113, 133, 0.24), 0 16px 48px rgba(2, 8, 23, 0.24);
		}

		.glass-card.tone-neutral {
			box-shadow: inset 0 1px 0 rgba(56, 189, 248, 0.18), 0 16px 48px rgba(2, 8, 23, 0.24);
		}

		.signal-title, .section-headline {
			color: #f8fafc;
			font-weight: 700;
		}

		.signal-title {
			font-size: 1rem;
			margin-bottom: 5px;
		}

		.signal-copy {
			color: #cbd5e1;
			font-size: 0.94rem;
			line-height: 1.6;
			margin: 0;
		}

		.prob-row + .prob-row {
			margin-top: 14px;
		}

		.prob-meta {
			display: flex;
			justify-content: space-between;
			gap: 1rem;
			font-size: 0.92rem;
			margin-bottom: 6px;
			color: #e2e8f0;
		}

		.prob-track {
			height: 11px;
			border-radius: 999px;
			background: rgba(148, 163, 184, 0.14);
			overflow: hidden;
		}

		.prob-fill {
			height: 100%;
			border-radius: 999px;
			background: linear-gradient(90deg, #2dd4bf 0%, #38bdf8 52%, #22c55e 100%);
		}

		.section-kicker {
			color: var(--accent);
			text-transform: uppercase;
			letter-spacing: 0.16em;
			font-size: 0.72rem;
			font-weight: 700;
			margin-bottom: 4px;
		}

		.section-headline {
			font-size: 1.4rem;
			margin-bottom: 4px;
		}

		.badge-row {
			display: flex;
			flex-wrap: wrap;
			gap: 0.65rem;
			margin-top: 0.65rem;
		}

		.badge-pill {
			display: inline-flex;
			align-items: center;
			padding: 0.42rem 0.8rem;
			border-radius: 999px;
			border: 1px solid var(--line);
			background: rgba(15, 23, 42, 0.8);
			color: #e2e8f0;
			font-size: 0.84rem;
			font-weight: 600;
		}

		div[data-baseweb="tab-list"] {
			flex-wrap: wrap;
			gap: 0.35rem;
		}

		button[data-baseweb="tab"] {
			border-radius: 999px;
			background: rgba(15, 23, 42, 0.72);
			border: 1px solid var(--line);
			color: #cbd5e1;
			white-space: normal;
			height: auto;
			min-height: 2.6rem;
			line-height: 1.2;
		}

		button[data-baseweb="tab"][aria-selected="true"] {
			background: linear-gradient(90deg, rgba(45, 212, 191, 0.18) 0%, rgba(56, 189, 248, 0.14) 100%);
			color: #f8fafc;
			border-color: rgba(45, 212, 191, 0.3);
		}

		[data-testid="stImage"] img, [data-testid="stDataFrame"] {
			border-radius: 18px;
			overflow: hidden;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


def load_json(path: Path) -> dict | None:
	if not path.exists():
		return None
	try:
		with path.open("r", encoding="utf-8") as fp:
			payload = json.load(fp)
	except (OSError, json.JSONDecodeError):
		return None
	return payload if isinstance(payload, dict) else None


def format_pct(value: float | None) -> str:
	if value is None:
		return "n/a"
	return f"{value:.2%}"


def format_float(value: float | None, digits: int = 4) -> str:
	if value is None:
		return "n/a"
	return f"{value:.{digits}f}"


def load_metric_catalog(classification_dir: Path) -> pd.DataFrame:
	candidates = [
		classification_dir / "test_results.json",
		classification_dir / "dl_results.json",
	]

	rows: list[dict] = []
	for candidate in candidates:
		payload = load_json(candidate)
		if not payload:
			continue

		for arch, metrics in payload.items():
			if not isinstance(metrics, Mapping):
				continue
			acc = metrics.get("test_accuracy", metrics.get("accuracy"))
			macro_f1 = metrics.get("macro_f1")
			if acc is None or macro_f1 is None:
				continue
			rows.append(
				{
					"architecture": str(arch),
					"accuracy": float(acc),
					"macro_f1": float(macro_f1),
					"weighted_f1": float(metrics.get("weighted_f1", 0.0)),
					"source": candidate.name,
				}
			)

	if not rows:
		return pd.DataFrame(columns=["architecture", "accuracy", "macro_f1", "weighted_f1", "source"])

	return pd.DataFrame(rows).sort_values(["source", "architecture"]).reset_index(drop=True)


def build_primary_metrics(metric_catalog: pd.DataFrame) -> pd.DataFrame:
	if metric_catalog.empty:
		return pd.DataFrame(columns=["architecture", "accuracy", "macro_f1", "weighted_f1", "source"])

	return (
		metric_catalog.sort_values(["macro_f1", "accuracy"], ascending=False)
		.drop_duplicates(subset=["architecture"], keep="first")
		.reset_index(drop=True)
	)


def pick_metric_row(metric_catalog: pd.DataFrame, arch: str, source: str | None = None) -> pd.Series | None:
	if metric_catalog.empty:
		return None

	subset = metric_catalog.loc[metric_catalog["architecture"] == arch]
	if source is not None:
		subset = subset.loc[subset["source"] == source]
	if subset.empty:
		return None
	return subset.sort_values(["macro_f1", "accuracy"], ascending=False).iloc[0]


def build_metric_discrepancy_note(metric_catalog: pd.DataFrame, arch: str) -> str | None:
	subset = metric_catalog.loc[metric_catalog["architecture"] == arch]
	if subset["source"].nunique() < 2:
		return None

	low = float(subset["accuracy"].min())
	high = float(subset["accuracy"].max())
	if high - low < 0.02:
		return None

	details = ", ".join(
		f"{row.source}: {row.accuracy:.2%}" for row in subset.sort_values("source").itertuples()
	)
	return (
		f"The repo contains multiple benchmark snapshots for {arch} ({details}). "
		"Treat them as separate evaluation runs rather than one deployed accuracy number."
	)


def pick_best_arch(
	checkpoints: dict[str, Path],
	metrics_df: pd.DataFrame,
	best_info: dict | None,
) -> str:
	if best_info and isinstance(best_info.get("best_arch"), str):
		best_arch = best_info["best_arch"]
		if best_arch in checkpoints:
			return best_arch

	if not metrics_df.empty:
		for arch in metrics_df["architecture"].tolist():
			if arch in checkpoints:
				return arch

	if "resnet50" in checkpoints:
		return "resnet50"

	return sorted(checkpoints.keys())[0]


def resolve_class_names(checkpoint_path: Path, data_dir: Path) -> list[str]:
	try:
		raw = torch.load(checkpoint_path, map_location="cpu")
		if isinstance(raw, Mapping):
			names = raw.get("class_names")
			if isinstance(names, list) and names:
				return [str(x) for x in names]
	except Exception:
		pass

	train_dir = data_dir / "train"
	if train_dir.exists():
		return datasets.ImageFolder(train_dir).classes

	return sorted(HAZARD_MAP.keys())


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
		raise ValueError(f"unsupported architecture: {arch}")

	return model


@st.cache_data(show_spinner=False)
def discover_dashboard_assets(classification_dir_str: str, data_dir_str: str) -> dict:
	classification_dir = Path(classification_dir_str)
	data_dir = Path(data_dir_str)

	checkpoints = discover_classification_checkpoints(classification_dir, SUPPORTED_ARCHES)
	if not checkpoints:
		raise FileNotFoundError(
			f"no classification checkpoints found under {classification_dir}. "
			"expected files like <arch>/<arch>_best.pth"
		)

	metric_catalog = load_metric_catalog(classification_dir)
	primary_metrics = build_primary_metrics(metric_catalog)
	best_info = read_best_model_info(classification_dir)
	best_arch = pick_best_arch(checkpoints, primary_metrics, best_info)
	best_checkpoint = checkpoints[best_arch]
	class_names = resolve_class_names(best_checkpoint, data_dir)

	return {
		"checkpoints": {k: str(v) for k, v in checkpoints.items()},
		"metric_catalog": metric_catalog.to_dict(orient="records"),
		"primary_metrics": primary_metrics.to_dict(orient="records"),
		"best_arch": best_arch,
		"best_checkpoint": str(best_checkpoint),
		"class_names": class_names,
		"best_info": best_info or {},
	}


@st.cache_data(show_spinner=False)
def load_dataset_profile(data_dir_str: str) -> dict:
	data_dir = Path(data_dir_str)
	rows: list[dict] = []
	for split_dir in sorted(data_dir.iterdir(), key=lambda p: p.name.lower()):
		if not split_dir.is_dir():
			continue
		for class_dir in sorted(split_dir.iterdir(), key=lambda p: p.name.lower()):
			if not class_dir.is_dir():
				continue
			count = sum(1 for p in class_dir.rglob("*") if p.is_file())
			rows.append(
				{
					"split": split_dir.name,
					"class_name": class_dir.name,
					"count": count,
				}
			)

	frame = pd.DataFrame(rows)
	totals = frame.groupby("split", dropna=False)["count"].sum().to_dict() if not frame.empty else {}
	return {
		"rows": frame.to_dict(orient="records"),
		"totals": {str(k): int(v) for k, v in totals.items()},
	}


@st.cache_data(show_spinner=False)
def load_supporting_metrics(project_root_str: str) -> dict:
	project_root = Path(project_root_str)
	return {
		"ann_results": load_json(project_root / "models" / "ann" / "ann_results.json") or {},
		"ann_results_18cls": load_json(project_root / "models" / "ann" / "ann_results_18cls.json") or {},
		"clustering_metrics": load_json(project_root / "models" / "clustering" / "clustering_metrics.json") or {},
		"clustering_results": load_json(project_root / "models" / "clustering" / "clustering_results.json") or {},
	}


@st.cache_resource(show_spinner=False)
def load_classifier(checkpoint_path: str, arch: str, class_names: tuple[str, ...], device_type: str) -> nn.Module:
	device = torch.device(device_type)
	model = build_model_for_inference(arch, num_classes=len(class_names)).to(device)
	raw_ckpt = torch.load(checkpoint_path, map_location=device)
	state = extract_model_state_dict(raw_ckpt)
	model.load_state_dict(state)
	model.eval()
	return model


@torch.inference_mode()
def infer_image(
	model: nn.Module,
	image: Image.Image,
	class_names: list[str],
	device: torch.device,
) -> dict:
	x = EVAL_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
	logits = model(x)
	probs = torch.softmax(logits, dim=1)[0].detach().cpu()

	top_k = min(5, len(class_names))
	values, indices = torch.topk(probs, k=top_k)

	top_predictions = []
	for score, idx in zip(values.tolist(), indices.tolist()):
		top_predictions.append(
			{
				"class_name": class_names[idx],
				"confidence": float(score),
			}
		)

	predicted_class = top_predictions[0]["class_name"]
	confidence = top_predictions[0]["confidence"]
	return {
		"class_name": predicted_class,
		"confidence": confidence,
		"top_predictions": top_predictions,
	}


def analyze_prediction(prediction: dict, threshold: float) -> dict:
	top_predictions = prediction.get("top_predictions", [])
	confidence = float(prediction.get("confidence", 0.0))
	second = float(top_predictions[1]["confidence"]) if len(top_predictions) > 1 else 0.0
	gap = max(confidence - second, 0.0)
	active_candidates = sum(1 for item in top_predictions if item["confidence"] >= 0.10)
	composite_suspected = confidence < 0.45 and (gap < 0.12 or active_candidates >= 3)
	below_threshold = confidence < threshold

	if composite_suspected:
		return {
			"tone": "warning",
			"headline": "Composite scene suspected",
			"detail": (
				"The upload looks like a mixed e-waste scene rather than a single isolated component. "
				"This classifier was trained on one label per image, so the top class is only a triage hint."
			),
			"needs_review": True,
		}

	if below_threshold:
		return {
			"tone": "warning",
			"headline": "Low-confidence triage",
			"detail": (
				f"Top confidence is {confidence:.2%}, below the operating threshold of {threshold:.2%}. "
				"Route this sample to assisted review before final disposal action."
			),
			"needs_review": True,
		}

	return {
		"tone": "success",
		"headline": "Within operating confidence band",
		"detail": (
			f"The classifier is above the {threshold:.2%} operating threshold with a top-vs-second gap of {gap:.2%}. "
			"This still assumes the upload contains one dominant component."
		),
		"needs_review": False,
	}


def analyze_scene_tiles(
	model: nn.Module,
	image: Image.Image,
	class_names: list[str],
	device: torch.device,
	grid_size: int,
	min_confidence: float,
) -> dict:
	width, height = image.size
	tiles: list[dict] = []

	for row in range(grid_size):
		for col in range(grid_size):
			x0 = int(width * col / grid_size)
			x1 = int(width * (col + 1) / grid_size)
			y0 = int(height * row / grid_size)
			y1 = int(height * (row + 1) / grid_size)
			tile_image = image.crop((x0, y0, x1, y1)).convert("RGB")
			prediction = infer_image(model=model, image=tile_image, class_names=class_names, device=device)
			hazard_level = HAZARD_MAP.get(prediction["class_name"], "UNKNOWN")
			tiles.append(
				{
					"tile_id": f"R{row + 1}C{col + 1}",
					"box": [x0, y0, x1, y1],
					"class_name": prediction["class_name"],
					"confidence": float(prediction["confidence"]),
					"hazard_level": hazard_level,
				}
			)

	notable_tiles = [tile for tile in tiles if tile["confidence"] >= min_confidence]
	if not notable_tiles:
		notable_tiles = sorted(tiles, key=lambda item: item["confidence"], reverse=True)[: min(3, len(tiles))]

	components: dict[str, dict] = {}
	hazard_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
	for tile in notable_tiles:
		hazard_counts[tile["hazard_level"]] = hazard_counts.get(tile["hazard_level"], 0) + 1
		component = tile["class_name"]
		if component not in components:
			components[component] = {
				"component": component,
				"hazard_level": tile["hazard_level"],
				"tiles_detected": 0,
				"peak_confidence": 0.0,
			}
		components[component]["tiles_detected"] += 1
		components[component]["peak_confidence"] = max(
			float(components[component]["peak_confidence"]),
			float(tile["confidence"]),
		)

	component_rows = sorted(
		components.values(),
		key=lambda item: (item["tiles_detected"], item["peak_confidence"]),
		reverse=True,
	)

	headline = (
		f"Composite evidence found across {len(component_rows)} component patterns"
		if len(component_rows) > 1
		else "Tile scan supports one dominant component pattern"
	)

	return {
		"grid_size": grid_size,
		"tiles": tiles,
		"components": component_rows,
		"hazard_counts": hazard_counts,
		"headline": headline,
		"tile_floor": min_confidence,
	}


def get_agent_decision(component: str, confidence: float, threshold: float) -> dict:
	if component not in HAZARD_MAP:
		return {
			"component": component,
			"hazard_level": "UNKNOWN",
			"material_profile": "not available",
			"disposal_pathway": "send to manual triage queue",
			"short_recommendation": "send to manual triage queue",
			"explanation": "predicted class is outside hazard policy map. manual review required.",
			"sdg_target": "SDG 12.4",
			"compliance_flag": False,
			"requires_human_review": True,
			"confidence_threshold": threshold,
			"confidence": confidence,
			"agent_mode": "fallback_policy_guardrail",
			"explanation_source": "rule-based",
			"tool_trace": [
				{
					"step": "hazard_lookup",
					"status": "blocked",
					"summary": "Predicted component is outside the registered hazard taxonomy.",
				}
			],
			"system_prompt_version": "fallback",
		}

	agent = EwasteDecisionAgent(confidence_threshold=threshold)
	return agent.run(AgentInput(component=component, confidence=confidence))


def pick_random_test_image(data_dir: Path) -> tuple[Image.Image | None, str | None]:
	test_root = data_dir / "test"
	if not test_root.exists():
		return None, None

	candidates: list[Path] = []
	for p in test_root.rglob("*"):
		if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
			candidates.append(p)

	if not candidates:
		return None, None
	selected = random.choice(candidates)
	return Image.open(selected).convert("RGB"), selected.name


def render_metric_tile(label: str, value: str, detail: str, tone: str = "neutral") -> None:
	st.markdown(f"**{label}**")
	st.markdown(f"### {value}")
	st.caption(detail)


def render_panel(title: str, value: str, copy: str) -> None:
	st.markdown(f"**{title}**")
	st.markdown(f"#### {value}")
	st.caption(copy)


def render_banner(title: str, copy: str, tone: str = "neutral") -> None:
	message = f"**{title}**\n\n{copy}"
	if tone == "success":
		st.success(message)
	elif tone == "warning":
		st.warning(message)
	elif tone == "danger":
		st.error(message)
	else:
		st.info(message)


def render_probability_rows(top_predictions: list[dict]) -> None:
	for item in top_predictions:
		confidence = float(item["confidence"])
		left, right = st.columns([4, 1])
		with left:
			st.markdown(f"**{item['class_name']}**")
		with right:
			st.markdown(f"**{confidence:.2%}**")
		st.progress(max(0, min(100, int(round(confidence * 100)))))


def render_section_intro(kicker: str, headline: str, copy: str) -> None:
	st.caption(kicker.upper())
	st.subheader(headline)
	st.write(copy)


def render_badge_row(badges: list[str]) -> None:
	st.caption(" | ".join(badges))


def build_operational_checklist(decision: dict, diagnostics: dict) -> list[str]:
	hazard = decision.get("hazard_level", "UNKNOWN")
	checklist = [
		"Log the uploaded image and confidence distribution for auditability.",
		"Retain the model output alongside the final disposal decision.",
	]
	if diagnostics.get("needs_review", False):
		checklist.append("Escalate the sample to manual review before any disposal action is finalized.")
	if hazard == "HIGH":
		checklist.append("Isolate the item from general waste and route through certified hazardous e-waste handling.")
	elif hazard == "MEDIUM":
		checklist.append("Route through controlled appliance or component recovery with condition verification.")
	elif hazard == "LOW":
		checklist.append("Route to the appropriate low-risk recovery stream after label confirmation.")
	return checklist


def call_groq_text(prompt: str, system_prompt: str, model_name: str | None = None) -> str:
	api_key = os.getenv("GROQ_API_KEY")
	if not api_key:
		raise RuntimeError("GROQ_API_KEY is not set.")

	payload = {
		"model": model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
		"temperature": 0.2,
		"messages": [
			{
				"role": "system",
				"content": system_prompt,
			},
			{
				"role": "user",
				"content": prompt,
			},
		],
	}

	req = urllib_request.Request(
		url="https://api.groq.com/openai/v1/chat/completions",
		data=json.dumps(payload).encode("utf-8"),
		headers={
			"Authorization": f"Bearer {api_key}",
			"Content-Type": "application/json",
			"Accept": "application/json",
			"User-Agent": "ewaste-vit-project/1.0 (+streamlit-dashboard)",
		},
		method="POST",
	)

	with urllib_request.urlopen(req, timeout=45) as response:
		body = json.loads(response.read().decode("utf-8"))

	choices = body.get("choices", [])
	if not choices:
		raise RuntimeError("Groq returned no choices.")

	content = choices[0].get("message", {}).get("content", "")
	if not content:
		raise RuntimeError("Groq returned empty content.")
	return str(content).strip()


def build_results_summary_prompt(
	best_arch: str,
	best_archived_row: pd.Series | None,
	best_script_row: pd.Series | None,
	discrepancy_note: str | None,
) -> str:
	parts = [
		f"Best deployed architecture: {best_arch}.",
		f"Archived benchmark accuracy: {format_pct(float(best_archived_row['accuracy'])) if best_archived_row is not None else 'n/a'}.",
		f"Script benchmark accuracy: {format_pct(float(best_script_row['accuracy'])) if best_script_row is not None else 'n/a'}.",
	]
	if discrepancy_note:
		parts.append(f"Important caveat: {discrepancy_note}")
	parts.append(
		"Write a concise research-paper results paragraph that is honest about benchmark variance and explains why mixed-scene uploads challenge a single-label classifier."
	)
	return "\n".join(parts)


def build_discussion_prompt(
	ann_results_18cls: dict,
	clustering_results: dict,
) -> str:
	return (
		f"Hazard ANN accuracy: {format_pct(float(ann_results_18cls.get('hazard_class_accuracy'))) if ann_results_18cls else 'n/a'}.\n"
		f"Clustering NMI: {format_float(float(clustering_results.get('normalized_mutual_info'))) if clustering_results else 'n/a'}.\n"
		f"Cluster count: {clustering_results.get('n_clusters', 'n/a')}.\n"
		"Write a short discussion paragraph for a research paper that explains what these supporting analyses add beyond the classifier, "
		"and mention limitations without overselling the results."
	)


def _is_streamlit_runtime_active() -> bool:
	try:
		from streamlit.runtime.scriptrunner import get_script_run_ctx

		return get_script_run_ctx() is not None
	except Exception:
		return False


def main() -> None:
	st.set_page_config(page_title="E-Waste Operations Console", layout="wide")
	inject_styles()

	runtime = detect_runtime()
	classification_dir = PROJECT_ROOT / "models" / "classification"
	data_dir = PROJECT_ROOT / "data"
	ann_dir = PROJECT_ROOT / "models" / "ann"
	clustering_dir = PROJECT_ROOT / "models" / "clustering"

	with st.sidebar:
		st.markdown("### Operations Console")
		st.caption("Classification, review routing, benchmark records, and data registry for the current research system.")
		confidence_threshold = st.slider("Human review threshold", 0.50, 0.95, 0.70, 0.01)
		enable_scene_scan = st.checkbox("Enable composite scene scan", value=True)
		scene_grid = st.selectbox("Scene scan grid", options=[2, 3], index=1)
		scene_tile_floor = st.slider("Tile evidence floor", 0.10, 0.80, 0.30, 0.05)
		workspace = st.selectbox("View", options=WORKSPACES, index=0)
		st.markdown("### Runtime")
		st.caption(f"Device: {runtime.device.type}")
		st.caption(f"GPU: {runtime.gpu_name or 'not detected'}")
		st.caption(f"VRAM: {f'{runtime.vram_gb:.2f} GB' if runtime.vram_gb is not None else 'n/a'}")
		st.info("This model is strongest on isolated single-component images. Mixed scenes should be treated as triage support, not detection output.")

	try:
		assets = discover_dashboard_assets(str(classification_dir), str(data_dir))
	except Exception as exc:
		st.error(f"failed to load model assets: {exc}")
		st.stop()

	class_names = assets["class_names"]
	best_arch = assets["best_arch"]
	best_checkpoint = assets["best_checkpoint"]
	metric_catalog = pd.DataFrame(assets["metric_catalog"])
	primary_metrics = pd.DataFrame(assets["primary_metrics"])
	dataset_profile_payload = load_dataset_profile(str(data_dir))
	dataset_profile = pd.DataFrame(dataset_profile_payload["rows"])
	supporting = load_supporting_metrics(str(PROJECT_ROOT))
	best_archived_row = pick_metric_row(metric_catalog, best_arch, "dl_results.json")
	best_script_row = pick_metric_row(metric_catalog, best_arch, "test_results.json")
	discrepancy_note = build_metric_discrepancy_note(metric_catalog, best_arch)

	model = load_classifier(
		checkpoint_path=best_checkpoint,
		arch=best_arch,
		class_names=tuple(class_names),
		device_type=runtime.device.type,
	)

	ann_results = supporting["ann_results"]
	ann_results_18cls = supporting["ann_results_18cls"]
	clustering_metrics = supporting["clustering_metrics"]
	clustering_results = supporting["clustering_results"]

	st.caption("CURRENT CLASSIFICATION SYSTEM")
	st.title("E-Waste Operations Console")
	st.write(
		"This interface presents the current single-label classifier, policy outputs, benchmark records, and supporting analytics "
		"in one place. It also marks when an input is outside the assumptions of the training data, especially for mixed-object scenes."
	)

	if discrepancy_note:
		render_banner(
			"Benchmark context",
			discrepancy_note + " The collage example is a mixed-object scene, so a low live confidence is an uncertainty signal rather than a reliable single-class decision.",
			tone="warning",
		)

	top_row = st.columns(3)
	with top_row[0]:
		render_metric_tile("Deployed architecture", best_arch, f"checkpoint: {Path(best_checkpoint).name}", "neutral")
	with top_row[1]:
		render_metric_tile("Archived benchmark", format_pct(float(best_archived_row["accuracy"])) if best_archived_row is not None else "n/a", "source: dl_results.json", "success")
	with top_row[2]:
		render_metric_tile("Script benchmark", format_pct(float(best_script_row["accuracy"])) if best_script_row is not None else "n/a", "source: test_results.json", "warning")

	second_row = st.columns(2)
	with second_row[0]:
		render_metric_tile("Hazard ANN accuracy", format_pct(float(ann_results_18cls.get("hazard_class_accuracy"))) if ann_results_18cls else "n/a", "18-class hazard snapshot", "success")
	with second_row[1]:
		render_metric_tile("Cluster groups", str(clustering_results.get("n_clusters", "n/a")), "unsupervised structure available", "neutral")

	if "pending_image" not in st.session_state:
		st.session_state["pending_image"] = None
	if "pending_image_label" not in st.session_state:
		st.session_state["pending_image_label"] = None
	if "pending_image_source" not in st.session_state:
		st.session_state["pending_image_source"] = None
	if "last_upload_token" not in st.session_state:
		st.session_state["last_upload_token"] = None
	if "llm_results_summary" not in st.session_state:
		st.session_state["llm_results_summary"] = None
	if "llm_discussion_summary" not in st.session_state:
		st.session_state["llm_discussion_summary"] = None

	if workspace == "Operations":
		render_section_intro(
			"Inference",
			"Operational inference review",
			"Submit an image, inspect confidence distribution, and review composite-scene cues before any downstream routing decision.",
		)

		left, right = st.columns([1.08, 0.92], gap="large")
		with left:
			uploaded = st.file_uploader(
				"Image intake",
				type=["jpg", "jpeg", "png", "webp"],
				accept_multiple_files=False,
				help="Best performance comes from a single dominant component in frame.",
			)
			upload_token = f"{uploaded.name}:{uploaded.size}" if uploaded is not None else None
			a1, a2, a3 = st.columns(3)
			with a1:
				sample_btn = st.button("Load test image")
			with a2:
				clear_btn = st.button("Clear")
			with a3:
				run_btn = st.button("Run inference", type="primary")

			if uploaded is not None and upload_token != st.session_state.get("last_upload_token"):
				st.session_state["pending_image"] = Image.open(uploaded).convert("RGB")
				st.session_state["pending_image_label"] = uploaded.name
				st.session_state["pending_image_source"] = "upload"
				st.session_state["last_upload_token"] = upload_token
				st.session_state.pop("last_result", None)
			elif sample_btn:
				sample_image, sample_name = pick_random_test_image(data_dir)
				st.session_state["pending_image"] = sample_image
				st.session_state["pending_image_label"] = sample_name
				st.session_state["pending_image_source"] = "sample"
				st.session_state.pop("last_result", None)
			elif clear_btn:
				st.session_state["pending_image"] = None
				st.session_state["pending_image_label"] = None
				st.session_state["pending_image_source"] = None
				st.session_state["last_upload_token"] = None
				st.session_state.pop("last_result", None)

			image: Image.Image | None = st.session_state.get("pending_image")
			if image is not None:
				st.image(image, width="stretch")
				render_badge_row(
					[
						f"source: {st.session_state.get('pending_image_label') or 'uploaded image'}",
						f"canvas: {image.width} x {image.height}",
						"single-label classifier",
					]
				)
			else:
				st.info("Load a test image or upload a sample to start inference.")

		image = st.session_state.get("pending_image")
		if image is not None and run_btn:
			start = time.time()
			prediction = infer_image(model=model, image=image, class_names=class_names, device=runtime.device)
			elapsed_ms = int((time.time() - start) * 1000)
			diagnostics = analyze_prediction(prediction, confidence_threshold)
			decision = get_agent_decision(prediction["class_name"], prediction["confidence"], confidence_threshold)
			scene_analysis = None
			if enable_scene_scan:
				scene_analysis = analyze_scene_tiles(
					model=model,
					image=image,
					class_names=class_names,
					device=runtime.device,
					grid_size=scene_grid,
					min_confidence=scene_tile_floor,
				)
			st.session_state["last_result"] = {
				"prediction": prediction,
				"decision": decision,
				"diagnostics": diagnostics,
				"scene_analysis": scene_analysis,
				"elapsed_ms": elapsed_ms,
			}

		with right:
			result = st.session_state.get("last_result")
			if not result:
				render_panel("Inference state", "Awaiting input", "Run inference to populate confidence, hazard, and scene evidence.")
			else:
				prediction = result["prediction"]
				decision = result["decision"]
				diagnostics = result["diagnostics"]
				r1, r2, r3 = st.columns(3)
				with r1:
					render_panel("Predicted class", prediction["class_name"], "single-label top prediction")
				with r2:
					render_panel("Confidence", format_pct(prediction["confidence"]), "measured on this input")
				with r3:
					render_panel("Latency", f"{result['elapsed_ms']} ms", "single-image forward pass")
				st.caption(f"Hazard band: {decision.get('hazard_level', 'UNKNOWN')}")
				render_banner(diagnostics["headline"], diagnostics["detail"], diagnostics["tone"])
				st.markdown("#### Top-5 Class Scores")
				render_probability_rows(prediction["top_predictions"])

		result = st.session_state.get("last_result")
		if result:
			scene_analysis = result.get("scene_analysis")
			st.markdown("### Composite Scene Review")
			if enable_scene_scan and scene_analysis:
				render_banner(
					scene_analysis["headline"],
					f"Tile scan uses a {scene_analysis['grid_size']}x{scene_analysis['grid_size']} grid and reports tiles above {scene_analysis['tile_floor']:.2%}. This is a review aid, not object detection.",
					"neutral",
				)
				s1, s2 = st.columns([1.05, 0.95], gap="large")
				with s1:
					component_frame = pd.DataFrame(scene_analysis["components"])
					if component_frame.empty:
						st.info("No tile met the evidence floor. Lower the tile threshold to inspect weaker scene cues.")
					else:
						component_frame["peak_confidence"] = component_frame["peak_confidence"].map(lambda x: f"{x:.2%}")
						st.dataframe(component_frame, width="stretch", hide_index=True)
				with s2:
					hazard_counts = scene_analysis["hazard_counts"]
					h1, h2, h3 = st.columns(3)
					with h1:
						render_metric_tile("High-risk tiles", str(hazard_counts.get("HIGH", 0)), "tile-level hazard evidence", "danger")
					with h2:
						render_metric_tile("Medium-risk tiles", str(hazard_counts.get("MEDIUM", 0)), "tile-level hazard evidence", "warning")
					with h3:
						render_metric_tile("Low-risk tiles", str(hazard_counts.get("LOW", 0)), "tile-level hazard evidence", "success")
				tile_columns = st.columns(scene_analysis["grid_size"])
				for idx, tile in enumerate(scene_analysis["tiles"]):
					crop = image.crop(tuple(tile["box"]))
					caption = f"{tile['tile_id']} | {tile['class_name']} | {tile['confidence']:.1%}"
					with tile_columns[idx % scene_analysis["grid_size"]]:
						st.image(crop, caption=caption, width="stretch")
			elif enable_scene_scan:
				st.info("Run inference to generate composite-scene evidence.")
			else:
				st.info("Enable composite scene scan in the sidebar to inspect mixed-object inputs.")

	if workspace == "Policy":
		render_section_intro(
			"Decision",
			"Policy and routing review",
			"This panel exposes the actual decision output: hazard lookup, compliance signal, recommendation mode, and the trace used to assemble the final response.",
		)
		result = st.session_state.get("last_result")
		if not result:
			st.info("Run inference first to generate hazard and routing guidance.")
		else:
			decision = result["decision"]
			diagnostics = result["diagnostics"]
			p1, p2, p3, p4 = st.columns(4)
			with p1:
				render_metric_tile("Hazard level", decision.get("hazard_level", "n/a"), "mapped from policy taxonomy", "danger" if decision.get("hazard_level") == "HIGH" else "warning")
			with p2:
				render_metric_tile("SDG target", decision.get("sdg_target", "n/a"), "policy alignment output", "neutral")
			with p3:
				render_metric_tile("Human review", "Required" if decision.get("requires_human_review", True) else "Not required", f"threshold: {decision.get('confidence_threshold', confidence_threshold):.2%}", "warning" if decision.get("requires_human_review", True) else "success")
			with p4:
				render_metric_tile("Decision mode", decision.get("agent_mode", "n/a"), f"provider: {decision.get('llm_provider', 'none')} | source: {decision.get('explanation_source', 'n/a')}", "neutral")
			l1, l2 = st.columns([1.05, 0.95], gap="large")
			with l1:
				render_panel("Recommended pathway", decision.get("short_recommendation", "n/a"), "routing output after tool execution")
				st.markdown("#### Material Profile")
				st.write(decision.get("material_profile", "n/a"))
				st.markdown("#### Decision Rationale")
				st.write(decision.get("explanation", "n/a"))
				if decision.get("llm_error"):
					st.warning(f"LLM augmentation failed and the system fell back to deterministic reasoning: {decision['llm_error']}")
			with l2:
				render_banner("Operating interpretation", "Tie the routing decision to confidence. Confident single-component inputs can proceed automatically; ambiguous or composite scenes should stop at triage.", "neutral")
				render_badge_row(
					[
						f"compliance: {'ready' if decision.get('compliance_flag', False) else 'escalate'}",
						f"mode: {decision.get('agent_mode', 'n/a')}",
						f"provider: {decision.get('llm_provider', 'none')}",
						f"source: {decision.get('explanation_source', 'n/a')}",
					]
				)
				st.markdown("#### Operational Checklist")
				for item in build_operational_checklist(decision, diagnostics):
					st.markdown(f"- {item}")
			trace = pd.DataFrame(decision.get("tool_trace", []))
			if not trace.empty:
				st.markdown("#### Tool Execution Trace")
				st.dataframe(trace, width="stretch", hide_index=True)
			with st.expander("Raw decision payload"):
				st.json(decision)

	if workspace == "Benchmarks":
		render_section_intro(
			"Benchmarks",
			"Evaluation records",
			"This section surfaces benchmark tables, confusion matrices, training curves, and interpretability artifacts already present in the repository.",
		)
		if discrepancy_note:
			render_banner("Benchmark caution", discrepancy_note + " Do not present the archived and script metrics as if they were one experiment.", "warning")
		b1, b2 = st.columns([0.95, 1.05], gap="large")
		with b1:
			if primary_metrics.empty:
				st.info("No benchmark metrics JSON found in models/classification.")
			else:
				display_primary = primary_metrics.copy()
				display_primary["accuracy"] = display_primary["accuracy"].map(lambda x: f"{x:.2%}")
				display_primary["macro_f1"] = display_primary["macro_f1"].map(lambda x: f"{x:.4f}")
				display_primary["weighted_f1"] = display_primary["weighted_f1"].map(lambda x: f"{x:.4f}")
				st.dataframe(display_primary, width="stretch", hide_index=True)
				source_options = sorted(metric_catalog["source"].unique().tolist())
				selected_source = st.selectbox("Metric source", options=source_options)
				source_frame = metric_catalog.loc[metric_catalog["source"] == selected_source]
				st.bar_chart(source_frame.set_index("architecture")[["accuracy", "macro_f1"]], width="stretch")
		with b2:
			st.markdown(
				"""
				- `dl_results.json` contains the stronger archived benchmark snapshot shown in the confusion matrix image.
				- `test_results.json` contains the script-generated benchmark for the current classification pipeline when that file is present.
				- Mixed-scene failure is expected because the dataset is structured as one class per image folder.
				- A low live confidence is an uncertainty signal, not proof that the model is confidently wrong.
				"""
			)
		i1, i2 = st.columns(2, gap="large")
		with i1:
			st.markdown("#### Confusion Matrices")
			st.image(str(classification_dir / "graphs" / "confusion_matrices_18cls.png"), width="stretch")
			st.markdown("#### Per-Class F1 Comparison")
			st.image(str(classification_dir / "graphs" / "per_class_f1_comparison.png"), width="stretch")
		with i2:
			st.markdown("#### Training Curves")
			st.image(str(classification_dir / "graphs" / "training_curves_18cls.png"), width="stretch")
			st.markdown("#### Interpretability Gallery")
			st.image(str(classification_dir / "graphs" / "gradcam_all_classes.png"), width="stretch")

		st.markdown("#### Results Drafting")
		if st.button("Draft results paragraph", key="generate_results_paragraph"):
			try:
				with st.spinner("Generating results paragraph..."):
					st.session_state["llm_results_summary"] = call_groq_text(
						prompt=build_results_summary_prompt(
							best_arch=best_arch,
							best_archived_row=best_archived_row,
							best_script_row=best_script_row,
							discrepancy_note=discrepancy_note,
						),
						system_prompt=(
							"You are helping write a rigorous research paper. "
							"Be concise, factual, publication-ready, and do not exaggerate novelty or performance."
						),
					)
			except Exception as exc:
				st.error(f"Groq generation failed: {exc}")
		if st.session_state.get("llm_results_summary"):
			st.write(st.session_state["llm_results_summary"])

	if workspace == "Analytics":
		render_section_intro(
			"Analytics",
			"Supporting analytical outputs",
			"These panels bring the hazard ANN and clustering outputs into the same interface so the analytical scope of the project is visible alongside the classifier.",
		)
		a1, a2, a3, a4 = st.columns(4)
		with a1:
			render_metric_tile("Hazard class accuracy", format_pct(float(ann_results_18cls.get("hazard_class_accuracy"))) if ann_results_18cls else "n/a", "18-class ANN hazard snapshot", "success")
		with a2:
			render_metric_tile("Regression R2", format_float(float(ann_results.get("regression_metrics", {}).get("r2"))) if ann_results else "n/a", "hazard severity regression", "neutral")
		with a3:
			render_metric_tile("Silhouette", format_float(float(clustering_metrics.get("silhouette_score"))) if clustering_metrics else "n/a", "cluster separation score", "warning")
		with a4:
			render_metric_tile("NMI", format_float(float(clustering_results.get("normalized_mutual_info"))) if clustering_results else "n/a", "alignment with known labels", "neutral")
		g1, g2 = st.columns(2, gap="large")
		with g1:
			st.markdown("#### Hazard Model Views")
			st.image(str(ann_dir / "feature_importance.png"), width="stretch")
			st.image(str(ann_dir / "hazard_class_confusion.png"), width="stretch")
		with g2:
			st.markdown("#### Unsupervised Structure")
			st.image(str(clustering_dir / "tsne_by_class.png"), width="stretch")
			st.image(str(clustering_dir / "tsne_by_hazard.png"), width="stretch")

		st.markdown("#### Discussion Drafting")
		if st.button("Draft discussion paragraph", key="generate_discussion_paragraph"):
			try:
				with st.spinner("Generating discussion paragraph..."):
					st.session_state["llm_discussion_summary"] = call_groq_text(
						prompt=build_discussion_prompt(
							ann_results_18cls=ann_results_18cls,
							clustering_results=clustering_results,
						),
						system_prompt=(
							"You are helping write the discussion section of a research paper. "
							"Explain value, limitations, and future work in a sober academic tone."
						),
					)
			except Exception as exc:
				st.error(f"Groq generation failed: {exc}")
		if st.session_state.get("llm_discussion_summary"):
			st.write(st.session_state["llm_discussion_summary"])

	if workspace == "Registry":
		render_section_intro(
			"Registry",
			"Model and data inventory",
			"This view inventories checkpoints, dataset balance, and the hazard taxonomy used by the policy layer.",
		)
		registry_rows = []
		for arch, path in assets["checkpoints"].items():
			registry_rows.append(
				{
					"architecture": arch,
					"checkpoint_path": str(path).replace("\\", "/"),
					"status": "active" if arch == best_arch else "available",
				}
			)
		st.dataframe(pd.DataFrame(registry_rows), width="stretch", hide_index=True)
		if not dataset_profile.empty:
			split_choice = st.selectbox("Dataset split", options=sorted(dataset_profile["split"].unique().tolist()))
			split_frame = dataset_profile.loc[dataset_profile["split"] == split_choice].copy()
			st.dataframe(split_frame.sort_values("class_name"), width="stretch", hide_index=True)
			st.bar_chart(split_frame.set_index("class_name")[["count"]], width="stretch")
			t1, t2, t3 = st.columns(3)
			with t1:
				render_metric_tile("Train images", str(dataset_profile_payload["totals"].get("train", "n/a")), "single-label foldered data", "neutral")
			with t2:
				render_metric_tile("Validation images", str(dataset_profile_payload["totals"].get("val", "n/a")), "held-out tuning split", "neutral")
			with t3:
				render_metric_tile("Test images", str(dataset_profile_payload["totals"].get("test", "n/a")), "held-out evaluation split", "neutral")
		taxonomy_rows = []
		for component in sorted(HAZARD_MAP.keys(), key=str.lower):
			taxonomy_rows.append(
				{
					"component": component,
					"hazard_level": HAZARD_MAP[component],
					"material_profile": MATERIAL_MAP.get(component, "n/a"),
					"disposal_pathway": DISPOSAL_MAP.get(component, "n/a"),
				}
			)
		st.markdown("#### Hazard Taxonomy")
		st.dataframe(pd.DataFrame(taxonomy_rows), width="stretch", hide_index=True)
		render_banner(
			"Deployment scope",
			"The current vision stack is a single-label component classifier. Present the scene scan as composite-image triage and reserve true detection or multi-label classification as future work.",
			"warning",
		)


if __name__ == "__main__":
	if not _is_streamlit_runtime_active():
		print("This is a Streamlit app. Launch it with: streamlit run dashboard/app.py")
		sys.exit(0)
	main()
