from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import json
import random
import sys
import time

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
from agent.tools import HAZARD_MAP  # noqa: E402
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
	"HIGH": "#dc2626",
	"MEDIUM": "#f59e0b",
	"LOW": "#10b981",
	"UNKNOWN": "#64748b",
}


def inject_styles() -> None:
	st.markdown(
		"""
		<style>
		@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

		:root {
			--ink: #0f172a;
			--muted: #475569;
			--card: #ffffff;
			--line: #dbe7ef;
			--accent: #0f766e;
			--accent-soft: #ccfbf1;
		}

		html, body, [data-testid="stAppViewContainer"] {
			font-family: 'Manrope', sans-serif;
			color: var(--ink);
			background:
				radial-gradient(1200px 420px at 8% -10%, #d9f7f1 0%, transparent 45%),
				radial-gradient(900px 360px at 94% 0%, #e5eef7 0%, transparent 50%),
				#f5f8fb;
		}

		h1, h2, h3 {
			font-family: 'Space Grotesk', sans-serif;
			letter-spacing: 0.01em;
			color: var(--ink);
		}

		.hero {
			border: 1px solid var(--line);
			border-radius: 18px;
			padding: 18px 22px;
			background: linear-gradient(135deg, #ffffff 0%, #f8fffd 55%, #eef7ff 100%);
			box-shadow: 0 10px 30px rgba(15, 23, 42, 0.07);
			margin-bottom: 8px;
		}

		.hero-kicker {
			text-transform: uppercase;
			font-size: 0.74rem;
			letter-spacing: 0.12em;
			font-weight: 700;
			color: #0f766e;
			margin-bottom: 6px;
		}

		.hero h1 {
			margin: 0;
			font-size: 1.9rem;
		}

		.hero p {
			margin-top: 8px;
			margin-bottom: 0;
			color: var(--muted);
			font-size: 0.96rem;
		}

		.glass-card {
			border: 1px solid var(--line);
			border-radius: 14px;
			padding: 14px;
			background: rgba(255, 255, 255, 0.92);
			box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
		}

		.kpi-value {
			font-size: 1.4rem;
			font-weight: 800;
			color: var(--ink);
		}

		.kpi-label {
			color: var(--muted);
			font-size: 0.82rem;
			text-transform: uppercase;
			letter-spacing: 0.08em;
			font-weight: 700;
		}

		.hazard-pill {
			display: inline-block;
			font-weight: 800;
			color: white;
			border-radius: 999px;
			padding: 0.28rem 0.72rem;
			font-size: 0.78rem;
			letter-spacing: 0.04em;
			text-transform: uppercase;
		}

		[data-testid="stSidebar"] {
			border-right: 1px solid var(--line);
			background: linear-gradient(180deg, #f8fafc 0%, #f2f7fb 100%);
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


def load_classification_metrics(classification_dir: Path) -> pd.DataFrame:
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

	frame = pd.DataFrame(rows)
	frame = frame.sort_values(["macro_f1", "accuracy"], ascending=False)
	frame = frame.drop_duplicates(subset=["architecture"], keep="first").reset_index(drop=True)
	return frame


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

	metrics_df = load_classification_metrics(classification_dir)
	best_info = read_best_model_info(classification_dir)
	best_arch = pick_best_arch(checkpoints, metrics_df, best_info)
	best_checkpoint = checkpoints[best_arch]
	class_names = resolve_class_names(best_checkpoint, data_dir)

	return {
		"checkpoints": {k: str(v) for k, v in checkpoints.items()},
		"metrics": metrics_df.to_dict(orient="records"),
		"best_arch": best_arch,
		"best_checkpoint": str(best_checkpoint),
		"class_names": class_names,
		"best_info": best_info or {},
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
			"system_prompt_version": "fallback",
		}

	agent = EwasteDecisionAgent(confidence_threshold=threshold)
	return agent.run(AgentInput(component=component, confidence=confidence))


def pick_random_test_image(data_dir: Path) -> Image.Image | None:
	test_root = data_dir / "test"
	if not test_root.exists():
		return None

	candidates: list[Path] = []
	for p in test_root.rglob("*"):
		if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
			candidates.append(p)

	if not candidates:
		return None
	selected = random.choice(candidates)
	return Image.open(selected).convert("RGB")


def render_kpi_card(label: str, value: str) -> None:
	st.markdown(
		f"""
		<div class="glass-card">
			<div class="kpi-label">{label}</div>
			<div class="kpi-value">{value}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


def main() -> None:
	st.set_page_config(page_title="E-Waste Research Intelligence Console", layout="wide")
	inject_styles()

	runtime = detect_runtime()
	classification_dir = PROJECT_ROOT / "models" / "classification"
	data_dir = PROJECT_ROOT / "data"

	st.markdown(
		"""
		<div class="hero">
			<div class="hero-kicker">Research Operations Console</div>
			<h1>E-Waste Intelligence Platform</h1>
			<p>
				Industry-grade inference workflow: vision classification, hazard stratification,
				compliance-aware disposal pathway, and transparent agent reasoning.
			</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

	with st.sidebar:
		st.header("Run Controls")
		confidence_threshold = st.slider(
			"human review threshold",
			min_value=0.50,
			max_value=0.95,
			value=0.70,
			step=0.01,
			help="predictions below this confidence are flagged for human review",
		)
		st.caption(f"runtime device: {runtime.device.type}")
		if runtime.gpu_name:
			st.caption(f"gpu: {runtime.gpu_name}")
			if runtime.vram_gb is not None:
				st.caption(f"vram: {runtime.vram_gb:.2f} GB")

	try:
		assets = discover_dashboard_assets(str(classification_dir), str(data_dir))
	except Exception as exc:
		st.error(f"failed to load model assets: {exc}")
		st.stop()

	class_names = assets["class_names"]
	best_arch = assets["best_arch"]
	best_checkpoint = assets["best_checkpoint"]

	model = load_classifier(
		checkpoint_path=best_checkpoint,
		arch=best_arch,
		class_names=tuple(class_names),
		device_type=runtime.device.type,
	)

	metrics_df = pd.DataFrame(assets["metrics"])

	k1, k2, k3, k4 = st.columns(4)
	with k1:
		render_kpi_card("deployed architecture", best_arch)
	with k2:
		render_kpi_card("classes", str(len(class_names)))
	with k3:
		render_kpi_card("checkpoints discovered", str(len(assets["checkpoints"])))
	with k4:
		if not metrics_df.empty and best_arch in set(metrics_df["architecture"]):
			row = metrics_df.loc[metrics_df["architecture"] == best_arch].iloc[0]
			render_kpi_card("macro-f1", f"{row['macro_f1']:.4f}")
		else:
			render_kpi_card("macro-f1", "n/a")

	tab_infer, tab_agent, tab_registry = st.tabs(
		["Image Inference", "Agentic Decision", "Model Registry"]
	)

	with tab_infer:
		left, right = st.columns([1.15, 1], gap="large")

		with left:
			st.subheader("Input Image")
			if "pending_image" not in st.session_state:
				st.session_state["pending_image"] = None

			uploaded = st.file_uploader(
				"Upload an e-waste image",
				type=["jpg", "jpeg", "png", "webp"],
				accept_multiple_files=False,
			)

			sample_btn = st.button("Use random test sample")
			clear_btn = st.button("Clear image")

			if uploaded is not None:
				st.session_state["pending_image"] = Image.open(uploaded).convert("RGB")
			elif sample_btn:
				st.session_state["pending_image"] = pick_random_test_image(data_dir)
			elif clear_btn:
				st.session_state["pending_image"] = None

			image: Image.Image | None = st.session_state.get("pending_image")

			if image is not None:
				st.image(image, use_container_width=True)
				if st.button("Run Classification", type="primary"):
					start = time.time()
					prediction = infer_image(
						model=model,
						image=image,
						class_names=class_names,
						device=runtime.device,
					)
					elapsed_ms = int((time.time() - start) * 1000)
					decision = get_agent_decision(
						component=prediction["class_name"],
						confidence=prediction["confidence"],
						threshold=confidence_threshold,
					)
					st.session_state["last_result"] = {
						"prediction": prediction,
						"decision": decision,
						"elapsed_ms": elapsed_ms,
					}
			else:
				st.info("Upload an image or use a random test sample to run inference.")

		with right:
			st.subheader("Prediction Output")
			result = st.session_state.get("last_result")
			if not result:
				st.caption("No inference has been run in this session yet.")
			else:
				prediction = result["prediction"]
				decision = result["decision"]

				pred_cls = prediction["class_name"]
				conf = prediction["confidence"]
				hazard = decision.get("hazard_level", "UNKNOWN")
				color = HAZARD_COLOR.get(hazard, HAZARD_COLOR["UNKNOWN"])

				c1, c2 = st.columns(2)
				with c1:
					st.metric("predicted class", pred_cls)
				with c2:
					st.metric("confidence", f"{conf:.2%}")

				st.markdown(
					f"<span class='hazard-pill' style='background:{color};'>{hazard} hazard</span>",
					unsafe_allow_html=True,
				)
				st.caption(f"inference latency: {result['elapsed_ms']} ms")

				st.markdown("#### Top-5 class probabilities")
				top_df = pd.DataFrame(prediction["top_predictions"])
				top_df["confidence"] = top_df["confidence"].map(lambda x: round(x, 4))
				st.dataframe(top_df, use_container_width=True, hide_index=True)

	with tab_agent:
		st.subheader("Hazard Intelligence and Disposal Pathway")
		result = st.session_state.get("last_result")
		if not result:
			st.info("Run image inference first to generate a policy decision.")
		else:
			decision = result["decision"]
			a1, a2, a3 = st.columns(3)
			with a1:
				st.metric("hazard level", decision.get("hazard_level", "n/a"))
			with a2:
				st.metric("sdg target", decision.get("sdg_target", "n/a"))
			with a3:
				st.metric(
					"human review",
					"required" if decision.get("requires_human_review", True) else "not required",
				)

			st.markdown("#### Recommended Disposal Pathway")
			st.success(decision.get("short_recommendation", "n/a"))

			st.markdown("#### Material Profile")
			st.write(decision.get("material_profile", "n/a"))

			st.markdown("#### Agent Explanation")
			st.write(decision.get("explanation", "n/a"))

			st.markdown("#### Agentic Processing Trace")
			st.markdown(
				"""
				1. Visual classifier predicts component class from uploaded image.
				2. Hazard policy layer maps component to risk band and material profile.
				3. Compliance layer checks confidence threshold and review requirement.
				4. Disposal planner generates final pathway aligned to SDG 12.4 / 12.5.
				"""
			)

			with st.expander("raw decision payload"):
				st.json(decision)

	with tab_registry:
		st.subheader("Model Registry and Checkpoint Inventory")

		registry_rows = []
		for arch, path in assets["checkpoints"].items():
			registry_rows.append(
				{
					"architecture": arch,
					"checkpoint_path": str(path).replace("\\", "/"),
					"status": "active" if arch == best_arch else "available",
				}
			)

		st.dataframe(pd.DataFrame(registry_rows), use_container_width=True, hide_index=True)

		st.markdown("#### Classification Benchmark Snapshot")
		if metrics_df.empty:
			st.info("No benchmark metrics JSON found in models/classification.")
		else:
			st.dataframe(metrics_df, use_container_width=True, hide_index=True)
			chart_df = metrics_df.set_index("architecture")[["accuracy", "macro_f1"]]
			st.bar_chart(chart_df)

		competition_path = PROJECT_ROOT / "models" / "competition" / "leaderboard.json"
		leaderboard = load_json(competition_path)
		st.markdown("#### Competition Leaderboard")
		if not leaderboard:
			st.caption("Competition leaderboard not generated yet. Run: python run_system.py compete")
		else:
			winner = leaderboard.get("winner", "n/a")
			st.success(f"Winner: {winner}")
			ranking = leaderboard.get("ranking", [])
			if isinstance(ranking, list) and ranking:
				st.dataframe(pd.DataFrame(ranking), use_container_width=True, hide_index=True)


if __name__ == "__main__":
	main()

