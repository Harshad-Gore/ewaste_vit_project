from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import html
from io import BytesIO
from pathlib import Path
import json
import os
import random
import sys
import tempfile
import time
from textwrap import dedent, wrap
from urllib import request as urllib_request

try:
	import cv2
except Exception:  # pragma: no cover - optional runtime dependency
	cv2 = None

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw
import seaborn as sns
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
from training.image_preprocessing import build_eval_transform  # noqa: E402


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
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".webm"}

EVAL_TRANSFORM = build_eval_transform(224)

COCO_INSTANCE_CATEGORIES = [
	"__background__",
	"person",
	"bicycle",
	"car",
	"motorcycle",
	"airplane",
	"bus",
	"train",
	"truck",
	"boat",
	"traffic light",
	"fire hydrant",
	"N/A",
	"stop sign",
	"parking meter",
	"bench",
	"bird",
	"cat",
	"dog",
	"horse",
	"sheep",
	"cow",
	"elephant",
	"bear",
	"zebra",
	"giraffe",
	"N/A",
	"backpack",
	"umbrella",
	"N/A",
	"N/A",
	"handbag",
	"tie",
	"suitcase",
	"frisbee",
	"skis",
	"snowboard",
	"sports ball",
	"kite",
	"baseball bat",
	"baseball glove",
	"skateboard",
	"surfboard",
	"tennis racket",
	"bottle",
	"N/A",
	"wine glass",
	"cup",
	"fork",
	"knife",
	"spoon",
	"bowl",
	"banana",
	"apple",
	"sandwich",
	"orange",
	"broccoli",
	"carrot",
	"hot dog",
	"pizza",
	"donut",
	"cake",
	"chair",
	"couch",
	"potted plant",
	"bed",
	"N/A",
	"dining table",
	"N/A",
	"N/A",
	"toilet",
	"N/A",
	"tv",
	"laptop",
	"mouse",
	"remote",
	"keyboard",
	"cell phone",
	"microwave",
	"oven",
	"toaster",
	"sink",
	"refrigerator",
	"N/A",
	"book",
	"clock",
	"vase",
	"scissors",
	"teddy bear",
	"hair drier",
	"toothbrush",
]

DETECTOR_PRIORITY_LABELS = {
	"tv",
	"laptop",
	"mouse",
	"keyboard",
	"cell phone",
	"microwave",
	"refrigerator",
	"remote",
	"clock",
}

DETECTOR_EXCLUDED_LABELS = {
	"person",
}

HAZARD_RANK = {
	"UNKNOWN": 0,
	"LOW": 1,
	"MEDIUM": 2,
	"HIGH": 3,
}

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
	"Workflow",
	"Policy",
	"Benchmarks",
	"Analytics",
	"Registry",
	"Copilot",
]

ICON_SVGS = {
	"spark": """
	<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
		<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3z" />
	</svg>
	""",
	"cpu": """
	<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
		<rect x="7" y="7" width="10" height="10" rx="2" />
		<path d="M9 1v3M15 1v3M9 20v3M15 20v3M1 9h3M1 15h3M20 9h3M20 15h3M10 10h4v4h-4z" />
	</svg>
	""",
	"chart": """
	<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
		<path d="M4 19h16M7 15l3-3 3 2 5-6" />
		<path d="M7 19v-4M10 19v-7M13 19v-5M18 19v-10" />
	</svg>
	""",
	"shield": """
	<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
		<path d="M12 3l7 3v5c0 4.5-2.7 7.9-7 10-4.3-2.1-7-5.5-7-10V6l7-3z" />
		<path d="M9.5 12.5l1.8 1.8 3.2-3.7" />
	</svg>
	""",
	"gauge": """
	<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
		<path d="M5 16a7 7 0 1 1 14 0" />
		<path d="M12 12l4-3" />
		<circle cx="12" cy="12" r="1.4" />
	</svg>
	""",
	"bolt": """
	<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
		<path d="M13 2L5 14h5l-1 8 8-12h-5l1-8z" />
	</svg>
	""",
	"gallery": """
	<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
		<rect x="3" y="5" width="18" height="14" rx="2" />
		<path d="M8 11l2.2 2.2 3.1-3.1 4.7 4.9" />
		<circle cx="9" cy="9" r="1.2" />
	</svg>
	""",
	"review": """
	<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
		<circle cx="11" cy="11" r="6" />
		<path d="M16 16l4.5 4.5" />
	</svg>
	""",
	"route": """
	<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
		<circle cx="6" cy="18" r="2" />
		<circle cx="18" cy="6" r="2" />
		<path d="M8 18h3a5 5 0 0 0 5-5V8" />
		<path d="M13 8h3V5" />
	</svg>
	""",
	"database": """
	<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
		<ellipse cx="12" cy="6" rx="7" ry="3" />
		<path d="M5 6v6c0 1.7 3.1 3 7 3s7-1.3 7-3V6" />
		<path d="M5 12v6c0 1.7 3.1 3 7 3s7-1.3 7-3v-6" />
	</svg>
	""",
	"cluster": """
	<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
		<circle cx="6" cy="8" r="2" />
		<circle cx="18" cy="7" r="2" />
		<circle cx="12" cy="17" r="2" />
		<path d="M7.7 9.1l2.8 5.1M16.3 8.4l-2.8 5.1M8 8h8" />
	</svg>
	""",
	"chat": """
	<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
		<path d="M5 6.5A2.5 2.5 0 0 1 7.5 4h9A2.5 2.5 0 0 1 19 6.5v6A2.5 2.5 0 0 1 16.5 15H11l-4 4v-4H7.5A2.5 2.5 0 0 1 5 12.5v-6z" />
		<path d="M8 8h8M8 11h5" />
	</svg>
	""",
	"warning": """
	<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
		<path d="M12 4l9 16H3L12 4z" />
		<path d="M12 9v4M12 17h.01" />
	</svg>
	""",
	"stack": """
	<svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
		<path d="M12 4l8 4-8 4-8-4 8-4z" />
		<path d="M4 12l8 4 8-4" />
		<path d="M4 16l8 4 8-4" />
	</svg>
	""",
}


def escape_html(value: object) -> str:
	return html.escape(str(value), quote=True)


def format_html_copy(value: object) -> str:
	return escape_html(value).replace("\n", "<br/>")


def render_html(markup: str) -> None:
	st.markdown(dedent(markup).strip(), unsafe_allow_html=True)


def guess_icon_name(label: str) -> str:
	text = label.lower()
	if any(token in text for token in {"arch", "model", "checkpoint", "runtime", "deployment"}):
		return "cpu"
	if any(token in text for token in {"benchmark", "f1", "metric", "score"}):
		return "chart"
	if any(token in text for token in {"hazard", "compliance", "risk"}):
		return "shield"
	if any(token in text for token in {"confidence", "threshold"}):
		return "gauge"
	if any(token in text for token in {"latency", "runtime", "speed"}):
		return "bolt"
	if any(token in text for token in {"route", "pathway", "policy", "decision"}):
		return "route"
	if any(token in text for token in {"registry", "dataset", "taxonomy", "data"}):
		return "database"
	if any(token in text for token in {"cluster", "analytics"}):
		return "cluster"
	if any(token in text for token in {"chat", "copilot", "assistant", "bot"}):
		return "chat"
	if any(token in text for token in {"image", "scene", "gallery", "intake"}):
		return "gallery"
	if any(token in text for token in {"review", "triage", "trace"}):
		return "review"
	if any(token in text for token in {"warning", "caution"}):
		return "warning"
	return "spark"


def inject_styles() -> None:
	st.markdown(
		"""
		<style>
		@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Sora:wght@500;600;700;800&display=swap');

		:root {
			--bg-0: #071018;
			--bg-1: #0a1724;
			--bg-2: #112132;
			--surface: rgba(11, 21, 33, 0.88);
			--surface-strong: rgba(13, 24, 37, 0.97);
			--surface-soft: rgba(20, 35, 52, 0.72);
			--line: rgba(146, 165, 185, 0.17);
			--line-strong: rgba(146, 165, 185, 0.28);
			--text: #f2f7ff;
			--muted: #92a4b8;
			--accent: #63d0d9;
			--accent-strong: #9ce7ed;
			--signal: #f38b59;
			--success: #3cc58d;
			--warning: #f4b168;
			--danger: #ff6f61;
			--shadow: 0 28px 80px rgba(3, 10, 18, 0.36);
			--radius-xl: 28px;
			--radius-lg: 22px;
			--radius-md: 18px;
		}

		html, body, [data-testid="stAppViewContainer"] {
			font-family: 'Manrope', sans-serif;
			color: var(--text);
			background:
				radial-gradient(circle at 0% 0%, rgba(99, 208, 217, 0.18), transparent 28%),
				radial-gradient(circle at 100% 0%, rgba(243, 139, 89, 0.14), transparent 26%),
				repeating-linear-gradient(
					90deg,
					rgba(255, 255, 255, 0.025) 0,
					rgba(255, 255, 255, 0.025) 1px,
					transparent 1px,
					transparent 84px
				),
				repeating-linear-gradient(
					0deg,
					rgba(255, 255, 255, 0.02) 0,
					rgba(255, 255, 255, 0.02) 1px,
					transparent 1px,
					transparent 84px
				),
				linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 46%, #08131f 100%);
		}

		[data-testid="stAppViewContainer"]::before {
			content: "";
			position: fixed;
			inset: 0;
			pointer-events: none;
			background:
				radial-gradient(540px 240px at 18% 14%, rgba(99, 208, 217, 0.11), transparent 62%),
				radial-gradient(760px 320px at 82% 9%, rgba(243, 139, 89, 0.07), transparent 58%);
			z-index: 0;
		}

		[data-testid="stHeader"] {
			background: transparent;
		}

		[data-testid="stAppViewContainer"] > .main,
		[data-testid="stSidebar"] {
			position: relative;
			z-index: 1;
		}

		[data-testid="block-container"] {
			max-width: 1320px;
			padding-top: 1.5rem;
			padding-bottom: 2rem;
		}

		h1, h2, h3, h4 {
			font-family: 'Sora', sans-serif;
			letter-spacing: -0.03em;
			color: #f8fbff;
		}

		p, li, [data-testid="stMarkdownContainer"] {
			color: var(--text);
		}

		[data-testid="stSidebar"] {
			border-right: 1px solid var(--line);
			background:
				linear-gradient(180deg, rgba(7, 16, 24, 0.98) 0%, rgba(10, 23, 36, 0.98) 100%);
		}

		[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
			padding-top: 0.5rem;
		}

		.stButton > button,
		button[kind="secondary"],
		button[kind="primary"] {
			border-radius: 16px;
			min-height: 46px;
			padding: 0.75rem 1rem;
			border: 1px solid var(--line-strong);
			background: linear-gradient(180deg, rgba(17, 31, 46, 0.92) 0%, rgba(10, 19, 30, 0.96) 100%);
			color: #eef7ff;
			font-weight: 700;
			transition: transform 140ms ease, box-shadow 140ms ease, border-color 140ms ease;
			box-shadow: 0 10px 24px rgba(3, 10, 18, 0.24);
		}

		button[kind="primary"] {
			background:
				linear-gradient(135deg, rgba(99, 208, 217, 0.96) 0%, rgba(109, 227, 198, 0.94) 100%);
			color: #071018;
			border-color: rgba(161, 238, 226, 0.45);
			box-shadow: 0 18px 36px rgba(99, 208, 217, 0.18);
		}

		.stButton > button:hover,
		button[kind="secondary"]:hover,
		button[kind="primary"]:hover {
			transform: translateY(-1px);
			border-color: rgba(156, 231, 237, 0.48);
			box-shadow: 0 18px 36px rgba(3, 10, 18, 0.3);
		}

		[data-testid="stFileUploaderDropzone"] {
			border-radius: var(--radius-lg);
			border: 1px dashed rgba(156, 231, 237, 0.32);
			background:
				linear-gradient(180deg, rgba(13, 24, 37, 0.95) 0%, rgba(9, 17, 27, 0.98) 100%);
			padding: 1rem 1.05rem;
			box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
		}

		[data-testid="stFileUploaderDropzone"]:hover {
			border-color: rgba(156, 231, 237, 0.5);
			background:
				linear-gradient(180deg, rgba(16, 28, 42, 0.98) 0%, rgba(10, 19, 30, 1) 100%);
		}

		div[data-baseweb="select"] > div,
		[data-baseweb="base-input"] > div,
		[data-testid="stTextInput"] input {
			background: rgba(14, 25, 38, 0.9);
			border-radius: 16px;
			border-color: var(--line-strong);
			color: #eef7ff;
		}

		div[data-baseweb="select"] svg {
			color: var(--accent-strong);
		}

		[data-baseweb="slider"] [role="slider"] {
			background: var(--accent);
			box-shadow: 0 0 0 6px rgba(99, 208, 217, 0.15);
		}

		[data-baseweb="slider"] > div > div > div {
			background: linear-gradient(90deg, rgba(99, 208, 217, 0.95), rgba(243, 139, 89, 0.82));
		}

		[data-testid="stProgressBar"] > div,
		[data-testid="stProgress"] > div {
			background: rgba(146, 165, 185, 0.12);
			border-radius: 999px;
			overflow: hidden;
		}

		[data-testid="stProgressBar"] > div > div,
		[data-testid="stProgress"] > div > div {
			background: linear-gradient(90deg, rgba(99, 208, 217, 1) 0%, rgba(103, 225, 184, 0.96) 50%, rgba(243, 139, 89, 0.92) 100%);
			border-radius: 999px;
		}

		[data-testid="stImage"] img,
		[data-testid="stDataFrame"],
		[data-testid="stTable"] {
			border-radius: var(--radius-lg);
			overflow: hidden;
			border: 1px solid var(--line);
			box-shadow: var(--shadow);
		}

		div[data-baseweb="tab-list"] {
			flex-wrap: wrap;
			gap: 0.5rem;
			margin-bottom: 0.8rem;
		}

		button[data-baseweb="tab"] {
			border-radius: 999px;
			padding: 0.55rem 1rem;
			background: rgba(15, 28, 41, 0.76);
			border: 1px solid var(--line);
			color: #c9d6e5;
			height: auto;
			min-height: 2.6rem;
			font-weight: 700;
		}

		button[data-baseweb="tab"][aria-selected="true"] {
			background: linear-gradient(135deg, rgba(99, 208, 217, 0.18) 0%, rgba(243, 139, 89, 0.14) 100%);
			color: #f8fbff;
			border-color: rgba(156, 231, 237, 0.36);
			box-shadow: 0 10px 26px rgba(3, 10, 18, 0.18);
		}

		.hero-shell,
		.metric-shell,
		.panel-shell,
		.banner-shell,
		.section-shell,
		.timeline-shell,
		.checklist-shell {
			animation: rise-in 420ms ease both;
		}

		.hero-shell {
			position: relative;
			overflow: hidden;
			border-radius: var(--radius-xl);
			border: 1px solid rgba(156, 231, 237, 0.18);
			padding: 1.3rem 1.3rem 1.15rem;
			background:
				radial-gradient(circle at 78% 22%, rgba(243, 139, 89, 0.14), transparent 26%),
				radial-gradient(circle at 20% 12%, rgba(99, 208, 217, 0.18), transparent 34%),
				linear-gradient(145deg, rgba(10, 21, 33, 0.98) 0%, rgba(12, 24, 37, 0.95) 42%, rgba(8, 18, 28, 0.98) 100%);
			box-shadow: 0 30px 90px rgba(3, 10, 18, 0.42);
			margin-bottom: 1rem;
		}

		.hero-shell::after {
			content: "";
			position: absolute;
			inset: auto -15% -52% 45%;
			height: 260px;
			background: radial-gradient(circle, rgba(99, 208, 217, 0.18), transparent 68%);
			pointer-events: none;
		}

		.hero-grid {
			display: grid;
			grid-template-columns: minmax(0, 1.7fr) minmax(280px, 0.95fr);
			gap: 1rem;
			align-items: start;
		}

		.hero-ribbon {
			display: inline-flex;
			align-items: center;
			gap: 0.5rem;
			border-radius: 999px;
			padding: 0.42rem 0.72rem;
			border: 1px solid rgba(156, 231, 237, 0.18);
			background: rgba(10, 25, 37, 0.72);
			font-size: 0.76rem;
			font-weight: 800;
			letter-spacing: 0.14em;
			text-transform: uppercase;
			color: var(--accent-strong);
			margin-bottom: 1rem;
		}

		.hero-title {
			font-family: 'Sora', sans-serif;
			font-size: clamp(1.75rem, 3.2vw, 2.55rem);
			line-height: 1.02;
			letter-spacing: -0.05em;
			margin: 0;
			max-width: 9.5em;
		}

		.hero-copy {
			margin: 0.9rem 0 0;
			color: #d7e3ef;
			font-size: 0.95rem;
			line-height: 1.68;
			max-width: 60rem;
		}

		.hero-chip-row {
			display: flex;
			flex-wrap: wrap;
			gap: 0.7rem;
			margin-top: 1rem;
		}

		.hero-chip {
			display: inline-flex;
			align-items: center;
			gap: 0.55rem;
			padding: 0.54rem 0.88rem;
			border-radius: 999px;
			border: 1px solid var(--line);
			background: rgba(16, 29, 43, 0.72);
			color: #e5eef8;
			font-size: 0.84rem;
			font-weight: 700;
		}

		.hero-side {
			display: grid;
			gap: 0.7rem;
		}

		.hero-side-note {
			padding: 1rem 1rem 0.9rem;
			border-radius: 20px;
			border: 1px solid var(--line);
			background: linear-gradient(180deg, rgba(17, 31, 46, 0.8), rgba(10, 19, 30, 0.98));
		}

		.hero-side-label {
			color: var(--muted);
			text-transform: uppercase;
			letter-spacing: 0.14em;
			font-size: 0.74rem;
			font-weight: 800;
			margin-bottom: 0.5rem;
		}

		.hero-side-value {
			font-size: 1.28rem;
			font-weight: 800;
			color: #f9fcff;
			margin-bottom: 0.35rem;
		}

		.hero-side-copy {
			font-size: 0.92rem;
			line-height: 1.55;
			color: #d2deea;
		}

		.section-shell {
			margin: 0.45rem 0 1rem;
			padding: 1rem 1.1rem;
			border-radius: 24px;
			border: 1px solid var(--line);
			background: linear-gradient(180deg, rgba(13, 24, 37, 0.88) 0%, rgba(9, 17, 27, 0.94) 100%);
			box-shadow: 0 20px 48px rgba(3, 10, 18, 0.24);
		}

		.section-head {
			display: flex;
			align-items: center;
			gap: 0.9rem;
			margin-bottom: 0.65rem;
		}

		.section-icon,
		.metric-icon,
		.panel-icon,
		.banner-icon,
		.hero-chip svg {
			display: inline-flex;
			align-items: center;
			justify-content: center;
		}

		.section-icon,
		.metric-icon,
		.panel-icon,
		.banner-icon {
			width: 42px;
			height: 42px;
			border-radius: 14px;
			border: 1px solid rgba(156, 231, 237, 0.16);
			background: rgba(15, 29, 43, 0.92);
			color: var(--accent-strong);
			flex-shrink: 0;
		}

		.section-icon svg,
		.metric-icon svg,
		.panel-icon svg,
		.banner-icon svg,
		.hero-chip svg {
			width: 20px;
			height: 20px;
			stroke: currentColor;
			stroke-width: 1.8;
			stroke-linecap: round;
			stroke-linejoin: round;
		}

		.section-kicker {
			color: var(--accent-strong);
			text-transform: uppercase;
			letter-spacing: 0.18em;
			font-size: 0.72rem;
			font-weight: 800;
			margin-bottom: 0.15rem;
		}

		.section-title {
			font-family: 'Sora', sans-serif;
			font-size: 1.55rem;
			line-height: 1.08;
			margin: 0;
		}

		.section-copy {
			font-size: 0.96rem;
			line-height: 1.7;
			color: #d5e2ee;
			margin: 0;
		}

		.metric-shell,
		.panel-shell {
			position: relative;
			overflow: hidden;
			border-radius: var(--radius-lg);
			border: 1px solid var(--line);
			background:
				linear-gradient(180deg, rgba(15, 28, 41, 0.88) 0%, rgba(10, 19, 30, 0.98) 100%);
			box-shadow: 0 20px 54px rgba(3, 10, 18, 0.26);
		}

		.metric-shell {
			padding: 1rem 1rem 0.95rem;
			min-height: 138px;
		}

		.panel-shell {
			padding: 1rem 1rem 0.92rem;
			min-height: 118px;
		}

		.metric-shell::before,
		.panel-shell::before {
			content: "";
			position: absolute;
			inset: 0 auto auto 0;
			height: 3px;
			width: 100%;
			background: linear-gradient(90deg, rgba(99, 208, 217, 0.7), rgba(243, 139, 89, 0.56));
			opacity: 0.6;
		}

		.metric-shell.tone-success::before,
		.panel-shell.tone-success::before {
			background: linear-gradient(90deg, rgba(60, 197, 141, 0.88), rgba(156, 231, 237, 0.66));
		}

		.metric-shell.tone-warning::before,
		.panel-shell.tone-warning::before {
			background: linear-gradient(90deg, rgba(244, 177, 104, 0.95), rgba(243, 139, 89, 0.72));
		}

		.metric-shell.tone-danger::before,
		.panel-shell.tone-danger::before {
			background: linear-gradient(90deg, rgba(255, 111, 97, 0.96), rgba(244, 177, 104, 0.64));
		}

		.metric-top,
		.panel-top {
			display: flex;
			align-items: flex-start;
			gap: 0.85rem;
		}

		.metric-label,
		.panel-label {
			font-size: 0.74rem;
			text-transform: uppercase;
			letter-spacing: 0.16em;
			color: var(--muted);
			font-weight: 800;
		}

		.metric-value {
			font-family: 'Sora', sans-serif;
			font-size: 1.52rem;
			line-height: 1.08;
			letter-spacing: -0.04em;
			margin-top: 0.28rem;
			color: #f7fbff;
		}

		.panel-value {
			font-family: 'Sora', sans-serif;
			font-size: 1.08rem;
			line-height: 1.18;
			letter-spacing: -0.03em;
			margin-top: 0.26rem;
			color: #f7fbff;
		}

		.metric-detail,
		.panel-copy {
			margin-top: 0.9rem;
			font-size: 0.92rem;
			line-height: 1.58;
			color: #d0ddea;
		}

		.banner-shell {
			display: grid;
			grid-template-columns: auto minmax(0, 1fr);
			gap: 0.9rem;
			align-items: start;
			padding: 0.95rem 1rem;
			border-radius: 20px;
			border: 1px solid var(--line);
			background: linear-gradient(180deg, rgba(17, 30, 44, 0.92) 0%, rgba(10, 18, 28, 0.97) 100%);
			box-shadow: 0 18px 42px rgba(3, 10, 18, 0.24);
			margin: 0.35rem 0 1rem;
		}

		.banner-shell.tone-warning {
			border-color: rgba(244, 177, 104, 0.28);
			background: linear-gradient(180deg, rgba(69, 44, 22, 0.38) 0%, rgba(10, 18, 28, 0.98) 100%);
		}

		.banner-shell.tone-success {
			border-color: rgba(60, 197, 141, 0.22);
			background: linear-gradient(180deg, rgba(17, 60, 45, 0.34) 0%, rgba(10, 18, 28, 0.98) 100%);
		}

		.banner-shell.tone-danger {
			border-color: rgba(255, 111, 97, 0.3);
			background: linear-gradient(180deg, rgba(76, 28, 26, 0.42) 0%, rgba(10, 18, 28, 0.98) 100%);
		}

		.banner-title {
			font-size: 1rem;
			font-weight: 800;
			color: #f8fbff;
			margin-bottom: 0.2rem;
		}

		.banner-copy {
			font-size: 0.93rem;
			line-height: 1.65;
			color: #d3dfea;
		}

		.badge-row {
			display: flex;
			flex-wrap: wrap;
			gap: 0.65rem;
			margin: 0.8rem 0 0.2rem;
		}

		.badge-pill {
			display: inline-flex;
			align-items: center;
			padding: 0.48rem 0.82rem;
			border-radius: 999px;
			border: 1px solid var(--line);
			background: rgba(15, 28, 41, 0.82);
			color: #dce7f2;
			font-size: 0.8rem;
			font-weight: 700;
			letter-spacing: 0.01em;
		}

		.timeline-shell,
		.checklist-shell {
			border-radius: var(--radius-lg);
			border: 1px solid var(--line);
			background: linear-gradient(180deg, rgba(14, 25, 38, 0.88) 0%, rgba(10, 18, 28, 0.96) 100%);
			padding: 0.95rem 1rem;
			box-shadow: 0 18px 44px rgba(3, 10, 18, 0.22);
		}

		.timeline-step,
		.checklist-item {
			display: grid;
			grid-template-columns: auto minmax(0, 1fr);
			gap: 0.8rem;
			align-items: start;
		}

		.timeline-step + .timeline-step,
		.checklist-item + .checklist-item {
			margin-top: 0.85rem;
			padding-top: 0.85rem;
			border-top: 1px solid rgba(146, 165, 185, 0.12);
		}

		.timeline-dot,
		.checklist-dot {
			width: 14px;
			height: 14px;
			border-radius: 999px;
			margin-top: 0.28rem;
			background: linear-gradient(135deg, rgba(99, 208, 217, 0.96), rgba(243, 139, 89, 0.78));
			box-shadow: 0 0 0 6px rgba(99, 208, 217, 0.08);
		}

		.timeline-step-name {
			font-size: 0.8rem;
			text-transform: uppercase;
			letter-spacing: 0.16em;
			font-weight: 800;
			color: var(--muted);
		}

		.timeline-step-title,
		.checklist-text {
			font-size: 0.94rem;
			line-height: 1.62;
			color: #dce6f0;
			margin-top: 0.15rem;
		}

		.timeline-status {
			display: inline-flex;
			align-items: center;
			margin-top: 0.42rem;
			padding: 0.22rem 0.56rem;
			border-radius: 999px;
			background: rgba(15, 28, 41, 0.72);
			border: 1px solid rgba(146, 165, 185, 0.15);
			font-size: 0.72rem;
			font-weight: 800;
			text-transform: uppercase;
			letter-spacing: 0.12em;
			color: var(--accent-strong);
		}

		.workflow-map-shell,
		.mapping-shell,
		.export-shell {
			border-radius: var(--radius-lg);
			border: 1px solid var(--line);
			background: linear-gradient(180deg, rgba(13, 24, 37, 0.9) 0%, rgba(9, 17, 27, 0.98) 100%);
			box-shadow: 0 22px 54px rgba(3, 10, 18, 0.24);
			padding: 1rem 1rem 1.05rem;
		}

		.workflow-map-grid {
			display: grid;
			grid-template-columns: repeat(6, minmax(160px, 1fr));
			gap: 0.85rem;
			align-items: stretch;
		}

		.workflow-stage {
			position: relative;
			overflow: visible;
			min-height: 200px;
			padding: 0.95rem 0.95rem 0.9rem;
			border-radius: 22px;
			border: 1px solid rgba(146, 165, 185, 0.18);
			background:
				radial-gradient(circle at top right, rgba(99, 208, 217, 0.16), transparent 42%),
				linear-gradient(180deg, rgba(15, 28, 41, 0.92) 0%, rgba(10, 18, 28, 0.99) 100%);
		}

		.workflow-stage::after {
			content: "";
			position: absolute;
			right: -0.72rem;
			top: 50%;
			width: 1.05rem;
			height: 1.05rem;
			transform: translateY(-50%) rotate(45deg);
			border-top: 2px solid rgba(99, 208, 217, 0.42);
			border-right: 2px solid rgba(99, 208, 217, 0.42);
		}

		.workflow-stage:last-child::after {
			display: none;
		}

		.workflow-stage-head {
			display: flex;
			align-items: center;
			gap: 0.75rem;
			margin-bottom: 0.8rem;
		}

		.workflow-stage-icon {
			width: 38px;
			height: 38px;
			border-radius: 12px;
			border: 1px solid rgba(156, 231, 237, 0.18);
			background: rgba(15, 29, 43, 0.94);
			color: var(--accent-strong);
			display: flex;
			align-items: center;
			justify-content: center;
			flex-shrink: 0;
		}

		.workflow-stage-icon svg {
			width: 18px;
			height: 18px;
			stroke: currentColor;
			stroke-width: 1.8;
			stroke-linecap: round;
			stroke-linejoin: round;
		}

		.workflow-stage-label {
			font-size: 0.68rem;
			font-weight: 800;
			letter-spacing: 0.18em;
			text-transform: uppercase;
			color: var(--muted);
			margin-bottom: 0.12rem;
		}

		.workflow-stage-title {
			font-family: 'Sora', sans-serif;
			font-size: 1rem;
			line-height: 1.18;
			color: #f6fbff;
		}

		.workflow-stage-copy {
			font-size: 0.88rem;
			line-height: 1.6;
			color: #d4e0eb;
			margin: 0;
		}

		.workflow-stage-footer {
			margin-top: 0.85rem;
			display: flex;
			flex-wrap: wrap;
			gap: 0.45rem;
		}

		.workflow-pill {
			display: inline-flex;
			align-items: center;
			padding: 0.28rem 0.56rem;
			border-radius: 999px;
			border: 1px solid rgba(146, 165, 185, 0.14);
			background: rgba(15, 28, 41, 0.78);
			font-size: 0.72rem;
			font-weight: 800;
			letter-spacing: 0.02em;
			color: #dce7f2;
		}

		.mapping-grid {
			display: grid;
			grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
			gap: 0.8rem;
		}

		.mapping-card {
			padding: 0.9rem;
			border-radius: 18px;
			border: 1px solid rgba(146, 165, 185, 0.16);
			background: rgba(15, 28, 41, 0.82);
		}

		.mapping-card-label {
			font-size: 0.7rem;
			font-weight: 800;
			letter-spacing: 0.16em;
			text-transform: uppercase;
			color: var(--muted);
			margin-bottom: 0.2rem;
		}

		.mapping-card-value {
			font-family: 'Sora', sans-serif;
			font-size: 0.98rem;
			line-height: 1.35;
			color: #f6fbff;
		}

		.mapping-card-copy {
			margin-top: 0.45rem;
			font-size: 0.86rem;
			line-height: 1.58;
			color: #d4e0eb;
		}

		.export-shell {
			margin-top: 0.35rem;
		}

		@keyframes rise-in {
			from {
				opacity: 0;
				transform: translateY(6px);
			}
			to {
				opacity: 1;
				transform: translateY(0);
			}
		}

		@media (max-width: 1440px) {
			[data-testid="block-container"] {
				padding-left: 0.9rem;
				padding-right: 0.9rem;
			}

			.hero-grid {
				grid-template-columns: 1fr;
			}

			.hero-title {
				font-size: clamp(1.7rem, 3vw, 2.3rem);
			}

			.metric-value {
				font-size: 1.38rem;
			}

			.workflow-map-grid {
				grid-template-columns: repeat(3, minmax(160px, 1fr));
			}
		}

		@media (max-width: 900px) {
			.hero-shell,
			.section-shell,
			.metric-shell,
			.panel-shell,
			.banner-shell,
			.timeline-shell,
			.checklist-shell {
				border-radius: 20px;
			}

			.section-head {
				align-items: flex-start;
			}

			.metric-detail,
			.panel-copy,
			.banner-copy {
				font-size: 0.9rem;
			}

			.workflow-map-grid {
				grid-template-columns: 1fr;
			}

			.workflow-stage::after {
				display: none;
			}
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


def to_float(value: object) -> float | None:
	try:
		if value is None:
			return None
		return float(value)
	except (TypeError, ValueError):
		return None


def format_size(num_bytes: int) -> str:
	if num_bytes < 1024:
		return f"{num_bytes} B"
	if num_bytes < 1024**2:
		return f"{num_bytes / 1024:.1f} KB"
	if num_bytes < 1024**3:
		return f"{num_bytes / (1024**2):.2f} MB"
	return f"{num_bytes / (1024**3):.2f} GB"


def classify_artifact(path: Path) -> str:
	ext = path.suffix.lower()
	if ext == ".pth":
		return "checkpoint"
	if ext == ".json":
		return "metrics_json"
	if ext == ".npy":
		return "array_npy"
	if ext in {".png", ".jpg", ".jpeg", ".webp", ".svg"}:
		return "visual"
	if ext == ".txt":
		return "report_txt"
	if ext == ".csv":
		return "table_csv"
	return "other"


def summarize_model_artifact(path: Path, category: str) -> str:
	try:
		if category == "metrics_json":
			payload = load_json(path)
			if not payload:
				return "JSON payload unavailable"

			metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else None
			if metrics:
				acc = to_float(metrics.get("accuracy"))
				macro_f1 = to_float(metrics.get("macro_f1"))
				if acc is not None and macro_f1 is not None:
					return f"accuracy {acc:.2%} | macro-F1 {macro_f1:.4f}"

			acc = to_float(payload.get("test_accuracy"))
			macro_f1 = to_float(payload.get("macro_f1"))
			if acc is not None and macro_f1 is not None:
				return f"test accuracy {acc:.2%} | macro-F1 {macro_f1:.4f}"

			if path.name == "leaderboard.json":
				ranking = payload.get("ranking")
				ranked_count = len(ranking) if isinstance(ranking, list) else 0
				return f"winner {payload.get('winner', 'n/a')} | ranked models {ranked_count}"

			if path.name == "per_class_f1_scores.json":
				return f"architectures tracked {len(payload)}"

			if path.name == "clustering_metrics.json":
				silhouette = to_float(payload.get("silhouette_score"))
				n_clusters = payload.get("n_clusters", "n/a")
				if silhouette is not None:
					return f"clusters {n_clusters} | silhouette {silhouette:.4f}"
				return f"clusters {n_clusters}"

			return f"top-level keys {len(payload)}"

		if category == "array_npy":
			arr = np.load(path, mmap_mode="r", allow_pickle=False)
			return f"shape {tuple(arr.shape)} | dtype {arr.dtype}"

		if category == "visual":
			with Image.open(path) as image:
				return f"resolution {image.width}x{image.height}"

		if category == "report_txt":
			with path.open("r", encoding="utf-8", errors="ignore") as fp:
				line_count = sum(1 for _ in fp)
			return f"report lines {line_count}"

		if category == "table_csv":
			head = pd.read_csv(path, nrows=5)
			return f"columns {len(head.columns)} | sampled rows {len(head)}"

		if category == "checkpoint":
			return f"binary checkpoint ({path.stem})"

		return "artifact metadata collected"
	except Exception:
		return "metadata unavailable"


def load_npy_array(path: Path) -> np.ndarray | None:
	if not path.exists():
		return None
	try:
		return np.load(path, allow_pickle=False)
	except Exception:
		return None


def build_history_figure(history: Mapping | None, title: str) -> plt.Figure | None:
	if not isinstance(history, Mapping):
		return None

	train_loss = history.get("train_loss") if isinstance(history.get("train_loss"), list) else history.get("train")
	val_loss = history.get("val_loss") if isinstance(history.get("val_loss"), list) else history.get("val")
	val_acc = history.get("val_acc") if isinstance(history.get("val_acc"), list) else None
	val_macro_f1 = history.get("val_macro_f1") if isinstance(history.get("val_macro_f1"), list) else None

	if not isinstance(train_loss, list) or not isinstance(val_loss, list) or not train_loss or not val_loss:
		return None

	n = min(len(train_loss), len(val_loss))
	if n <= 0:
		return None

	epochs = np.arange(1, n + 1)
	fig, ax = plt.subplots(figsize=(8, 4.3))
	ax.plot(epochs, train_loss[:n], label="train_loss", color="#2563eb", linewidth=2.0)
	ax.plot(epochs, val_loss[:n], label="val_loss", color="#ea580c", linewidth=2.0)
	ax.set_xlabel("epoch")
	ax.set_ylabel("loss")
	ax.grid(alpha=0.25)

	ax2 = ax.twinx()
	has_score = False
	if isinstance(val_acc, list) and val_acc:
		n_acc = min(n, len(val_acc))
		ax2.plot(epochs[:n_acc], val_acc[:n_acc], label="val_acc", color="#0ea5e9", linestyle="--", linewidth=1.8)
		has_score = True
	if isinstance(val_macro_f1, list) and val_macro_f1:
		n_f1 = min(n, len(val_macro_f1))
		ax2.plot(epochs[:n_f1], val_macro_f1[:n_f1], label="val_macro_f1", color="#22c55e", linestyle=":", linewidth=2.0)
		has_score = True

	if has_score:
		ax2.set_ylabel("score")

	handles, labels = ax.get_legend_handles_labels()
	if has_score:
		h2, l2 = ax2.get_legend_handles_labels()
		handles.extend(h2)
		labels.extend(l2)
	ax.legend(handles, labels, loc="best", fontsize=8)
	ax.set_title(title, fontweight="bold")
	fig.tight_layout()
	return fig


def build_confusion_figure(
	payload: Mapping | None,
	fallback_class_names: list[str] | None,
	title: str,
	normalized: bool = False,
) -> plt.Figure | None:
	if not isinstance(payload, Mapping):
		return None

	key = "confusion_matrix_normalized" if normalized else "confusion_matrix"
	matrix = payload.get(key)

	if matrix is None and normalized and payload.get("confusion_matrix") is not None:
		raw = np.array(payload["confusion_matrix"], dtype=float)
		row_sums = raw.sum(axis=1, keepdims=True)
		matrix = np.divide(raw, row_sums, out=np.zeros_like(raw, dtype=float), where=row_sums != 0)

	if matrix is None:
		return None

	array = np.array(matrix, dtype=float)
	n_classes = array.shape[0]
	labels = payload.get("class_names") if isinstance(payload.get("class_names"), list) else fallback_class_names
	if not isinstance(labels, list) or len(labels) != n_classes:
		labels = [str(i) for i in range(n_classes)]

	fig, ax = plt.subplots(figsize=(10.2, 8.2))
	annot = n_classes <= 10
	fmt = ".2f" if normalized else "d"
	display = array if normalized else np.rint(array).astype(int)
	sns.heatmap(
		display,
		ax=ax,
		cmap="Blues",
		annot=annot,
		fmt=fmt,
		xticklabels=labels,
		yticklabels=labels,
		cbar=True,
	)
	acc = None
	if isinstance(payload.get("metrics"), Mapping):
		acc = to_float(payload["metrics"].get("accuracy"))
	if acc is None:
		acc = to_float(payload.get("test_accuracy"))
	suffix = f" | acc={acc:.4f}" if acc is not None else ""
	ax.set_title(f"{title}{suffix}", fontweight="bold")
	ax.set_xlabel("predicted label")
	ax.set_ylabel("true label")
	ax.tick_params(axis="x", rotation=55, labelsize=8)
	ax.tick_params(axis="y", rotation=0, labelsize=8)
	fig.tight_layout()
	return fig


def build_class_report_heatmap_figure(payload: Mapping | None, title: str) -> plt.Figure | None:
	if not isinstance(payload, Mapping):
		return None
	report = payload.get("classification_report")
	if not isinstance(report, Mapping):
		return None

	rows: list[dict] = []
	for class_name, metrics in report.items():
		if not isinstance(metrics, Mapping):
			continue
		if str(class_name).lower() in {"accuracy", "macro avg", "weighted avg"}:
			continue
		precision = to_float(metrics.get("precision"))
		recall = to_float(metrics.get("recall"))
		f1_score = to_float(metrics.get("f1-score"))
		if precision is None or recall is None or f1_score is None:
			continue
		rows.append(
			{
				"class": str(class_name),
				"precision": precision,
				"recall": recall,
				"f1": f1_score,
			}
		)

	if not rows:
		return None

	frame = pd.DataFrame(rows).set_index("class")
	fig, ax = plt.subplots(figsize=(7.8, max(4.5, len(frame) * 0.3)))
	sns.heatmap(
		frame,
		ax=ax,
		annot=True,
		fmt=".3f",
		vmin=0,
		vmax=1,
		cmap="YlOrRd",
	)
	ax.set_title(title, fontweight="bold")
	ax.set_xlabel("metric")
	ax.set_ylabel("class")
	fig.tight_layout()
	return fig


def build_per_class_f1_heatmap_figure(per_class_payload: Mapping | None) -> plt.Figure | None:
	if not isinstance(per_class_payload, Mapping) or not per_class_payload:
		return None
	frame = pd.DataFrame(per_class_payload)
	if frame.empty:
		return None
	ordered_cols = [arch for arch in SUPPORTED_ARCHES if arch in frame.columns]
	remaining = [col for col in frame.columns if col not in ordered_cols]
	frame = frame[ordered_cols + remaining]
	fig, ax = plt.subplots(figsize=(9.6, max(5.2, len(frame) * 0.35)))
	sns.heatmap(
		frame,
		ax=ax,
		annot=True,
		fmt=".2f",
		vmin=0,
		vmax=1,
		cmap="YlGnBu",
	)
	ax.set_title("Per-Class F1 Comparison", fontweight="bold")
	ax.set_xlabel("architecture")
	ax.set_ylabel("class")
	fig.tight_layout()
	return fig


def build_ann_overview_figure(ann_payload: Mapping | None) -> plt.Figure | None:
	if not isinstance(ann_payload, Mapping) or not ann_payload:
		return None

	fig, axes = plt.subplots(1, 3, figsize=(17.2, 4.8))

	history = ann_payload.get("history") if isinstance(ann_payload.get("history"), Mapping) else {}
	train_loss = history.get("train") if isinstance(history.get("train"), list) else None
	val_loss = history.get("val") if isinstance(history.get("val"), list) else None
	if isinstance(train_loss, list) and isinstance(val_loss, list) and train_loss and val_loss:
		n = min(len(train_loss), len(val_loss))
		epochs = np.arange(1, n + 1)
		axes[0].plot(epochs, train_loss[:n], color="#2563eb", linewidth=2.0, label="train_loss")
		axes[0].plot(epochs, val_loss[:n], color="#ea580c", linewidth=2.0, label="val_loss")
		axes[0].set_title("ANN Training Curves", fontweight="bold")
		axes[0].set_xlabel("epoch")
		axes[0].set_ylabel("loss")
		axes[0].grid(alpha=0.25)
		axes[0].legend(fontsize=8)
	else:
		axes[0].text(0.5, 0.5, "history unavailable", ha="center", va="center")
		axes[0].set_axis_off()

	reg = ann_payload.get("regression_metrics") if isinstance(ann_payload.get("regression_metrics"), Mapping) else {}
	reg_items = []
	for key in ("mae", "rmse", "mape", "r2"):
		metric = to_float(reg.get(key))
		if metric is not None:
			reg_items.append((key.upper(), metric))
	if reg_items:
		labels = [item[0] for item in reg_items]
		values = [item[1] for item in reg_items]
		bars = axes[1].bar(labels, values, color=["#0ea5e9", "#22c55e", "#a855f7", "#f43f5e"][: len(labels)])
		axes[1].set_title("ANN Regression Metrics", fontweight="bold")
		axes[1].grid(axis="y", alpha=0.25)
		for bar, value in zip(bars, values):
			axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
	else:
		axes[1].text(0.5, 0.5, "regression metrics unavailable", ha="center", va="center")
		axes[1].set_axis_off()

	report = ann_payload.get("classification_report") if isinstance(ann_payload.get("classification_report"), Mapping) else {}
	report_rows: list[tuple[str, list[float]]] = []
	for class_name, metrics in report.items():
		if not isinstance(metrics, Mapping):
			continue
		if str(class_name).lower() in {"accuracy", "macro avg", "weighted avg"}:
			continue
		row = [
			to_float(metrics.get("precision")),
			to_float(metrics.get("recall")),
			to_float(metrics.get("f1-score")),
		]
		if all(value is not None for value in row):
			report_rows.append((str(class_name), [float(value) for value in row]))

	if report_rows:
		report_frame = pd.DataFrame(
			[item[1] for item in report_rows],
			index=[item[0] for item in report_rows],
			columns=["precision", "recall", "f1"],
		)
		sns.heatmap(report_frame, ax=axes[2], annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1)
		axes[2].set_title("ANN Hazard Class Report", fontweight="bold")
		axes[2].set_xlabel("metric")
		axes[2].set_ylabel("class")
	else:
		axes[2].text(0.5, 0.5, "classification report unavailable", ha="center", va="center")
		axes[2].set_axis_off()

	fig.suptitle("ANN Research Insights", fontsize=14, fontweight="bold")
	fig.tight_layout()
	return fig


def build_ann_backprop_figure(backprop_payload: Mapping | None) -> plt.Figure | None:
	if not isinstance(backprop_payload, Mapping) or not backprop_payload:
		return None

	tracking = backprop_payload.get("gradient_tracking") if isinstance(backprop_payload.get("gradient_tracking"), Mapping) else {}
	epochs = tracking.get("epochs") if isinstance(tracking.get("epochs"), list) else []
	total_grad = tracking.get("total_grad_norm") if isinstance(tracking.get("total_grad_norm"), list) else []
	first_grad = tracking.get("first_layer_grad_norm") if isinstance(tracking.get("first_layer_grad_norm"), list) else []
	last_grad = tracking.get("last_layer_grad_norm") if isinstance(tracking.get("last_layer_grad_norm"), list) else []
	update_norm = tracking.get("update_norm") if isinstance(tracking.get("update_norm"), list) else []
	best_epoch = backprop_payload.get("best_epoch")

	if not epochs:
		return None

	fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.8))
	if total_grad and first_grad and last_grad:
		n = min(len(epochs), len(total_grad), len(first_grad), len(last_grad))
		x = epochs[:n]
		axes[0].plot(x, total_grad[:n], label="total", linewidth=2.0, color="#0ea5e9")
		axes[0].plot(x, first_grad[:n], label="first layer", linewidth=1.8, color="#22c55e")
		axes[0].plot(x, last_grad[:n], label="last layer", linewidth=1.8, color="#f97316")
		if isinstance(best_epoch, int):
			axes[0].axvline(best_epoch, linestyle="--", linewidth=1.2, color="#e11d48", alpha=0.7)
		axes[0].set_title("Backprop Gradient Norms", fontweight="bold")
		axes[0].set_xlabel("epoch")
		axes[0].set_ylabel("gradient norm")
		axes[0].grid(alpha=0.22)
		axes[0].legend(fontsize=8)
	else:
		axes[0].text(0.5, 0.5, "gradient traces unavailable", ha="center", va="center")
		axes[0].set_axis_off()

	summary = backprop_payload.get("backprop_summary") if isinstance(backprop_payload.get("backprop_summary"), Mapping) else {}
	summary_items = [
		("best val", to_float(summary.get("best_val_loss"))),
		("clip", to_float(summary.get("clip_grad_norm"))),
		("peak grad", to_float(summary.get("peak_total_grad_norm"))),
		("mean grad", to_float(summary.get("mean_total_grad_norm"))),
	]
	summary_items = [item for item in summary_items if item[1] is not None]
	if summary_items:
		labels = [item[0] for item in summary_items]
		values = [item[1] for item in summary_items]
		bars = axes[1].bar(labels, values, color=["#38bdf8", "#a855f7", "#f59e0b", "#34d399"][: len(labels)])
		axes[1].set_title("Backprop Summary Metrics", fontweight="bold")
		axes[1].grid(axis="y", alpha=0.22)
		for bar, value in zip(bars, values):
			axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
	else:
		axes[1].text(0.5, 0.5, "backprop summary unavailable", ha="center", va="center")
		axes[1].set_axis_off()

	fig.suptitle("ANN Backpropagation Diagnostics", fontsize=14, fontweight="bold")
	fig.tight_layout()
	return fig


def build_clustering_overview_figure(clustering_payload: Mapping | None) -> plt.Figure | None:
	if not isinstance(clustering_payload, Mapping) or not clustering_payload:
		return None

	fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.7))
	cluster_sizes = clustering_payload.get("cluster_sizes") if isinstance(clustering_payload.get("cluster_sizes"), Mapping) else {}
	parsed_sizes: list[tuple[str, int]] = []
	for key, value in cluster_sizes.items():
		v = to_float(value)
		if v is None:
			continue
		parsed_sizes.append((f"cluster_{key}", int(v)))

	if parsed_sizes:
		bars = axes[0].bar([item[0] for item in parsed_sizes], [item[1] for item in parsed_sizes], color="#0ea5e9")
		axes[0].set_title("Cluster Size Distribution", fontweight="bold")
		axes[0].set_ylabel("samples")
		axes[0].grid(axis="y", alpha=0.25)
		for bar, value in zip(bars, [item[1] for item in parsed_sizes]):
			axes[0].text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=8)
	else:
		axes[0].text(0.5, 0.5, "cluster size data unavailable", ha="center", va="center")
		axes[0].set_axis_off()

	metrics = [
		("silhouette", to_float(clustering_payload.get("silhouette_score"))),
		("ARI", to_float(clustering_payload.get("adjusted_rand_index"))),
		("NMI", to_float(clustering_payload.get("normalized_mutual_info"))),
		("DBI", to_float(clustering_payload.get("davies_bouldin_index"))),
	]
	metrics = [item for item in metrics if item[1] is not None]
	if metrics:
		bars = axes[1].bar([item[0] for item in metrics], [item[1] for item in metrics], color=["#22c55e", "#a855f7", "#f59e0b", "#ef4444"][: len(metrics)])
		axes[1].set_title("Clustering Quality Metrics", fontweight="bold")
		axes[1].grid(axis="y", alpha=0.25)
		for bar, value in zip(bars, [item[1] for item in metrics]):
			axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
	else:
		axes[1].text(0.5, 0.5, "quality metrics unavailable", ha="center", va="center")
		axes[1].set_axis_off()

	fig.suptitle("Clustering Research Insights", fontsize=14, fontweight="bold")
	fig.tight_layout()
	return fig


def build_clustering_comparison_figure(comparison_payload: Mapping | None) -> plt.Figure | None:
	if not isinstance(comparison_payload, Mapping) or not comparison_payload:
		return None

	algorithms = []
	for name in ("kmeans", "kmedoids"):
		payload = comparison_payload.get(name)
		if isinstance(payload, Mapping) and payload:
			algorithms.append((name, payload))
	if not algorithms:
		return None

	metric_specs = [
		("silhouette_score", "Silhouette"),
		("adjusted_rand_index", "ARI"),
		("normalized_mutual_info", "NMI"),
		("davies_bouldin_index", "DBI"),
	]
	fig, axes = plt.subplots(1, len(metric_specs), figsize=(4.1 * len(metric_specs), 4.8))
	axes = np.atleast_1d(axes).ravel()
	colors = {"kmeans": "#0ea5e9", "kmedoids": "#f97316"}

	for ax, (metric_key, metric_label) in zip(axes, metric_specs):
		values = []
		labels = []
		bar_colors = []
		for algo_name, payload in algorithms:
			value = to_float(payload.get(metric_key))
			if value is None:
				continue
			labels.append(algo_name.upper())
			values.append(value)
			bar_colors.append(colors.get(algo_name, "#94a3b8"))
		if values:
			bars = ax.bar(labels, values, color=bar_colors)
			for bar, value in zip(bars, values):
				ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
			ax.grid(axis="y", alpha=0.22)
			ax.set_title(metric_label, fontweight="bold")
		else:
			ax.text(0.5, 0.5, "n/a", ha="center", va="center")
			ax.set_axis_off()

	fig.suptitle("K-Means vs K-Medoids Comparison", fontsize=14, fontweight="bold")
	fig.tight_layout()
	return fig


def build_cluster_composition_figure(
	clustering_dir: Path,
	class_names: list[str] | None = None,
	cluster_labels_name: str = "cluster_labels.npy",
) -> plt.Figure | None:
	class_labels = load_npy_array(clustering_dir / "labels.npy")
	cluster_labels = load_npy_array(clustering_dir / cluster_labels_name)
	if class_labels is None or cluster_labels is None:
		return None
	if len(class_labels) != len(cluster_labels):
		return None

	cluster_ids = cluster_labels.astype(int)
	n_clusters = int(cluster_ids.max()) + 1 if len(cluster_ids) else 0
	if n_clusters <= 0:
		return None

	if np.issubdtype(class_labels.dtype, np.number):
		class_ids = class_labels.astype(int)
		unique_classes = np.unique(class_ids)
		class_map = {class_id: idx for idx, class_id in enumerate(unique_classes)}
		matrix = np.zeros((len(unique_classes), n_clusters), dtype=int)
		for class_id, cluster_id in zip(class_ids, cluster_ids):
			matrix[class_map[class_id], cluster_id] += 1
		if isinstance(class_names, list) and class_names and int(unique_classes.max()) < len(class_names):
			y_labels = [class_names[int(class_id)] for class_id in unique_classes]
		else:
			y_labels = [str(int(class_id)) for class_id in unique_classes]
	else:
		class_values = class_labels.astype(str)
		unique_classes = np.unique(class_values)
		class_map = {class_name: idx for idx, class_name in enumerate(unique_classes)}
		matrix = np.zeros((len(unique_classes), n_clusters), dtype=int)
		for class_name, cluster_id in zip(class_values, cluster_ids):
			matrix[class_map[class_name], cluster_id] += 1
		y_labels = [str(class_name) for class_name in unique_classes]

	x_labels = [f"cluster_{idx}" for idx in range(n_clusters)]
	fig, ax = plt.subplots(figsize=(8.8, max(4.8, len(y_labels) * 0.35)))
	sns.heatmap(matrix, ax=ax, annot=True, fmt="d", cmap="Blues", xticklabels=x_labels, yticklabels=y_labels)
	ax.set_title("Class vs Cluster Composition", fontweight="bold")
	ax.set_xlabel("cluster")
	ax.set_ylabel("class")
	fig.tight_layout()
	return fig


def build_tsne_cluster_figure(
	clustering_dir: Path,
	max_points: int = 8000,
	tsne_clusters_name: str = "tsne_clusters.npy",
) -> plt.Figure | None:
	tsne = load_npy_array(clustering_dir / "tsne_result.npy")
	clusters = load_npy_array(clustering_dir / tsne_clusters_name)
	if tsne is None or clusters is None:
		return None
	if tsne.ndim != 2 or tsne.shape[1] < 2 or len(tsne) != len(clusters):
		return None

	total_points = len(tsne)
	if total_points > max_points:
		rng = np.random.default_rng(42)
		indices = rng.choice(total_points, size=max_points, replace=False)
	else:
		indices = np.arange(total_points)

	points = tsne[indices, :2]
	cluster_ids = clusters[indices].astype(int)

	fig, ax = plt.subplots(figsize=(8.4, 6.6))
	scatter = ax.scatter(points[:, 0], points[:, 1], c=cluster_ids, cmap="tab10", s=9, alpha=0.72, linewidths=0)
	colorbar = fig.colorbar(scatter, ax=ax)
	colorbar.set_label("cluster id")
	ax.set_title("t-SNE Projection Colored by Cluster", fontweight="bold")
	ax.set_xlabel("t-SNE 1")
	ax.set_ylabel("t-SNE 2")
	ax.grid(alpha=0.16)
	fig.tight_layout()
	return fig


@st.cache_data(show_spinner=False)
def load_architecture_payloads(classification_dir_str: str) -> dict:
	classification_dir = Path(classification_dir_str)
	payloads: dict[str, dict] = {}
	for results_path in sorted(classification_dir.glob("*/results.json"), key=lambda p: p.parent.name.lower()):
		payload = load_json(results_path)
		if payload:
			payloads[results_path.parent.name] = payload
	return payloads


@st.cache_data(show_spinner=False)
def load_models_inventory(models_dir_str: str) -> dict:
	models_dir = Path(models_dir_str)
	rows: list[dict] = []
	for path in sorted(models_dir.rglob("*"), key=lambda p: str(p).lower()):
		if not path.is_file():
			continue
		stat = path.stat()
		rel_path = path.relative_to(models_dir).as_posix()
		section = rel_path.split("/")[0] if "/" in rel_path else rel_path
		category = classify_artifact(path)
		rows.append(
			{
				"section": section,
				"path": rel_path,
				"file_name": path.name,
				"extension": path.suffix.lower() or "(none)",
				"category": category,
				"size_bytes": int(stat.st_size),
				"size": format_size(int(stat.st_size)),
				"size_mb": float(stat.st_size / (1024**2)),
				"modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
				"insight": summarize_model_artifact(path, category),
			}
		)

	if not rows:
		empty = pd.DataFrame(columns=["section", "path", "file_name", "extension", "category", "size", "size_mb", "modified", "insight"])
		return {
			"files": empty.to_dict(orient="records"),
			"by_category": empty.to_dict(orient="records"),
			"by_section": empty.to_dict(orient="records"),
		}

	frame = pd.DataFrame(rows).sort_values(["section", "category", "path"]).reset_index(drop=True)
	by_category = (
		frame.groupby("category", dropna=False)
		.agg(file_count=("path", "count"), total_mb=("size_mb", "sum"))
		.reset_index()
		.sort_values(["file_count", "total_mb"], ascending=False)
	)
	by_section = (
		frame.groupby("section", dropna=False)
		.agg(file_count=("path", "count"), total_mb=("size_mb", "sum"))
		.reset_index()
		.sort_values(["file_count", "total_mb"], ascending=False)
	)

	return {
		"files": frame.to_dict(orient="records"),
		"by_category": by_category.to_dict(orient="records"),
		"by_section": by_section.to_dict(orient="records"),
	}


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


def resolve_benchmark_rows(
	metric_catalog: pd.DataFrame,
	arch: str,
) -> tuple[pd.Series | None, str | None, pd.Series | None, str | None]:
	if metric_catalog.empty:
		return None, None, None, None

	subset = metric_catalog.loc[metric_catalog["architecture"] == arch]
	if subset.empty:
		return None, None, None, None

	unique_sources = subset["source"].dropna().astype(str).unique().tolist()
	priority = ["dl_results.json", "test_results.json"]
	ordered_sources = [source for source in priority if source in unique_sources]
	ordered_sources.extend(sorted(source for source in unique_sources if source not in ordered_sources))

	primary_source = ordered_sources[0] if ordered_sources else None
	secondary_source = ordered_sources[1] if len(ordered_sources) > 1 else None

	primary_row = pick_metric_row(metric_catalog, arch, primary_source) if primary_source else None
	secondary_row = pick_metric_row(metric_catalog, arch, secondary_source) if secondary_source else None

	return primary_row, primary_source, secondary_row, secondary_source


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

	if "vit_b16" in checkpoints:
		return "vit_b16"

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
	ann_primary = load_json(project_root / "models" / "ann" / "ann_results_18cls.json") or {}
	ann_legacy = load_json(project_root / "models" / "ann" / "ann_results.json") or {}
	return {
		"ann_results": ann_primary or ann_legacy,
		"ann_results_18cls": ann_primary,
		"ann_backprop_results": load_json(project_root / "models" / "ann" / "ann_backprop_results.json") or {},
		"clustering_metrics": load_json(project_root / "models" / "clustering" / "clustering_metrics.json") or {},
		"clustering_results": load_json(project_root / "models" / "clustering" / "clustering_results.json") or {},
		"kmedoids_metrics": load_json(project_root / "models" / "clustering" / "kmedoids_metrics.json") or {},
		"kmedoids_results": load_json(project_root / "models" / "clustering" / "kmedoids_results.json") or {},
		"clustering_comparison": load_json(project_root / "models" / "clustering" / "clustering_comparison.json") or {},
		"competition_leaderboard": load_json(project_root / "models" / "competition" / "leaderboard.json") or {},
		"competition_all_players": load_json(project_root / "models" / "competition" / "all_players_results.json") or {},
		"competition_deep": load_json(project_root / "models" / "competition" / "deep_results.json") or {},
		"competition_traditional": load_json(project_root / "models" / "competition" / "traditional_ml_results.json") or {},
		"per_class_f1_scores": load_json(project_root / "models" / "classification" / "per_class_f1_scores.json") or {},
		"benchmark_summary": load_json(project_root / "models" / "classification" / "benchmark_summary.json") or {},
	}


def resolve_detector_label(categories: list[str], label_idx: int) -> str:
	has_background_slot = bool(categories) and str(categories[0]).strip() in {"__background__", "background", "N/A"}
	candidate_order = (label_idx, label_idx - 1) if has_background_slot else (label_idx - 1, label_idx)
	for candidate in candidate_order:
		if 0 <= candidate < len(categories):
			label = str(categories[candidate]).strip()
			if label and label not in {"__background__", "N/A"}:
				return label
	return f"class_{label_idx}"


@st.cache_resource(show_spinner=False)
def load_detection_model(model_key: str, device_type: str) -> dict:
	try:
		from torchvision.models import detection as detection_models
	except Exception as exc:  # pragma: no cover - defensive import path
		raise RuntimeError("torchvision detection models are unavailable in this environment.") from exc

	device = torch.device(device_type)
	if model_key != "fasterrcnn_mobilenet_v3_large_320_fpn":
		raise ValueError(f"unsupported detector: {model_key}")

	factory = getattr(detection_models, model_key, None)
	if factory is None:
		raise RuntimeError(
			"fasterrcnn_mobilenet_v3_large_320_fpn is not available in the installed torchvision build."
		)

	weights_enum = getattr(detection_models, "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights", None)
	categories = list(COCO_INSTANCE_CATEGORIES)

	try:
		if weights_enum is not None:
			weights = weights_enum.DEFAULT
			model = factory(weights=weights)
			meta = getattr(weights, "meta", {})
			weight_categories = meta.get("categories")
			if isinstance(weight_categories, list) and weight_categories:
				categories = [str(item) for item in weight_categories]
		else:  # pragma: no cover - compatibility path for older torchvision
			model = factory(pretrained=True)
	except Exception as exc:
		raise RuntimeError(
			"Could not load pretrained detector weights. On first use, torchvision may need internet access "
			"to download Faster R-CNN MobileNetV3 weights into the local torch cache."
		) from exc

	model.to(device)
	model.eval()
	return {
		"model": model,
		"device": device,
		"name": "Faster R-CNN MobileNetV3 320 FPN",
		"categories": categories,
	}


def box_iou(box_a: list[int], box_b: list[int]) -> float:
	ax0, ay0, ax1, ay1 = box_a
	bx0, by0, bx1, by1 = box_b
	inter_x0 = max(ax0, bx0)
	inter_y0 = max(ay0, by0)
	inter_x1 = min(ax1, bx1)
	inter_y1 = min(ay1, by1)
	if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
		return 0.0
	inter_area = float((inter_x1 - inter_x0) * (inter_y1 - inter_y0))
	area_a = float(max(ax1 - ax0, 0) * max(ay1 - ay0, 0))
	area_b = float(max(bx1 - bx0, 0) * max(by1 - by0, 0))
	union = max(area_a + area_b - inter_area, 1e-6)
	return inter_area / union


def expand_box(box: list[int], image_size: tuple[int, int], pad_ratio: float = 0.08) -> list[int]:
	width, height = image_size
	x0, y0, x1, y1 = box
	box_width = max(x1 - x0, 1)
	box_height = max(y1 - y0, 1)
	pad_x = int(round(box_width * pad_ratio))
	pad_y = int(round(box_height * pad_ratio))
	return [
		max(0, x0 - pad_x),
		max(0, y0 - pad_y),
		min(width, x1 + pad_x),
		min(height, y1 + pad_y),
	]


@torch.inference_mode()
def run_object_detector(
	detector_payload: Mapping,
	image: Image.Image,
	score_threshold: float,
	max_objects: int,
	min_area_fraction: float,
	relevant_only: bool,
) -> list[dict]:
	model = detector_payload["model"]
	device = detector_payload["device"]
	categories = detector_payload.get("categories", COCO_INSTANCE_CATEGORIES)

	rgb_image = image.convert("RGB")
	width, height = rgb_image.size
	input_tensor = transforms.functional.to_tensor(rgb_image).to(device)
	outputs = model([input_tensor])[0]

	boxes = outputs.get("boxes")
	labels = outputs.get("labels")
	scores = outputs.get("scores")
	if boxes is None or labels is None or scores is None:
		return []

	detections: list[dict] = []
	for box_tensor, label_tensor, score_tensor in zip(
		boxes.detach().cpu(),
		labels.detach().cpu(),
		scores.detach().cpu(),
	):
		score = float(score_tensor.item())
		if score < score_threshold:
			break

		label_idx = int(label_tensor.item())
		detector_label = resolve_detector_label(list(categories), label_idx)
		label_key = detector_label.lower()
		if label_key in {"n/a", "__background__"} or label_key in DETECTOR_EXCLUDED_LABELS:
			continue
		if relevant_only and label_key not in DETECTOR_PRIORITY_LABELS:
			continue

		x0, y0, x1, y1 = [int(round(v)) for v in box_tensor.tolist()]
		x0 = max(0, min(x0, width - 1))
		y0 = max(0, min(y0, height - 1))
		x1 = max(x0 + 1, min(x1, width))
		y1 = max(y0 + 1, min(y1, height))
		area_fraction = float(((x1 - x0) * (y1 - y0)) / max(width * height, 1))
		if area_fraction < min_area_fraction:
			continue

		candidate = {
			"box": [x0, y0, x1, y1],
			"detector_label": detector_label,
			"detector_score": score,
			"area_fraction": area_fraction,
		}
		if any(box_iou(candidate["box"], existing["box"]) >= 0.75 for existing in detections):
			continue

		detections.append(candidate)
		if len(detections) >= max_objects:
			break

	return detections


def summarize_detected_components(objects: list[dict]) -> list[dict]:
	component_map: dict[str, dict] = {}
	for obj in objects:
		component = str(obj["prediction"]["class_name"])
		row = component_map.setdefault(
			component,
			{
				"component": component,
				"count": 0,
				"hazard_level": obj["hazard_level"],
				"avg_classifier_confidence": 0.0,
				"avg_detector_score": 0.0,
				"peak_classifier_confidence": 0.0,
				"requires_review_count": 0,
				"detector_labels": set(),
			},
		)
		row["count"] += 1
		row["avg_classifier_confidence"] += float(obj["prediction"]["confidence"])
		row["avg_detector_score"] += float(obj["detector_score"])
		row["peak_classifier_confidence"] = max(
			float(row["peak_classifier_confidence"]),
			float(obj["prediction"]["confidence"]),
		)
		row["requires_review_count"] += int(bool(obj["needs_review"]))
		row["detector_labels"].add(str(obj["detector_label"]))
		if HAZARD_RANK.get(obj["hazard_level"], 0) > HAZARD_RANK.get(row["hazard_level"], 0):
			row["hazard_level"] = obj["hazard_level"]

	rows: list[dict] = []
	for row in component_map.values():
		count = max(int(row["count"]), 1)
		row["avg_classifier_confidence"] = float(row["avg_classifier_confidence"] / count)
		row["avg_detector_score"] = float(row["avg_detector_score"] / count)
		row["detector_labels"] = ", ".join(sorted(row["detector_labels"]))
		rows.append(row)

	return sorted(
		rows,
		key=lambda item: (
			int(item["count"]),
			float(item["peak_classifier_confidence"]),
			HAZARD_RANK.get(str(item["hazard_level"]), 0),
		),
		reverse=True,
	)


def build_aggregate_detection_result(
	*,
	analysis_kind: str,
	workflow_label: str,
	detector_name: str,
	detection_scope: str,
	objects: list[dict],
	confidence_threshold: float,
	fallback_to_all: bool,
	frame_count: int | None = None,
	sample_every_seconds: float | None = None,
) -> dict:
	component_rows = summarize_detected_components(objects)
	hazard_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
	for obj in objects:
		hazard_counts[obj["hazard_level"]] = hazard_counts.get(obj["hazard_level"], 0) + 1

	if not objects:
		detail = (
			f"{detector_name} did not return object proposals above the configured threshold for this {workflow_label}. "
			"That means the detector missed the scene or the threshold is too strict. Lower the detector threshold "
			"or switch detector scope to all proposals before relying on the report."
		)
		return {
			"analysis_kind": analysis_kind,
			"workflow_label": workflow_label,
			"detector_name": detector_name,
			"detection_scope": detection_scope,
			"fallback_to_all": fallback_to_all,
			"object_count": 0,
			"components": [],
			"objects": [],
			"hazard_counts": hazard_counts,
			"prediction": {
				"class_name": "No objects detected",
				"confidence": 0.0,
				"top_predictions": [],
			},
			"diagnostics": {
				"tone": "warning",
				"headline": "Detector found no actionable objects",
				"detail": detail,
				"needs_review": True,
			},
			"decision": {
				"component": "cluster",
				"hazard_level": "UNKNOWN",
				"material_profile": "no object crops were available for component classification",
				"disposal_pathway": "hold the belt segment for manual inspection",
				"short_recommendation": "hold the belt segment for manual inspection",
				"explanation": detail,
				"sdg_target": "SDG 12.4",
				"compliance_flag": False,
				"requires_human_review": True,
				"confidence_threshold": confidence_threshold,
				"confidence": 0.0,
				"agent_mode": analysis_kind,
				"explanation_source": "aggregated rule-based",
				"llm_provider": "none",
				"tool_trace": [
					{
						"step": "object_detection",
						"status": "attention",
						"summary": "The pretrained detector did not produce retained boxes for this input.",
					},
					{
						"step": "cluster_routing",
						"status": "blocked",
						"summary": "Because no objects were localized, final routing must be completed manually.",
					},
				],
				"detected_objects": 0,
				"unique_components": 0,
			},
		}

	cluster_hazard = max(
		(str(obj["hazard_level"]) for obj in objects),
		key=lambda level: HAZARD_RANK.get(level, 0),
	)
	review_count = sum(int(bool(obj["needs_review"])) for obj in objects)
	avg_detector_score = float(np.mean([float(obj["detector_score"]) for obj in objects]))
	avg_classifier_confidence = float(np.mean([float(obj["prediction"]["confidence"]) for obj in objects]))
	dominant_component = component_rows[0]["component"]
	dominant_confidence = float(component_rows[0]["peak_classifier_confidence"])
	component_preview = ", ".join(
		f"{row['component']} x{row['count']} ({row['hazard_level']})"
		for row in component_rows[:4]
	)
	if len(component_rows) > 4:
		component_preview += ", ..."

	if cluster_hazard == "HIGH":
		pathway = (
			"pause automated routing for this cluster, isolate the high-hazard items, and route each detected object "
			"through certified hazardous e-waste handling"
		)
		tone = "danger"
	elif cluster_hazard == "MEDIUM":
		pathway = (
			"separate the cluster into controlled appliance/component recovery lanes and verify each object before routing"
		)
		tone = "warning"
	else:
		pathway = "continue low-risk recovery with operator confirmation for the detected object set"
		tone = "success" if review_count == 0 else "warning"

	if review_count > 0:
		tone = "warning" if tone != "danger" else tone

	scope_line = f"Detector scope: {detection_scope}."
	if fallback_to_all:
		scope_line += " The workflow fell back to all COCO proposals because no e-waste-priority labels were retained."
	if frame_count and frame_count > 1:
		scope_line += (
			f" This report aggregates sampled detections across {frame_count} frames and does not perform multi-object tracking,"
			" so repeated items may appear more than once."
		)

	explanation = (
		f"{detector_name} localized {len(objects)} object crops for {workflow_label}. "
		f"Each crop was classified by the e-waste classifier, producing {len(component_rows)} component groupings: {component_preview}. "
		f"Highest hazard observed: {cluster_hazard}. {review_count} object(s) remain below the {confidence_threshold:.0%} routing threshold. "
		f"Recommended action: {pathway}. {scope_line}"
	)
	headline = (
		f"{len(objects)} detected object(s) across {len(component_rows)} predicted component group(s)"
		if analysis_kind != "video_belt_review"
		else f"{len(objects)} object events across {frame_count or 0} sampled frame(s)"
	)
	detail = (
		f"Dominant component evidence: {dominant_component} at {dominant_confidence:.2%}. "
		f"Average detector score {avg_detector_score:.2%}; average crop-classification confidence {avg_classifier_confidence:.2%}. "
		f"Highest hazard band present: {cluster_hazard}."
	)
	if sample_every_seconds is not None and frame_count and frame_count > 1:
		detail += f" Frames were sampled every {sample_every_seconds:.2f} second(s)."

	tool_trace = [
		{
			"step": "object_detection",
			"status": "completed",
			"summary": (
				f"{detector_name} retained {len(objects)} box(es) using {detection_scope}. "
				f"Average detector score was {avg_detector_score:.2%}."
			),
		},
		{
			"step": "crop_classification",
			"status": "completed",
			"summary": (
				f"Each localized crop was classified independently. "
				f"Average top-1 confidence was {avg_classifier_confidence:.2%}."
			),
		},
		{
			"step": "hazard_aggregation",
			"status": "completed",
			"summary": (
				f"Aggregated hazard counts: HIGH {hazard_counts.get('HIGH', 0)}, "
				f"MEDIUM {hazard_counts.get('MEDIUM', 0)}, LOW {hazard_counts.get('LOW', 0)}."
			),
		},
		{
			"step": "cluster_routing",
			"status": "completed" if review_count == 0 else "attention",
			"summary": explanation,
		},
	]
	if frame_count and frame_count > 1:
		tool_trace.insert(
			0,
			{
				"step": "video_sampling",
				"status": "completed",
				"summary": (
					f"Sampled {frame_count} frame(s) from uploaded video"
					f"{f' every {sample_every_seconds:.2f} second(s)' if sample_every_seconds is not None else ''}."
				),
			},
		)

	decision = {
		"component": dominant_component,
		"hazard_level": cluster_hazard,
		"material_profile": f"multi-object cluster containing {component_preview}",
		"disposal_pathway": pathway,
		"short_recommendation": pathway,
		"explanation": explanation,
		"sdg_target": "SDG 12.4" if cluster_hazard in {"HIGH", "MEDIUM", "UNKNOWN"} else "SDG 12.5",
		"compliance_flag": True,
		"requires_human_review": review_count > 0,
		"confidence_threshold": confidence_threshold,
		"confidence": dominant_confidence,
		"agent_mode": analysis_kind,
		"explanation_source": "aggregated rule-based",
		"llm_provider": "none",
		"tool_trace": tool_trace,
		"detected_objects": len(objects),
		"unique_components": len(component_rows),
	}
	if frame_count and frame_count > 1:
		decision["sampled_frames"] = frame_count

	return {
		"analysis_kind": analysis_kind,
		"workflow_label": workflow_label,
		"detector_name": detector_name,
		"detection_scope": detection_scope,
		"fallback_to_all": fallback_to_all,
		"object_count": len(objects),
		"components": component_rows,
		"objects": objects,
		"hazard_counts": hazard_counts,
		"prediction": {
			"class_name": dominant_component,
			"confidence": dominant_confidence,
			"top_predictions": [
				{
					"class_name": row["component"],
					"confidence": float(row["avg_classifier_confidence"]),
				}
				for row in component_rows[: min(5, len(component_rows))]
			],
		},
		"diagnostics": {
			"tone": tone,
			"headline": headline,
			"detail": detail,
			"needs_review": review_count > 0,
		},
		"decision": decision,
	}


def draw_detection_overlay(image: Image.Image, objects: list[dict]) -> Image.Image:
	canvas = image.convert("RGB").copy()
	draw = ImageDraw.Draw(canvas)

	for obj in objects:
		box = [int(v) for v in obj["box"]]
		hazard = str(obj.get("hazard_level", "UNKNOWN"))
		color = ImageColor.getrgb(HAZARD_COLOR.get(hazard, HAZARD_COLOR["UNKNOWN"]))
		label = (
			f"{obj['object_id']} | {obj['prediction']['class_name']} | "
			f"{obj['prediction']['confidence']:.0%} | {hazard}"
		)
		draw.rectangle(box, outline=color, width=4)
		try:
			text_box = draw.textbbox((0, 0), label)
			text_width = int(text_box[2] - text_box[0])
			text_height = int(text_box[3] - text_box[1])
		except Exception:  # pragma: no cover - pillow compatibility fallback
			text_width = max(110, len(label) * 7)
			text_height = 16
		text_x = box[0]
		text_y = max(0, box[1] - text_height - 8)
		draw.rectangle(
			[text_x, text_y, min(canvas.width, text_x + text_width + 10), text_y + text_height + 6],
			fill=(7, 16, 24),
			outline=color,
			width=2,
		)
		draw.text((text_x + 5, text_y + 3), label, fill=(242, 247, 255))

	return canvas


def build_detected_objects_frame(objects: list[dict]) -> pd.DataFrame:
	rows: list[dict] = []
	for obj in objects:
		row = {
			"object_id": obj["object_id"],
			"predicted_class": obj["prediction"]["class_name"],
			"classifier_confidence": float(obj["prediction"]["confidence"]),
			"detector_label": obj["detector_label"],
			"detector_score": float(obj["detector_score"]),
			"hazard_level": obj["hazard_level"],
			"requires_review": bool(obj["needs_review"]),
			"recommended_route": obj["decision"]["short_recommendation"],
		}
		if "frame_id" in obj:
			row["frame_id"] = obj["frame_id"]
		if "timestamp_s" in obj:
			row["timestamp_s"] = float(obj["timestamp_s"])
		rows.append(row)
	return pd.DataFrame(rows)


def build_exportable_detection_report(report: Mapping) -> dict:
	export = {
		"analysis_kind": report.get("analysis_kind"),
		"workflow_label": report.get("workflow_label"),
		"detector_name": report.get("detector_name"),
		"detection_scope": report.get("detection_scope"),
		"fallback_to_all": report.get("fallback_to_all"),
		"object_count": report.get("object_count"),
		"components": report.get("components"),
		"hazard_counts": report.get("hazard_counts"),
		"prediction": report.get("prediction"),
		"diagnostics": report.get("diagnostics"),
		"decision": report.get("decision"),
	}
	objects = []
	for obj in report.get("objects", []):
		objects.append(
			{
				"object_id": obj.get("object_id"),
				"box": obj.get("box"),
				"detector_label": obj.get("detector_label"),
				"detector_score": obj.get("detector_score"),
				"prediction": obj.get("prediction"),
				"hazard_level": obj.get("hazard_level"),
				"requires_review": obj.get("needs_review"),
				"frame_id": obj.get("frame_id"),
				"timestamp_s": obj.get("timestamp_s"),
			}
		)
	export["objects"] = objects
	if isinstance(report.get("frame_summaries"), list):
		export["frame_summaries"] = report.get("frame_summaries")
	return export


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


def get_agent_decision(component: str, confidence: float, threshold: float, use_llm: bool = True) -> dict:
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
	if not use_llm:
		agent.llm_fn = None
		agent.llm_provider = "none"
	return agent.run(AgentInput(component=component, confidence=confidence))


def analyze_detected_cluster(
	*,
	detector_payload: Mapping,
	classifier_model: nn.Module,
	image: Image.Image,
	class_names: list[str],
	classifier_device: torch.device,
	confidence_threshold: float,
	detector_score_threshold: float,
	max_objects: int,
	min_area_fraction: float,
	relevant_only: bool,
	crop_padding: float = 0.08,
) -> dict:
	rgb_image = image.convert("RGB")
	width, height = rgb_image.size

	detections = run_object_detector(
		detector_payload=detector_payload,
		image=rgb_image,
		score_threshold=detector_score_threshold,
		max_objects=max_objects,
		min_area_fraction=min_area_fraction,
		relevant_only=relevant_only,
	)
	fallback_to_all = False
	detection_scope = "e-waste priority COCO labels" if relevant_only else "all COCO proposals"
	if not detections and relevant_only:
		detections = run_object_detector(
			detector_payload=detector_payload,
			image=rgb_image,
			score_threshold=detector_score_threshold,
			max_objects=max_objects,
			min_area_fraction=min_area_fraction,
			relevant_only=False,
		)
		if detections:
			fallback_to_all = True
			detection_scope = "all COCO proposals fallback"

	objects: list[dict] = []
	for idx, detection in enumerate(detections, start=1):
		crop_box = expand_box(detection["box"], (width, height), pad_ratio=crop_padding)
		crop = rgb_image.crop(tuple(crop_box)).convert("RGB")
		prediction = infer_image(
			model=classifier_model,
			image=crop,
			class_names=class_names,
			device=classifier_device,
		)
		decision = get_agent_decision(
			prediction["class_name"],
			float(prediction["confidence"]),
			confidence_threshold,
			use_llm=False,
		)
		objects.append(
			{
				"object_id": f"O{idx}",
				"box": detection["box"],
				"crop_box": crop_box,
				"crop": crop,
				"detector_label": detection["detector_label"],
				"detector_score": float(detection["detector_score"]),
				"area_fraction": float(detection["area_fraction"]),
				"prediction": prediction,
				"decision": decision,
				"hazard_level": decision.get("hazard_level", "UNKNOWN"),
				"needs_review": bool(decision.get("requires_human_review", True)),
			}
		)

	report = build_aggregate_detection_result(
		analysis_kind="detector_assisted_cluster_review",
		workflow_label="cluster-image review",
		detector_name=str(detector_payload.get("name", "detector")),
		detection_scope=detection_scope,
		objects=objects,
		confidence_threshold=confidence_threshold,
		fallback_to_all=fallback_to_all,
	)
	report["overlay_image"] = draw_detection_overlay(rgb_image, objects) if objects else rgb_image
	return report


def sample_video_frames(
	video_bytes: bytes,
	file_suffix: str,
	sample_every_seconds: float,
	max_frames: int,
) -> dict:
	if cv2 is None:
		raise RuntimeError("opencv-python is not available, so video sampling cannot run in this environment.")

	tmp_path: str | None = None
	try:
		with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
			tmp_file.write(video_bytes)
			tmp_path = tmp_file.name

		capture = cv2.VideoCapture(tmp_path)
		if not capture.isOpened():
			raise RuntimeError("the uploaded video could not be opened")

		fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
		total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
		duration_seconds = (total_frames / fps) if fps > 0 and total_frames > 0 else None
		frame_stride = max(1, int(round(fps * sample_every_seconds))) if fps > 0 else max(1, int(round(sample_every_seconds * 10)))

		target_indices = list(range(0, max(total_frames, frame_stride * max_frames), frame_stride))[:max_frames]
		target_lookup = set(target_indices)
		frames: list[dict] = []
		frame_idx = 0
		while len(frames) < len(target_indices):
			ok, frame = capture.read()
			if not ok:
				break
			if frame_idx in target_lookup:
				frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				frames.append(
					{
						"frame_id": f"F{len(frames) + 1}",
						"frame_index": frame_idx,
						"timestamp_s": (frame_idx / fps) if fps > 0 else (len(frames) * sample_every_seconds),
						"image": Image.fromarray(frame_rgb),
					}
				)
			frame_idx += 1

		capture.release()
		if not frames:
			raise RuntimeError("no frames could be sampled from the uploaded video")
		return {
			"frames": frames,
			"fps": fps,
			"total_frames": total_frames,
			"duration_seconds": duration_seconds,
			"sample_every_seconds": sample_every_seconds,
		}
	finally:
		if tmp_path and os.path.exists(tmp_path):
			try:
				os.remove(tmp_path)
			except OSError:
				pass


def analyze_video_belt(
	*,
	detector_payload: Mapping,
	classifier_model: nn.Module,
	video_payload: Mapping,
	class_names: list[str],
	classifier_device: torch.device,
	confidence_threshold: float,
	detector_score_threshold: float,
	max_objects_per_frame: int,
	min_area_fraction: float,
	relevant_only: bool,
	crop_padding: float = 0.08,
) -> dict:
	frame_reports: list[dict] = []
	aggregated_objects: list[dict] = []
	frame_summaries: list[dict] = []

	for frame in video_payload.get("frames", []):
		frame_report = analyze_detected_cluster(
			detector_payload=detector_payload,
			classifier_model=classifier_model,
			image=frame["image"],
			class_names=class_names,
			classifier_device=classifier_device,
			confidence_threshold=confidence_threshold,
			detector_score_threshold=detector_score_threshold,
			max_objects=max_objects_per_frame,
			min_area_fraction=min_area_fraction,
			relevant_only=relevant_only,
			crop_padding=crop_padding,
		)
		frame_reports.append(
			{
				"frame_id": frame["frame_id"],
				"frame_index": frame["frame_index"],
				"timestamp_s": frame["timestamp_s"],
				"overlay_image": frame_report.get("overlay_image"),
				"report": frame_report,
			}
		)
		for obj in frame_report.get("objects", []):
			with_frame = dict(obj)
			with_frame["frame_id"] = frame["frame_id"]
			with_frame["timestamp_s"] = float(frame["timestamp_s"])
			aggregated_objects.append(with_frame)

		frame_decision = frame_report.get("decision", {})
		frame_prediction = frame_report.get("prediction", {})
		frame_summaries.append(
			{
				"frame_id": frame["frame_id"],
				"frame_index": int(frame["frame_index"]),
				"timestamp_s": float(frame["timestamp_s"]),
				"detected_objects": int(frame_report.get("object_count", 0)),
				"dominant_component": frame_prediction.get("class_name", "n/a"),
				"highest_hazard": frame_decision.get("hazard_level", "UNKNOWN"),
				"requires_review": bool(frame_decision.get("requires_human_review", True)),
			}
		)

	report = build_aggregate_detection_result(
		analysis_kind="video_belt_review",
		workflow_label="video-belt review",
		detector_name=str(detector_payload.get("name", "detector")),
		detection_scope="e-waste priority COCO labels" if relevant_only else "all COCO proposals",
		objects=aggregated_objects,
		confidence_threshold=confidence_threshold,
		fallback_to_all=False,
		frame_count=len(frame_reports),
		sample_every_seconds=to_float(video_payload.get("sample_every_seconds")),
	)
	report["frame_reports"] = frame_reports
	report["frame_summaries"] = frame_summaries
	report["video_meta"] = {
		"fps": to_float(video_payload.get("fps")),
		"total_frames": int(video_payload.get("total_frames", 0) or 0),
		"duration_seconds": to_float(video_payload.get("duration_seconds")),
		"sample_every_seconds": to_float(video_payload.get("sample_every_seconds")),
	}
	return report


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


def render_hero(
	title: str,
	copy: str,
	chips: list[str],
	side_notes: list[tuple[str, str, str]],
) -> None:
	chips_html = "".join(
		f"<span class='hero-chip'>{ICON_SVGS[guess_icon_name(chip)]}{escape_html(chip)}</span>"
		for chip in chips
	)
	side_html = "".join(
		dedent(
			f"""
			<div class="hero-side-note">
				<div class="hero-side-label">{escape_html(label)}</div>
				<div class="hero-side-value">{escape_html(value)}</div>
				<div class="hero-side-copy">{format_html_copy(detail)}</div>
			</div>
			"""
		).strip()
		for label, value, detail in side_notes
	)
	render_html(
		f"""
		<section class="hero-shell">
			<div class="hero-ribbon">{ICON_SVGS['stack']}Research operations surface</div>
			<div class="hero-grid">
				<div>
					<h1 class="hero-title">{escape_html(title)}</h1>
					<p class="hero-copy">{format_html_copy(copy)}</p>
					<div class="hero-chip-row">{chips_html}</div>
				</div>
				<div class="hero-side">{side_html}</div>
			</div>
		</section>
		"""
	)


def render_metric_tile(
	label: str,
	value: str,
	detail: str,
	tone: str = "neutral",
	icon: str | None = None,
) -> None:
	icon_name = icon or guess_icon_name(label)
	render_html(
		f"""
		<div class="metric-shell tone-{escape_html(tone)}">
			<div class="metric-top">
				<div class="metric-icon">{ICON_SVGS[icon_name]}</div>
				<div>
					<div class="metric-label">{escape_html(label)}</div>
					<div class="metric-value">{escape_html(value)}</div>
				</div>
			</div>
			<div class="metric-detail">{format_html_copy(detail)}</div>
		</div>
		"""
	)


def render_panel(
	title: str,
	value: str,
	copy: str,
	tone: str = "neutral",
	icon: str | None = None,
) -> None:
	icon_name = icon or guess_icon_name(title)
	render_html(
		f"""
		<div class="panel-shell tone-{escape_html(tone)}">
			<div class="panel-top">
				<div class="panel-icon">{ICON_SVGS[icon_name]}</div>
				<div>
					<div class="panel-label">{escape_html(title)}</div>
					<div class="panel-value">{escape_html(value)}</div>
				</div>
			</div>
			<div class="panel-copy">{format_html_copy(copy)}</div>
		</div>
		"""
	)


def render_banner(
	title: str,
	copy: str,
	tone: str = "neutral",
	icon: str | None = None,
) -> None:
	icon_name = icon or guess_icon_name(title)
	render_html(
		f"""
		<div class="banner-shell tone-{escape_html(tone)}">
			<div class="banner-icon">{ICON_SVGS[icon_name]}</div>
			<div>
				<div class="banner-title">{escape_html(title)}</div>
				<div class="banner-copy">{format_html_copy(copy)}</div>
			</div>
		</div>
		"""
	)


def render_probability_rows(top_predictions: list[dict]) -> None:
	for item in top_predictions:
		confidence = float(item["confidence"])
		left, right = st.columns([4, 1], gap="small")
		with left:
			st.markdown(f"**{item['class_name']}**")
		with right:
			st.markdown(f"**{confidence:.2%}**")
		st.progress(max(0, min(100, int(round(confidence * 100)))))


def render_section_intro(
	kicker: str,
	headline: str,
	copy: str,
	icon: str | None = None,
) -> None:
	icon_name = icon or guess_icon_name(f"{kicker} {headline}")
	render_html(
		f"""
		<section class="section-shell">
			<div class="section-head">
				<div class="section-icon">{ICON_SVGS[icon_name]}</div>
				<div>
					<div class="section-kicker">{escape_html(kicker)}</div>
					<h2 class="section-title">{escape_html(headline)}</h2>
				</div>
			</div>
			<p class="section-copy">{format_html_copy(copy)}</p>
		</section>
		"""
	)


def render_badge_row(badges: list[str]) -> None:
	chips = "".join(f"<span class='badge-pill'>{escape_html(badge)}</span>" for badge in badges)
	render_html(f"<div class='badge-row'>{chips}</div>")


def render_trace_steps(trace_steps: list[dict]) -> None:
	if not trace_steps:
		return
	steps_html = "".join(
		dedent(
			f"""
			<div class="timeline-step">
				<div class="timeline-dot"></div>
				<div>
					<div class="timeline-step-name">{escape_html(step.get('step', 'step'))}</div>
					<div class="timeline-step-title">{format_html_copy(step.get('summary', 'No summary available.'))}</div>
					<div class="timeline-status">{escape_html(step.get('status', 'unknown'))}</div>
				</div>
			</div>
			"""
		).strip()
		for step in trace_steps
	)
	render_html(f"<div class='timeline-shell'>{steps_html}</div>")


def render_checklist(items: list[str]) -> None:
	if not items:
		return
	items_html = "".join(
		dedent(
			f"""
			<div class="checklist-item">
				<div class="checklist-dot"></div>
				<div class="checklist-text">{format_html_copy(item)}</div>
			</div>
			"""
		).strip()
		for item in items
	)
	render_html(f"<div class='checklist-shell'>{items_html}</div>")


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


def build_cluster_operational_checklist(report: Mapping) -> list[str]:
	decision = report.get("decision", {}) if isinstance(report.get("decision"), Mapping) else {}
	hazard_counts = report.get("hazard_counts", {}) if isinstance(report.get("hazard_counts"), Mapping) else {}
	checklist = [
		"Retain the detector overlay, per-object classification table, and aggregated routing summary for the batch audit trail.",
		"Route objects by their individual hazard band, not just by the dominant cluster label.",
	]
	if hazard_counts.get("HIGH", 0):
		checklist.append("Isolate the high-hazard detections from general throughput before downstream disposal handling.")
	if decision.get("requires_human_review", True):
		checklist.append("Escalate low-confidence or ambiguous detections to a human operator before releasing the cluster decision.")
	if report.get("analysis_kind") == "video_belt_review":
		checklist.append("Treat sampled frame detections as event counts only until object tracking is added; repeated items may appear in multiple frames.")
	return checklist


def truncate_text(value: object, limit: int = 88) -> str:
	text = str(value or "")
	if len(text) <= limit:
		return text
	return text[: max(limit - 3, 0)].rstrip() + "..."


def build_processing_note(component: str, hazard_level: str, disposal_pathway: str) -> str:
	component_key = component.lower()
	pathway_key = disposal_pathway.lower()

	if "battery" in component_key or "battery" in pathway_key:
		return "Isolate the cell pack, prevent short-circuit risk, and move the unit into certified battery recovery."
	if component_key in {"refrigerator", "air-conditioner"} or "refrigerant" in pathway_key:
		return "Recover refrigerant and gas-bearing assemblies before dismantling the remaining metal and plastic body."
	if "printer" in component_key or "toner" in pathway_key:
		return "Separate toner-bearing modules first, then route the shell and PCB content through controlled e-waste recovery."
	if component_key in {"pcb", "microchip-ic", "passive-component", "resistor", "transistor"} or "metal recovery" in pathway_key:
		return "Send this electronics-heavy stream through certified metal recovery and residue-safe processing."
	if "light bulbs" in component_key:
		return "Keep fragile lamp units isolated and route them through lamp-safe hazardous waste treatment."
	if hazard_level == "HIGH":
		return "Keep the item isolated from general throughput and process it only through certified hazardous e-waste handling."
	if hazard_level == "MEDIUM":
		return "Route through controlled dismantling and verify condition before releasing the component into recovery lanes."
	return "Confirm the label, then move the item into low-risk material recovery with normal e-waste segregation."


def build_component_route_rows(objects: list[dict]) -> list[dict]:
	component_map: dict[str, dict] = {}
	for obj in objects:
		component = str(obj.get("predicted_class", "Unknown"))
		confidence = to_float(obj.get("classifier_confidence")) or 0.0
		hazard = str(obj.get("hazard_level", "UNKNOWN"))
		entry = component_map.setdefault(
			component,
			{
				"component": component,
				"count": 0,
				"hazard_level": hazard,
				"avg_confidence": 0.0,
				"peak_confidence": 0.0,
				"review_required": False,
				"material_profile": str(obj.get("material_profile", MATERIAL_MAP.get(component, "n/a"))),
				"disposal_pathway": str(obj.get("disposal_pathway", DISPOSAL_MAP.get(component, "manual review"))),
				"sdg_target": str(obj.get("sdg_target", "SDG 12.4")),
			},
		)
		entry["count"] += 1
		entry["avg_confidence"] += confidence
		entry["peak_confidence"] = max(float(entry["peak_confidence"]), confidence)
		entry["review_required"] = bool(entry["review_required"]) or bool(obj.get("requires_review", True))
		if HAZARD_RANK.get(hazard, 0) > HAZARD_RANK.get(str(entry["hazard_level"]), 0):
			entry["hazard_level"] = hazard
		if obj.get("material_profile"):
			entry["material_profile"] = str(obj["material_profile"])
		if obj.get("disposal_pathway"):
			entry["disposal_pathway"] = str(obj["disposal_pathway"])
		if obj.get("sdg_target"):
			entry["sdg_target"] = str(obj["sdg_target"])

	rows: list[dict] = []
	for entry in component_map.values():
		count = max(int(entry["count"]), 1)
		entry["avg_confidence"] = float(entry["avg_confidence"] / count)
		entry["process_further"] = build_processing_note(
			component=str(entry["component"]),
			hazard_level=str(entry["hazard_level"]),
			disposal_pathway=str(entry["disposal_pathway"]),
		)
		rows.append(entry)

	return sorted(
		rows,
		key=lambda item: (
			HAZARD_RANK.get(str(item["hazard_level"]), 0),
			int(item["count"]),
			float(item["peak_confidence"]),
		),
		reverse=True,
	)


def build_unified_workflow_summary(last_result: Mapping | None, session_state: Mapping) -> dict | None:
	if not isinstance(last_result, Mapping) or not last_result:
		return None

	analysis_kind = str(last_result.get("analysis_kind", "single_item_review"))
	prediction = last_result.get("prediction", {}) if isinstance(last_result.get("prediction"), Mapping) else {}
	decision = last_result.get("decision", {}) if isinstance(last_result.get("decision"), Mapping) else {}
	diagnostics = last_result.get("diagnostics", {}) if isinstance(last_result.get("diagnostics"), Mapping) else {}

	workflow_name_map = {
		"single_item_review": "Single-item triage",
		"cluster_image_review": "Cluster image review",
		"detector_assisted_cluster_review": "Cluster image review",
		"video_belt_review": "Video belt review",
	}
	workflow_name = workflow_name_map.get(analysis_kind, "Workflow review")

	report: Mapping | None = None
	preview_image: Image.Image | None = None
	source_label = "active session"
	normalized_objects: list[dict] = []
	hazard_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
	frame_summaries: list[dict] = []
	video_meta: dict = {}
	localization_copy = "The image goes directly into the classifier because the benchmarked lane assumes one dominant component."
	intake_copy = "Single uploaded image under one-label triage assumptions."
	detection_scope = "direct classifier"
	detector_name = "Not required"
	media_badges: list[str] = []
	object_count_value = 1

	if analysis_kind == "single_item_review":
		component = str(prediction.get("class_name", "Unknown"))
		confidence = to_float(prediction.get("confidence")) or 0.0
		hazard = str(decision.get("hazard_level", "UNKNOWN"))
		preview_image = session_state.get("pending_image") if isinstance(session_state.get("pending_image"), Image.Image) else None
		source_label = str(session_state.get("pending_image_label") or "uploaded image")
		media_badges = [f"source {source_label}", "direct single-image lane", "no detector stage"]
		normalized_objects = [
			{
				"object_id": "O1",
				"predicted_class": component,
				"classifier_confidence": confidence,
				"detector_label": "direct image intake",
				"detector_score": 1.0,
				"hazard_level": hazard,
				"requires_review": bool(decision.get("requires_human_review", True)),
				"recommended_route": str(decision.get("short_recommendation", decision.get("disposal_pathway", "manual review"))),
				"material_profile": str(decision.get("material_profile", MATERIAL_MAP.get(component, "n/a"))),
				"disposal_pathway": str(decision.get("disposal_pathway", DISPOSAL_MAP.get(component, "manual review"))),
				"sdg_target": str(decision.get("sdg_target", "SDG 12.4")),
			}
		]
		normalized_objects[0]["process_further"] = build_processing_note(
			component=component,
			hazard_level=hazard,
			disposal_pathway=normalized_objects[0]["disposal_pathway"],
		)
		hazard_counts[hazard] = hazard_counts.get(hazard, 0) + 1
	else:
		report_key = "video_report" if analysis_kind == "video_belt_review" else "cluster_report"
		report = last_result.get(report_key) if isinstance(last_result.get(report_key), Mapping) else None
		if analysis_kind == "video_belt_review":
			intake_copy = "Uploaded conveyor-belt footage sampled at a controlled frame interval."
			detector_name = str(report.get("detector_name", "Pretrained detector")) if report else "Pretrained detector"
			detection_scope = str(report.get("detection_scope", "detector-assisted")) if report else "detector-assisted"
			video_meta = dict(report.get("video_meta", {})) if report and isinstance(report.get("video_meta"), Mapping) else {}
			frame_summaries = list(report.get("frame_summaries", [])) if report and isinstance(report.get("frame_summaries"), list) else []
			first_frame_report = report.get("frame_reports", []) if report and isinstance(report.get("frame_reports"), list) else []
			preview_image = first_frame_report[0].get("overlay_image") if first_frame_report else None
			source_label = str(session_state.get("pending_video_name") or "uploaded video")
			sample_every = to_float(video_meta.get("sample_every_seconds"))
			localization_copy = (
				f"Frames are sampled{' every ' + format_float(sample_every, 2) + ' s' if sample_every is not None else ''}, "
				"then hybrid localization uses pretrained detector proposals and classifier scene evidence before crop classification."
			)
			media_badges = [
				f"source {source_label}",
				f"sampled frames {len(frame_summaries)}",
				"frame-wise aggregation",
			]
		else:
			intake_copy = "Mixed-object e-waste image under detector-assisted review."
			detector_name = str(report.get("detector_name", "Pretrained detector")) if report else "Pretrained detector"
			detection_scope = str(report.get("detection_scope", "detector-assisted")) if report else "detector-assisted"
			preview_image = report.get("overlay_image") if report and isinstance(report.get("overlay_image"), Image.Image) else None
			source_label = str(session_state.get("cluster_pending_label") or "cluster image")
			localization_copy = (
				f"{detector_name} proposes candidate objects and the classifier scene scan helps recover mixed small parts "
				f"before each retained crop is classified."
			)
			media_badges = [
				f"source {source_label}",
				f"scope {detection_scope}",
				"detector + crop classifier",
			]

		if report:
			object_count_value = int(report.get("object_count", 0) or 0)
			hazard_counts = dict(report.get("hazard_counts", hazard_counts)) if isinstance(report.get("hazard_counts"), Mapping) else hazard_counts
			for obj in report.get("objects", []):
				if not isinstance(obj, Mapping):
					continue
				pred = obj.get("prediction", {}) if isinstance(obj.get("prediction"), Mapping) else {}
				obj_decision = obj.get("decision", {}) if isinstance(obj.get("decision"), Mapping) else {}
				component = str(pred.get("class_name", "Unknown"))
				pathway = str(obj_decision.get("short_recommendation", obj_decision.get("disposal_pathway", DISPOSAL_MAP.get(component, "manual review"))))
				hazard = str(obj.get("hazard_level", obj_decision.get("hazard_level", "UNKNOWN")))
				normalized_row = {
					"object_id": str(obj.get("object_id", f"O{len(normalized_objects) + 1}")),
					"predicted_class": component,
					"classifier_confidence": to_float(pred.get("confidence")) or 0.0,
					"detector_label": str(obj.get("detector_label", "detector proposal")),
					"detector_score": to_float(obj.get("detector_score")) or 0.0,
					"hazard_level": hazard,
					"requires_review": bool(obj.get("needs_review", obj_decision.get("requires_human_review", True))),
					"recommended_route": pathway,
					"material_profile": str(obj_decision.get("material_profile", MATERIAL_MAP.get(component, "n/a"))),
					"disposal_pathway": pathway,
					"sdg_target": str(obj_decision.get("sdg_target", "SDG 12.4")),
					"frame_id": obj.get("frame_id"),
					"timestamp_s": to_float(obj.get("timestamp_s")),
				}
				normalized_row["process_further"] = build_processing_note(
					component=component,
					hazard_level=hazard,
					disposal_pathway=normalized_row["disposal_pathway"],
				)
				normalized_objects.append(normalized_row)

	if not normalized_objects:
		component = str(decision.get("component", prediction.get("class_name", "Manual review")))
		pathway = str(decision.get("short_recommendation", decision.get("disposal_pathway", "manual review")))
		hazard = str(decision.get("hazard_level", "UNKNOWN"))
		normalized_objects = [
			{
				"object_id": "M1",
				"predicted_class": component,
				"classifier_confidence": to_float(prediction.get("confidence")) or 0.0,
				"detector_label": "no retained object",
				"detector_score": 0.0,
				"hazard_level": hazard,
				"requires_review": bool(decision.get("requires_human_review", True)),
				"recommended_route": pathway,
				"material_profile": str(decision.get("material_profile", "manual inspection required")),
				"disposal_pathway": pathway,
				"sdg_target": str(decision.get("sdg_target", "SDG 12.4")),
				"process_further": build_processing_note(component=component, hazard_level=hazard, disposal_pathway=pathway),
				"frame_id": None,
				"timestamp_s": None,
			}
		]

	component_rows = build_component_route_rows(normalized_objects)
	route_rows = build_route_distribution_rows(component_rows)
	stream_rows = build_operational_stream_rows(component_rows)
	dominant_component = str(prediction.get("class_name", component_rows[0]["component"] if component_rows else "n/a"))
	dominant_confidence = to_float(prediction.get("confidence")) or (to_float(component_rows[0]["peak_confidence"]) if component_rows else 0.0) or 0.0
	unique_components = len(component_rows)
	review_required = bool(decision.get("requires_human_review", True))
	route_count = len({row["disposal_pathway"] for row in component_rows if row.get("disposal_pathway")})
	highest_hazard = str(decision.get("hazard_level", "UNKNOWN"))
	high_hazard_count = int(hazard_counts.get("HIGH", 0) or 0)
	medium_hazard_count = int(hazard_counts.get("MEDIUM", 0) or 0)
	low_hazard_count = int(hazard_counts.get("LOW", 0) or 0)
	review_object_count = sum(1 for row in normalized_objects if row.get("requires_review"))
	automation_clear_rate = 0.0 if not normalized_objects else max(len(normalized_objects) - review_object_count, 0) / len(normalized_objects)
	average_confidence = 0.0 if not normalized_objects else sum(float(row.get("classifier_confidence", 0.0) or 0.0) for row in normalized_objects) / len(normalized_objects)
	detector_scores = [float(row.get("detector_score", 0.0) or 0.0) for row in normalized_objects if float(row.get("detector_score", 0.0) or 0.0) > 0]
	average_detector_score = (sum(detector_scores) / len(detector_scores)) if detector_scores else None
	objects_per_frame = (object_count_value / len(frame_summaries)) if frame_summaries else None
	hazard_mix_rows = []
	for label in ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]:
		count = int(hazard_counts.get(label, 0) or 0)
		hazard_mix_rows.append(
			{
				"hazard": label,
				"count": count,
				"share": (count / max(object_count_value, 1)) if object_count_value else 0.0,
			}
		)
	process_notes = [f"{row.get('component', 'n/a')}: {row.get('process_further', 'n/a')}" for row in component_rows[:5]]
	sdg_targets = sorted({str(row.get("sdg_target", "SDG 12.4")) for row in component_rows if row.get("sdg_target")})
	sdg_alignment_copy = (
		f"This run supports SDG 12.4 by separating {object_count_value} retained object event(s) into {route_count} explicit downstream stream(s), "
		f"preserving auditable hazard logic for {unique_components} component group(s), and escalating {review_object_count} item(s) for human review."
	)
	workflow_headline = str(diagnostics.get("headline", "Workflow summary"))
	workflow_detail = str(diagnostics.get("detail", "No diagnostics available."))

	stage_cards = [
		{
			"label": "01 Intake",
			"title": workflow_name,
			"copy": intake_copy,
			"footer": [source_label, f"latency {last_result.get('elapsed_ms', 'n/a')} ms"],
			"icon": "gallery",
		},
		{
			"label": "02 Localization",
			"title": "Object discovery lane",
			"copy": localization_copy,
			"footer": (
				["direct classifier", "1 dominant object assumption"]
				if analysis_kind == "single_item_review"
				else [detector_name, f"retained {object_count_value} object event(s)"]
			),
			"icon": "cluster" if analysis_kind != "single_item_review" else "review",
		},
		{
			"label": "03 Classification",
			"title": "ConvNeXt-Tiny evidence",
			"copy": (
				f"The classifier assigns component labels to {'the full image' if analysis_kind == 'single_item_review' else 'each retained crop'}, "
				f"with dominant evidence for {dominant_component} at {dominant_confidence:.2%}."
			),
			"footer": [f"top class {dominant_component}", f"confidence {dominant_confidence:.2%}"],
			"icon": "cpu",
		},
		{
			"label": "04 Hard Mapping",
			"title": "Hazard and material registry",
			"copy": (
				"Predicted components are matched against fixed hazard, material, disposal, and SDG mappings so the downstream logic stays explicit and auditable."
			),
			"footer": [f"unique components {unique_components}", f"route patterns {route_count}"],
			"icon": "database",
		},
		{
			"label": "05 Decision Support",
			"title": "Routing and review control",
			"copy": truncate_text(decision.get("explanation", "No decision explanation available."), 180),
			"footer": [
				f"highest hazard {highest_hazard}",
				"human review required" if review_required else "human review clear",
			],
			"icon": "route",
		},
		{
			"label": "06 Output",
			"title": "Operator-ready dossier",
			"copy": (
				"This page consolidates visuals, object summaries, hard mappings, materials, disposal actions, and a downloadable PDF report for faculty or operator review."
			),
			"footer": ["workflow atlas", "PDF export", "audit-ready summary"],
			"icon": "chart",
		},
	]

	component_frame = pd.DataFrame(component_rows)
	if not component_frame.empty:
		component_frame["avg_confidence"] = component_frame["avg_confidence"].map(lambda x: f"{float(x):.2%}")
		component_frame["peak_confidence"] = component_frame["peak_confidence"].map(lambda x: f"{float(x):.2%}")
		component_frame["review_required"] = component_frame["review_required"].map(lambda x: "Yes" if x else "No")

	object_frame = pd.DataFrame(normalized_objects)
	if not object_frame.empty:
		object_frame["classifier_confidence"] = object_frame["classifier_confidence"].map(lambda x: f"{float(x):.2%}")
		object_frame["detector_score"] = object_frame["detector_score"].map(lambda x: f"{float(x):.2%}")
		if "timestamp_s" in object_frame.columns:
			object_frame["timestamp_s"] = object_frame["timestamp_s"].map(lambda x: "" if x is None else f"{float(x):.2f}")
		object_frame["requires_review"] = object_frame["requires_review"].map(lambda x: "Yes" if x else "No")

	export_objects = []
	for row in normalized_objects:
		export_row = dict(row)
		export_objects.append(export_row)

	export_payload = {
		"analysis_kind": analysis_kind,
		"workflow_name": workflow_name,
		"source_label": source_label,
		"prediction": prediction,
		"decision": decision,
		"diagnostics": diagnostics,
		"components": component_rows,
		"objects": export_objects,
		"hazard_counts": hazard_counts,
		"frame_summaries": frame_summaries,
		"video_meta": video_meta,
		"stage_cards": stage_cards,
	}

	return {
		"analysis_kind": analysis_kind,
		"workflow_name": workflow_name,
		"source_label": source_label,
		"preview_image": preview_image,
		"prediction": prediction,
		"decision": decision,
		"diagnostics": diagnostics,
		"headline": workflow_headline,
		"detail": workflow_detail,
		"media_badges": media_badges,
		"stage_cards": stage_cards,
		"component_rows": component_rows,
		"route_rows": route_rows,
		"stream_rows": stream_rows,
		"hazard_mix_rows": hazard_mix_rows,
		"component_frame": component_frame,
		"object_frame": object_frame,
		"objects": normalized_objects,
		"hazard_counts": hazard_counts,
		"checklist": (
			build_cluster_operational_checklist(report or last_result)
			if analysis_kind != "single_item_review"
			else build_operational_checklist(decision, diagnostics)
		),
		"trace_steps": decision.get("tool_trace", []) if isinstance(decision.get("tool_trace"), list) else [],
		"frame_summaries": frame_summaries,
		"video_meta": video_meta,
		"dominant_component": dominant_component,
		"dominant_confidence": dominant_confidence,
		"object_count_value": object_count_value if analysis_kind != "single_item_review" else 1,
		"unique_components": unique_components,
		"highest_hazard": highest_hazard,
		"route_count": route_count,
		"high_hazard_count": high_hazard_count,
		"medium_hazard_count": medium_hazard_count,
		"low_hazard_count": low_hazard_count,
		"review_object_count": review_object_count,
		"automation_clear_rate": automation_clear_rate,
		"average_confidence": average_confidence,
		"average_detector_score": average_detector_score,
		"objects_per_frame": objects_per_frame,
		"process_notes": process_notes,
		"sdg_targets": sdg_targets,
		"sdg_alignment_copy": sdg_alignment_copy,
		"review_required": review_required,
		"export_payload": export_payload,
	}


def render_mapping_cards(component_rows: list[dict]) -> None:
	card_markup = []
	for row in component_rows[:4]:
		card_markup.append(
			dedent(
				f"""
				<div class="mapping-card">
					<div class="mapping-card-label">{escape_html(row.get('component', 'Component'))}</div>
					<div class="mapping-card-value">{escape_html(row.get('hazard_level', 'UNKNOWN'))} hazard | x{escape_html(row.get('count', 0))}</div>
					<div class="mapping-card-copy"><strong>Material:</strong> {escape_html(row.get('material_profile', 'n/a'))}</div>
					<div class="mapping-card-copy"><strong>Route:</strong> {escape_html(row.get('disposal_pathway', 'n/a'))}</div>
					<div class="mapping-card-copy"><strong>Process next:</strong> {escape_html(row.get('process_further', 'n/a'))}</div>
				</div>
				"""
			).strip()
		)

	render_html(
		f"""
		<div class="mapping-shell">
			<div class="mapping-grid">{''.join(card_markup)}</div>
		</div>
		"""
	)


def build_route_distribution_rows(component_rows: list[dict]) -> list[dict]:
	route_map: dict[str, int] = {}
	for row in component_rows:
		pathway = str(row.get("disposal_pathway", "manual review"))
		route_map[pathway] = route_map.get(pathway, 0) + int(row.get("count", 0) or 0)
	return sorted(
		[{"route": route, "count": count} for route, count in route_map.items()],
		key=lambda item: item["count"],
		reverse=True,
	)


def infer_operational_stream(component: str, disposal_pathway: str, hazard_level: str) -> str:
	component_key = component.lower()
	pathway_key = disposal_pathway.lower()
	hazard_key = hazard_level.upper()

	if "battery" in component_key or "battery" in pathway_key:
		return "Battery recovery"
	if component_key in {"refrigerator", "air-conditioner"} or "refrigerant" in pathway_key or "gas" in pathway_key:
		return "Refrigerant recovery"
	if "light bulb" in component_key or "lamp" in pathway_key:
		return "Lamp hazardous handling"
	if component_key in {"pcb", "microchip-ic", "passive-component", "resistor", "transistor"} or "metal recovery" in pathway_key:
		return "PCB and metal recovery"
	if "manual" in pathway_key or "review" in pathway_key:
		return "Manual review gate"
	if "dismant" in pathway_key:
		return "Controlled dismantling"
	if hazard_key == "HIGH":
		return "Certified hazardous stream"
	return "General e-waste recovery"


def build_operational_stream_rows(component_rows: list[dict]) -> list[dict]:
	stream_map: dict[str, int] = {}
	for row in component_rows:
		stream = infer_operational_stream(
			component=str(row.get("component", "Unknown")),
			disposal_pathway=str(row.get("disposal_pathway", "manual review")),
			hazard_level=str(row.get("hazard_level", "UNKNOWN")),
		)
		stream_map[stream] = stream_map.get(stream, 0) + int(row.get("count", 0) or 0)
	return sorted(
		[{"stream": stream, "count": count} for stream, count in stream_map.items()],
		key=lambda item: item["count"],
		reverse=True,
	)


def build_component_confidence_rows(component_rows: list[dict]) -> list[dict]:
	rows: list[dict] = []
	for row in component_rows:
		rows.append(
			{
				"component": str(row.get("component", "Unknown")),
				"avg_confidence": float(to_float(row.get("avg_confidence")) or 0.0),
				"hazard_level": str(row.get("hazard_level", "UNKNOWN")),
			}
		)
	return sorted(rows, key=lambda item: item["avg_confidence"], reverse=True)


def style_plot_axis(ax: plt.Axes, title: str | None = None) -> None:
	ax.set_facecolor("#102131")
	for spine in ax.spines.values():
		spine.set_color("#35506b")
		spine.set_linewidth(1.0)
	ax.tick_params(colors="#dce7f2", labelsize=9)
	ax.yaxis.label.set_color("#dce7f2")
	ax.xaxis.label.set_color("#dce7f2")
	if title:
		ax.set_title(title, color="#f7fbff", fontsize=12, fontweight="bold", pad=10)


def wrap_plot_text(value: object, width: int, max_lines: int | None = None) -> list[str]:
	lines = wrap(str(value or ""), width=width) or [""]
	if max_lines is not None and len(lines) > max_lines:
		trimmed = lines[:max_lines]
		trimmed[-1] = truncate_text(" ".join(lines[max_lines - 1 :]), width)
		return trimmed
	return lines


def draw_canvas_stat_card(
	ax: plt.Axes,
	*,
	x: float,
	y: float,
	w: float,
	h: float,
	title: str,
	value: str,
	detail: str,
	accent: str,
) -> None:
	card = FancyBboxPatch(
		(x, y),
		w,
		h,
		boxstyle="round,pad=0.012,rounding_size=0.024",
		linewidth=1.35,
		edgecolor=accent,
		facecolor="#102131",
	)
	ax.add_patch(card)
	ax.text(x + 0.018, y + h - 0.03, title.upper(), color="#92a4b8", fontsize=8.5, fontweight="bold", va="top")
	ax.text(x + 0.018, y + h - 0.075, value, color="#f7fbff", fontsize=16, fontweight="bold", va="top")
	ax.text(x + 0.018, y + 0.028, detail, color="#d5e2ee", fontsize=8.8, va="bottom")


def build_workflow_diagram_figure(summary: Mapping, *, pdf_mode: bool = False) -> plt.Figure:
	stage_cards = list(summary.get("stage_cards", []))
	fig = plt.figure(figsize=(11.69, 8.27) if pdf_mode else (12.8, 8.6), facecolor="#071018")
	ax = fig.add_axes([0, 0, 1, 1])
	ax.set_axis_off()
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)

	ax.text(0.055, 0.955, "Workflow Blueprint", color="#f7fbff", fontsize=22 if pdf_mode else 24, fontweight="bold", va="top")
	ax.text(
		0.055,
		0.918,
		f"{summary.get('workflow_name', 'Workflow review')} | source: {summary.get('source_label', 'active session')}",
		color="#9ce7ed",
		fontsize=11 if pdf_mode else 12,
		va="top",
	)

	stage_accents = ["#63d0d9", "#7ad9e0", "#3cc58d", "#f4b168", "#f38b59", "#ff6f61"]
	box_x = 0.1
	box_w = 0.8
	box_h = 0.102
	start_y = 0.83
	step_y = 0.12

	for idx, card in enumerate(stage_cards[:6]):
		y = start_y - idx * step_y
		accent = stage_accents[min(idx, len(stage_accents) - 1)]
		box = FancyBboxPatch(
			(box_x, y - box_h),
			box_w,
			box_h,
			boxstyle="round,pad=0.012,rounding_size=0.025",
			linewidth=1.6,
			edgecolor=accent,
			facecolor="#102131",
		)
		ax.add_patch(box)
		ax.add_patch(
			FancyBboxPatch(
				(box_x + 0.012, y - box_h + 0.018),
				0.062,
				box_h - 0.036,
				boxstyle="round,pad=0.01,rounding_size=0.02",
				linewidth=0,
				facecolor=accent,
				alpha=0.18,
			)
		)
		ax.text(box_x + 0.028, y - 0.044, f"{idx + 1:02d}", color=accent, fontsize=16, fontweight="bold", va="center")
		ax.text(box_x + 0.092, y - 0.026, str(card.get("label", f"Stage {idx + 1}")).upper(), color="#92a4b8", fontsize=8.5, fontweight="bold", va="top")
		ax.text(box_x + 0.092, y - 0.05, str(card.get("title", "")), color="#f7fbff", fontsize=13, fontweight="bold", va="top")
		copy_lines = wrap_plot_text(card.get("copy", ""), width=84 if pdf_mode else 88, max_lines=3)
		ax.text(box_x + 0.092, y - 0.077, "\n".join(copy_lines), color="#d5e2ee", fontsize=9.2, va="top", linespacing=1.25)
		footer = "  |  ".join(str(item) for item in card.get("footer", [])[:2] if str(item).strip())
		ax.text(box_x + box_w - 0.016, y - box_h + 0.018, footer, color=accent, fontsize=8.4, fontweight="bold", va="bottom", ha="right")

		if idx < min(len(stage_cards), 6) - 1:
			ax.annotate(
				"",
				xy=(0.5, y - box_h - 0.008),
				xytext=(0.5, y - box_h - 0.035),
				arrowprops=dict(arrowstyle="-|>", color=accent, lw=1.8, shrinkA=0, shrinkB=0),
			)

	ax.text(
		0.055,
		0.07,
		"Plant interpretation: the system ingests an image or belt segment, localizes or isolates evidence, maps every retained component to hard hazard rules, and ends with an operator-facing disposal action.",
		color="#d5e2ee",
		fontsize=10,
	)
	return fig


def build_workflow_operations_figure(summary: Mapping, *, pdf_mode: bool = False) -> plt.Figure:
	fig = plt.figure(figsize=(11.69, 8.27) if pdf_mode else (13.2, 7.8), facecolor="#071018")
	gs = fig.add_gridspec(2, 2, hspace=0.22, wspace=0.18)
	preview_image = summary.get("preview_image")
	component_rows = list(summary.get("component_rows", []))
	frame_summaries = list(summary.get("frame_summaries", []))
	prediction = summary.get("prediction", {}) if isinstance(summary.get("prediction"), Mapping) else {}
	decision = summary.get("decision", {}) if isinstance(summary.get("decision"), Mapping) else {}
	hazard_counts = dict(summary.get("hazard_counts", {})) if isinstance(summary.get("hazard_counts"), Mapping) else {}
	objects_per_frame = to_float(summary.get("objects_per_frame"))
	average_detector_score = to_float(summary.get("average_detector_score"))

	ax_image = fig.add_subplot(gs[0, 0])
	if isinstance(preview_image, Image.Image):
		ax_image.imshow(np.asarray(preview_image.convert("RGB")))
		ax_image.set_xticks([])
		ax_image.set_yticks([])
		style_plot_axis(ax_image, "Observed evidence")
	else:
		ax_image.set_axis_off()
		ax_image.set_facecolor("#102131")
		ax_image.text(0.5, 0.55, "No preview image\navailable for this run", ha="center", va="center", color="#dce7f2", fontsize=14, fontweight="bold")

	ax_text = fig.add_subplot(gs[0, 1])
	ax_text.set_axis_off()
	ax_text.set_facecolor("#102131")
	ax_text.text(0.0, 1.0, "Operations Snapshot", color="#f7fbff", fontsize=16, fontweight="bold", va="top")
	snapshot_lines = [
		f"Workflow: {summary.get('workflow_name', 'n/a')} | source: {summary.get('source_label', 'n/a')}",
		f"Retained objects / events: {summary.get('object_count_value', 0)} | unique components: {summary.get('unique_components', 0)}",
		f"Dominant class: {summary.get('dominant_component', 'n/a')} at {format_pct(to_float(summary.get('dominant_confidence')))}",
		f"Average classifier confidence: {format_pct(to_float(summary.get('average_confidence')))}"
		+ (f" | detector score: {format_pct(average_detector_score)}" if average_detector_score is not None else ""),
		f"Automation clear rate: {format_pct(to_float(summary.get('automation_clear_rate')))} | review burden: {summary.get('review_object_count', 0)} item(s)",
		f"Highest hazard: {summary.get('highest_hazard', 'UNKNOWN')} | route patterns: {summary.get('route_count', 0)}"
		+ (f" | objects/frame: {format_float(objects_per_frame, 2)}" if objects_per_frame is not None else ""),
	]
	ax_text.text(0.0, 0.86, "\n".join(snapshot_lines), color="#d5e2ee", fontsize=10.5, va="top", linespacing=1.35)
	ax_text.text(0.0, 0.34, "Decision support", color="#9ce7ed", fontsize=12, fontweight="bold", va="top")
	ax_text.text(
		0.0,
		0.29,
		"\n".join(wrap_plot_text(decision.get("short_recommendation", "No route available."), width=48, max_lines=4)),
		color="#f7fbff",
		fontsize=13,
		fontweight="bold",
		va="top",
	)
	ax_text.text(
		0.0,
		0.12,
		"\n".join(wrap_plot_text(summary.get("detail", ""), width=58, max_lines=5)),
		color="#d5e2ee",
		fontsize=9.6,
		va="top",
	)

	ax_hazard = fig.add_subplot(gs[1, 0])
	hazard_labels = ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]
	hazard_values = [int(hazard_counts.get(label, 0) or 0) for label in hazard_labels]
	hazard_colors = [HAZARD_COLOR.get(label, "#94a3b8") for label in hazard_labels]
	ax_hazard.bar(hazard_labels, hazard_values, color=hazard_colors, width=0.58)
	style_plot_axis(ax_hazard, "Hazard footprint")
	ax_hazard.set_ylabel("objects / events")
	ax_hazard.grid(axis="y", alpha=0.18, color="#6f879e")

	ax_secondary = fig.add_subplot(gs[1, 1])
	if summary.get("analysis_kind") == "video_belt_review" and frame_summaries:
		timestamps = [float(item.get("timestamp_s", 0.0)) for item in frame_summaries]
		counts = [int(item.get("detected_objects", 0) or 0) for item in frame_summaries]
		review_flags = [1 if item.get("requires_review") else 0 for item in frame_summaries]
		ax_secondary.plot(timestamps, counts, color="#63d0d9", linewidth=2.3, marker="o")
		ax_secondary.scatter(
			[t for t, flag in zip(timestamps, review_flags) if flag],
			[c for c, flag in zip(counts, review_flags) if flag],
			color="#ff6f61",
			s=42,
			label="review frame",
			zorder=3,
		)
		style_plot_axis(ax_secondary, "Frame-level activity")
		ax_secondary.set_xlabel("timestamp (s)")
		ax_secondary.set_ylabel("object events")
		ax_secondary.grid(alpha=0.18, color="#6f879e")
		if any(review_flags):
			ax_secondary.legend(facecolor="#102131", edgecolor="#35506b", labelcolor="#dce7f2")
	elif component_rows:
		top_components = component_rows[:6]
		ax_secondary.barh(
			list(reversed([truncate_text(row.get("component", ""), 18) for row in top_components])),
			list(reversed([int(row.get("count", 0) or 0) for row in top_components])),
			color="#63d0d9",
		)
		style_plot_axis(ax_secondary, "Component mix")
		ax_secondary.set_xlabel("count")
		ax_secondary.grid(axis="x", alpha=0.18, color="#6f879e")
	else:
		top_predictions = prediction.get("top_predictions", []) if isinstance(prediction.get("top_predictions"), list) else []
		labels = [truncate_text(item.get("class_name", ""), 18) for item in top_predictions[:5]]
		values = [float(to_float(item.get("confidence")) or 0.0) for item in top_predictions[:5]]
		ax_secondary.barh(list(reversed(labels)), list(reversed(values)), color="#63d0d9")
		style_plot_axis(ax_secondary, "Confidence spread")
		ax_secondary.set_xlabel("confidence")
		ax_secondary.set_xlim(0, 1)
		ax_secondary.grid(axis="x", alpha=0.18, color="#6f879e")

	fig.suptitle("Plant Operations View", x=0.06, y=0.985, ha="left", color="#f7fbff", fontsize=18, fontweight="bold")
	return fig


def build_component_intelligence_figure(summary: Mapping, *, pdf_mode: bool = False) -> plt.Figure:
	component_rows = list(summary.get("component_rows", []))
	route_rows = list(summary.get("route_rows", [])) or build_route_distribution_rows(component_rows)
	stream_rows = list(summary.get("stream_rows", [])) or build_operational_stream_rows(component_rows)
	confidence_rows = build_component_confidence_rows(component_rows)
	hazard_mix_rows = list(summary.get("hazard_mix_rows", []))
	process_notes = list(summary.get("process_notes", []))
	fig = plt.figure(figsize=(11.69, 8.27) if pdf_mode else (13.2, 8.6), facecolor="#071018")
	gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.18)

	ax_route = fig.add_subplot(gs[0, 0])
	if route_rows:
		route_labels = list(reversed([truncate_text(row["route"], 32) for row in route_rows[:6]]))
		route_values = list(reversed([int(row["count"]) for row in route_rows[:6]]))
		ax_route.barh(route_labels, route_values, color="#f38b59")
		style_plot_axis(ax_route, "Downstream route distribution")
		ax_route.set_xlabel("objects / events")
		ax_route.grid(axis="x", alpha=0.18, color="#6f879e")
	else:
		ax_route.set_axis_off()
		ax_route.text(0.5, 0.5, "No route distribution available", ha="center", va="center", color="#dce7f2")

	ax_hazard = fig.add_subplot(gs[0, 1])
	ax_hazard.set_facecolor("#102131")
	ax_hazard.set_title("Hazard composition", color="#f7fbff", fontsize=12, fontweight="bold", pad=10)
	nonzero_hazards = [row for row in hazard_mix_rows if int(row.get("count", 0) or 0) > 0]
	if nonzero_hazards:
		labels = [str(row["hazard"]) for row in nonzero_hazards]
		values = [int(row["count"]) for row in nonzero_hazards]
		colors = [HAZARD_COLOR.get(label, "#94a3b8") for label in labels]
		wedges, _ = ax_hazard.pie(
			values,
			colors=colors,
			startangle=90,
			counterclock=False,
			wedgeprops=dict(width=0.38, edgecolor="#071018", linewidth=1.2),
		)
		ax_hazard.text(0.0, 0.1, f"{sum(values)}", ha="center", va="center", color="#f7fbff", fontsize=22, fontweight="bold")
		ax_hazard.text(0.0, -0.12, "retained events", ha="center", va="center", color="#92a4b8", fontsize=10)
		legend = ax_hazard.legend(
			wedges,
			[f"{label} ({value})" for label, value in zip(labels, values)],
			loc="lower center",
			bbox_to_anchor=(0.5, -0.08),
			ncol=2,
			frameon=False,
			fontsize=9,
		)
		for text in legend.get_texts():
			text.set_color("#dce7f2")
	else:
		ax_hazard.text(0.5, 0.5, "No hazard mix available", ha="center", va="center", color="#dce7f2", transform=ax_hazard.transAxes)
	ax_hazard.set_aspect("equal")

	ax_conf = fig.add_subplot(gs[1, 0])
	if confidence_rows:
		labels = list(reversed([truncate_text(row["component"], 18) for row in confidence_rows[:8]]))
		values = list(reversed([float(row["avg_confidence"]) for row in confidence_rows[:8]]))
		colors = list(reversed([HAZARD_COLOR.get(str(row["hazard_level"]), "#94a3b8") for row in confidence_rows[:8]]))
		ax_conf.barh(labels, values, color=colors)
		style_plot_axis(ax_conf, "Average confidence by component")
		ax_conf.set_xlabel("confidence")
		ax_conf.set_xlim(0, 1)
		ax_conf.grid(axis="x", alpha=0.18, color="#6f879e")
	else:
		ax_conf.set_axis_off()
		ax_conf.text(0.5, 0.5, "No component confidence view available", ha="center", va="center", color="#dce7f2")

	ax_panel = fig.add_subplot(gs[1, 1])
	ax_panel.set_axis_off()
	ax_panel.set_xlim(0, 1)
	ax_panel.set_ylim(0, 1)
	ax_panel.set_facecolor("#102131")
	ax_panel.text(0.0, 0.98, "Plant-readiness intelligence", color="#f7fbff", fontsize=14, fontweight="bold", va="top")

	draw_canvas_stat_card(
		ax_panel,
		x=0.0,
		y=0.68,
		w=0.47,
		h=0.2,
		title="Automation clear",
		value=format_pct(to_float(summary.get("automation_clear_rate"))),
		detail="share of retained events not blocked for review",
		accent="#3cc58d",
	)
	draw_canvas_stat_card(
		ax_panel,
		x=0.51,
		y=0.68,
		w=0.47,
		h=0.2,
		title="Review burden",
		value=f"{summary.get('review_object_count', 0)}/{summary.get('object_count_value', 0)}",
		detail="items that still need operator confirmation",
		accent="#f4b168",
	)
	draw_canvas_stat_card(
		ax_panel,
		x=0.0,
		y=0.43,
		w=0.47,
		h=0.18,
		title="Avg confidence",
		value=format_pct(to_float(summary.get("average_confidence"))),
		detail="mean crop / image confidence across retained evidence",
		accent="#63d0d9",
	)
	draw_canvas_stat_card(
		ax_panel,
		x=0.51,
		y=0.43,
		w=0.47,
		h=0.18,
		title="Plant streams",
		value=str(len(stream_rows)),
		detail="distinct downstream handling lanes activated",
		accent="#f38b59",
	)

	ax_panel.text(0.0, 0.34, "Operational streams", color="#9ce7ed", fontsize=11, fontweight="bold", va="top")
	if stream_rows:
		stream_lines = [f"{idx}. {row['stream']} -> {row['count']} event(s)" for idx, row in enumerate(stream_rows[:4], start=1)]
		ax_panel.text(0.0, 0.305, "\n".join(stream_lines), color="#d5e2ee", fontsize=9.5, va="top", linespacing=1.45)
	else:
		ax_panel.text(0.0, 0.305, "No operational streams were generated.", color="#d5e2ee", fontsize=9.5, va="top")

	ax_panel.text(0.0, 0.145, "Process notes", color="#9ce7ed", fontsize=11, fontweight="bold", va="top")
	if process_notes:
		process_lines = [f"- {truncate_text(note, 82)}" for note in process_notes[:3]]
		ax_panel.text(0.0, 0.11, "\n".join(process_lines), color="#d5e2ee", fontsize=9.4, va="top", linespacing=1.4)
	else:
		ax_panel.text(0.0, 0.11, "No downstream process notes available.", color="#d5e2ee", fontsize=9.4, va="top")

	fig.suptitle("Component and Routing Intelligence", x=0.06, y=0.98, ha="left", color="#f7fbff", fontsize=17, fontweight="bold")
	return fig


def build_workflow_cover_figure(summary: Mapping) -> plt.Figure:
	fig = plt.figure(figsize=(11.69, 8.27), facecolor="#071018")
	ax = fig.add_axes([0, 0, 1, 1])
	ax.set_axis_off()
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)

	preview_image = summary.get("preview_image")
	ax.text(0.055, 0.94, "E-Waste Workflow Dossier", color="#f7fbff", fontsize=24, fontweight="bold")
	ax.text(
		0.055,
		0.902,
		f"{summary.get('workflow_name', 'Workflow review')} | generated {datetime.now().strftime('%d %b %Y %H:%M')}",
		color="#9ce7ed",
		fontsize=12,
	)
	ax.text(0.055, 0.85, "\n".join(wrap_plot_text(summary.get("detail", ""), width=84, max_lines=4)), color="#d5e2ee", fontsize=11, va="top")

	def draw_stat_card(x: float, title: str, value: str, detail: str, accent: str) -> None:
		card = FancyBboxPatch(
			(x, 0.56),
			0.18,
			0.15,
			boxstyle="round,pad=0.012,rounding_size=0.024",
			linewidth=1.4,
			edgecolor=accent,
			facecolor="#102131",
		)
		ax.add_patch(card)
		ax.text(x + 0.02, 0.675, title.upper(), color="#92a4b8", fontsize=8.5, fontweight="bold")
		ax.text(x + 0.02, 0.62, value, color="#f7fbff", fontsize=18, fontweight="bold")
		ax.text(x + 0.02, 0.58, detail, color="#d5e2ee", fontsize=9)

	draw_stat_card(0.055, "Workflow", str(summary.get("workflow_name", "n/a")), "Current active lane", "#63d0d9")
	draw_stat_card(0.255, "Objects", str(summary.get("object_count_value", 0)), "Retained objects / events", "#f4b168")
	draw_stat_card(0.455, "Dominant", str(summary.get("dominant_component", "n/a")), f"confidence {format_pct(to_float(summary.get('dominant_confidence')))}", "#3cc58d")
	draw_stat_card(0.655, "Hazard", str(summary.get("highest_hazard", "UNKNOWN")), "Worst-case routing band", "#ff6f61")

	if isinstance(preview_image, Image.Image):
		image_ax = fig.add_axes([0.72, 0.16, 0.23, 0.26])
		image_ax.imshow(np.asarray(preview_image.convert("RGB")))
		image_ax.set_xticks([])
		image_ax.set_yticks([])
		image_ax.set_title("Representative evidence", color="#f7fbff", fontsize=11)
		for spine in image_ax.spines.values():
			spine.set_edgecolor("#63d0d9")
			spine.set_linewidth(1.2)

	ax.text(0.055, 0.47, "Decision narrative", color="#f7fbff", fontsize=15, fontweight="bold")
	ax.text(
		0.055,
		0.435,
		"\n".join(wrap_plot_text(summary.get("decision", {}).get("explanation", "No decision explanation available."), width=78, max_lines=8)),
		color="#d5e2ee",
		fontsize=10,
		va="top",
	)
	ax.text(0.055, 0.16, "Plant takeaway", color="#9ce7ed", fontsize=12, fontweight="bold")
	ax.text(
		0.055,
		0.132,
		"This report translates raw model output into plant-facing evidence: what was observed, how many objects were retained, which hazard bands were triggered, what material stream each component belongs to, and what downstream action is recommended.",
		color="#d5e2ee",
		fontsize=10.2,
		va="top",
	)
	return fig


def build_workflow_table_figure(summary: Mapping, *, kind: str = "components") -> plt.Figure:
	component_rows = list(summary.get("component_rows", []))
	route_rows = list(summary.get("route_rows", []))
	stream_rows = list(summary.get("stream_rows", []))
	objects = list(summary.get("objects", []))
	frame_summaries = list(summary.get("frame_summaries", []))
	fig = plt.figure(figsize=(11.69, 8.27), facecolor="#071018")
	ax = fig.add_axes([0, 0, 1, 1])
	ax.set_axis_off()
	ax.set_xlim(0, 1)
	ax.set_ylim(0, 1)

	if kind == "components":
		ax.text(0.05, 0.94, "Materials and Disposal Matrix", color="#f7fbff", fontsize=22, fontweight="bold")
		draw_canvas_stat_card(
			ax,
			x=0.05,
			y=0.79,
			w=0.19,
			h=0.1,
			title="Component groups",
			value=str(summary.get("unique_components", 0)),
			detail="distinct classified groups in this run",
			accent="#63d0d9",
		)
		draw_canvas_stat_card(
			ax,
			x=0.27,
			y=0.79,
			w=0.19,
			h=0.1,
			title="Route patterns",
			value=str(summary.get("route_count", 0)),
			detail="downstream streams activated",
			accent="#f38b59",
		)
		draw_canvas_stat_card(
			ax,
			x=0.49,
			y=0.79,
			w=0.19,
			h=0.1,
			title="Highest hazard",
			value=str(summary.get("highest_hazard", "UNKNOWN")),
			detail="worst-case disposal control band",
			accent="#ff6f61" if str(summary.get("highest_hazard", "UNKNOWN")) == "HIGH" else "#f4b168",
		)
		draw_canvas_stat_card(
			ax,
			x=0.71,
			y=0.79,
			w=0.24,
			h=0.1,
			title="SDG lens",
			value=", ".join(summary.get("sdg_targets", [])[:2]) or "SDG 12.4",
			detail="safe handling, recovery, and auditability",
			accent="#3cc58d",
		)
		component_display = []
		for row in component_rows[:10]:
			component_display.append(
				[
					row.get("component", ""),
					int(row.get("count", 0) or 0),
					row.get("hazard_level", ""),
					f"{to_float(row.get('avg_confidence')) or 0.0:.2%}",
					truncate_text(row.get("material_profile", ""), 30),
					truncate_text(row.get("disposal_pathway", ""), 30),
					truncate_text(row.get("process_further", ""), 34),
				]
			)
		table_ax = fig.add_axes([0.04, 0.39, 0.92, 0.33])
		table_ax.set_axis_off()
		if component_display:
			table = table_ax.table(
				cellText=component_display,
				colLabels=["Component", "Count", "Hazard", "Avg conf", "Material", "Route", "Next process"],
				loc="center",
				cellLoc="left",
				colLoc="left",
				colWidths=[0.14, 0.06, 0.09, 0.08, 0.21, 0.19, 0.23],
			)
			table.auto_set_font_size(False)
			table.set_fontsize(8.2)
			table.scale(1, 1.45)
			for (row_idx, col_idx), cell in table.get_celld().items():
				cell.set_edgecolor("#35506b")
				cell.set_linewidth(0.6)
				if row_idx == 0:
					cell.set_facecolor("#13283d")
					cell.get_text().set_color("#f7fbff")
					cell.get_text().set_fontweight("bold")
				else:
					cell.set_facecolor("#102131" if row_idx % 2 else "#112637")
					cell.get_text().set_color("#dce7f2")

		stream_ax = fig.add_axes([0.05, 0.09, 0.39, 0.2])
		if stream_rows:
			stream_labels = list(reversed([truncate_text(row["stream"], 24) for row in stream_rows[:5]]))
			stream_values = list(reversed([int(row["count"]) for row in stream_rows[:5]]))
			stream_ax.barh(stream_labels, stream_values, color="#63d0d9")
			style_plot_axis(stream_ax, "Activated plant streams")
			stream_ax.set_xlabel("objects / events")
			stream_ax.grid(axis="x", alpha=0.18, color="#6f879e")
		else:
			stream_ax.set_axis_off()
			stream_ax.text(0.5, 0.5, "No downstream streams available", ha="center", va="center", color="#dce7f2")

		notes_ax = fig.add_axes([0.5, 0.08, 0.45, 0.21])
		notes_ax.set_axis_off()
		notes_ax.set_facecolor("#102131")
		notes_ax.text(0.0, 0.96, "Downstream processing guidance", color="#f7fbff", fontsize=15, fontweight="bold", va="top")
		note_lines: list[str] = []
		for idx, row in enumerate(component_rows[:4], start=1):
			note_text = " ".join(wrap_plot_text(row.get("process_further", "n/a"), width=58, max_lines=2))
			note_lines.append(f"{idx}. {row.get('component', 'n/a')} -> {note_text}")
		if not note_lines:
			note_lines = ["No component-specific process notes available."]
		notes_ax.text(0.0, 0.8, "\n".join(note_lines), color="#d5e2ee", fontsize=9.6, va="top", linespacing=1.48)
	else:
		ax.text(0.05, 0.94, "Object and Frame Evidence", color="#f7fbff", fontsize=22, fontweight="bold")
		preview_image = summary.get("preview_image")
		draw_canvas_stat_card(
			ax,
			x=0.05,
			y=0.8,
			w=0.2,
			h=0.1,
			title="Detected events",
			value=str(summary.get("object_count_value", 0)),
			detail="retained objects or frame events",
			accent="#63d0d9",
		)
		draw_canvas_stat_card(
			ax,
			x=0.28,
			y=0.8,
			w=0.2,
			h=0.1,
			title="Review burden",
			value=str(summary.get("review_object_count", 0)),
			detail="items awaiting human confirmation",
			accent="#f4b168",
		)
		draw_canvas_stat_card(
			ax,
			x=0.51,
			y=0.8,
			w=0.2,
			h=0.1,
			title="Avg confidence",
			value=format_pct(to_float(summary.get("average_confidence"))),
			detail="mean classifier confidence",
			accent="#3cc58d",
		)
		if isinstance(preview_image, Image.Image):
			image_ax = fig.add_axes([0.76, 0.73, 0.19, 0.16])
			image_ax.imshow(np.asarray(preview_image.convert("RGB")))
			image_ax.set_xticks([])
			image_ax.set_yticks([])
			image_ax.set_title("Representative evidence", color="#f7fbff", fontsize=10)
			for spine in image_ax.spines.values():
				spine.set_edgecolor("#63d0d9")
				spine.set_linewidth(1.0)
		object_display = []
		for row in objects[:16]:
			object_display.append(
				[
					row.get("object_id", ""),
					truncate_text(row.get("predicted_class", ""), 16),
					f"{to_float(row.get('classifier_confidence')) or 0.0:.2%}",
					row.get("hazard_level", ""),
					"Yes" if row.get("requires_review", True) else "No",
					truncate_text(row.get("detector_label", ""), 18),
					truncate_text(row.get("frame_id", "") or "-", 10),
				]
			)
		table_ax = fig.add_axes([0.04, 0.42, 0.92, 0.26])
		table_ax.set_axis_off()
		if object_display:
			table = table_ax.table(
				cellText=object_display,
				colLabels=["ID", "Class", "Confidence", "Hazard", "Review", "Localization source", "Frame"],
				loc="center",
				cellLoc="left",
				colLoc="left",
				colWidths=[0.08, 0.16, 0.11, 0.1, 0.09, 0.28, 0.1],
			)
			table.auto_set_font_size(False)
			table.set_fontsize(8.4)
			table.scale(1, 1.4)
			for (row_idx, col_idx), cell in table.get_celld().items():
				cell.set_edgecolor("#35506b")
				cell.set_linewidth(0.6)
				if row_idx == 0:
					cell.set_facecolor("#13283d")
					cell.get_text().set_color("#f7fbff")
					cell.get_text().set_fontweight("bold")
				else:
					cell.set_facecolor("#102131" if row_idx % 2 else "#112637")
					cell.get_text().set_color("#dce7f2")

		route_ax = fig.add_axes([0.05, 0.09, 0.4, 0.22])
		if route_rows:
			route_labels = list(reversed([truncate_text(row["route"], 24) for row in route_rows[:5]]))
			route_values = list(reversed([int(row["count"]) for row in route_rows[:5]]))
			route_ax.barh(route_labels, route_values, color="#f38b59")
			style_plot_axis(route_ax, "Route allocation")
			route_ax.set_xlabel("objects / events")
			route_ax.grid(axis="x", alpha=0.18, color="#6f879e")
		else:
			route_ax.set_axis_off()
			route_ax.text(0.5, 0.5, "No route evidence available", ha="center", va="center", color="#dce7f2")

		if frame_summaries:
			frame_display = []
			for row in frame_summaries[:10]:
				frame_display.append(
					[
						row.get("frame_id", ""),
						f"{to_float(row.get('timestamp_s')) or 0.0:.2f}s",
						int(row.get("detected_objects", 0) or 0),
						truncate_text(row.get("dominant_component", ""), 18),
						row.get("highest_hazard", ""),
						"Yes" if row.get("requires_review", True) else "No",
					]
				)
			frame_ax = fig.add_axes([0.51, 0.08, 0.45, 0.24])
			frame_ax.set_axis_off()
			frame_table = frame_ax.table(
				cellText=frame_display,
				colLabels=["Frame", "Timestamp", "Objects", "Dominant class", "Hazard", "Review"],
				loc="center",
				cellLoc="left",
				colLoc="left",
				colWidths=[0.13, 0.15, 0.12, 0.26, 0.12, 0.1],
			)
			frame_table.auto_set_font_size(False)
			frame_table.set_fontsize(8.7)
			frame_table.scale(1, 1.35)
			for (row_idx, col_idx), cell in frame_table.get_celld().items():
				cell.set_edgecolor("#35506b")
				cell.set_linewidth(0.6)
				if row_idx == 0:
					cell.set_facecolor("#13283d")
					cell.get_text().set_color("#f7fbff")
					cell.get_text().set_fontweight("bold")
				else:
					cell.set_facecolor("#102131" if row_idx % 2 else "#112637")
					cell.get_text().set_color("#dce7f2")
		else:
			notes_ax = fig.add_axes([0.51, 0.09, 0.43, 0.21])
			notes_ax.set_axis_off()
			notes_ax.text(0.0, 0.96, "Localization interpretation", color="#f7fbff", fontsize=14, fontweight="bold", va="top")
			notes_ax.text(
				0.0,
				0.8,
				"\n".join(
					wrap_plot_text(
						"The table above lists every retained object candidate, its class, hazard band, and downstream route. This page is intended to show plant operators exactly what evidence contributed to the final disposal recommendation.",
						width=62,
						max_lines=8,
					)
				),
				color="#d5e2ee",
				fontsize=9.8,
				va="top",
				linespacing=1.45,
			)

	return fig


def build_workflow_pdf_report(summary: Mapping) -> bytes:
	buffer = BytesIO()
	with PdfPages(buffer) as pdf:
		pdf_info = pdf.infodict()
		pdf_info["Title"] = f"E-Waste Workflow Dossier - {summary.get('workflow_name', 'Workflow review')}"
		pdf_info["Author"] = "E-Waste Operations Console"
		pdf_info["Subject"] = "Plant-facing workflow report for e-waste triage, clustering, and belt review"
		pdf_info["Keywords"] = "e-waste, workflow, disposal, SDG 12, convnext, detection, routing"
		pdf_info["CreationDate"] = datetime.now()
		for fig in [
			build_workflow_cover_figure(summary),
			build_workflow_diagram_figure(summary, pdf_mode=True),
			build_workflow_operations_figure(summary, pdf_mode=True),
			build_component_intelligence_figure(summary, pdf_mode=True),
			build_workflow_table_figure(summary, kind="components"),
			build_workflow_table_figure(summary, kind="objects"),
		]:
			pdf.savefig(fig, facecolor=fig.get_facecolor())
			plt.close(fig)
	buffer.seek(0)
	return buffer.getvalue()


def render_workflow_workspace(last_result: Mapping | None) -> None:
	render_section_intro(
		"Workflow Atlas",
		"Unified operational infographic",
		"This page packages the current run into one faculty-friendly operations narrative: what entered the plant lane, how the system processed it, what it extracted, how hazard and material rules were applied, and what downstream action is recommended.",
		icon="stack",
	)
	summary = build_unified_workflow_summary(last_result, st.session_state)
	if summary is None:
		render_panel(
			"Workflow status",
			"Awaiting a run",
			"Run single-item triage, cluster review, or video review first. This workspace will then generate a structured operations infographic and an exportable dossier.",
			icon="route",
		)
		render_banner(
			"What this page is for",
			"It is meant to impress a reviewer quickly: one page, one story, one export, with plant-facing visuals instead of scattered UI fragments.",
			"neutral",
			icon="chart",
		)
		return

	render_banner(
		str(summary.get("headline", "Workflow summary")),
		str(summary.get("detail", "No diagnostic detail available.")),
		"warning" if summary.get("review_required", True) else "success",
		icon="route",
	)
	render_badge_row(summary.get("media_badges", []))

	top_metrics = st.columns(4, gap="small")
	with top_metrics[0]:
		render_metric_tile("Workflow", str(summary.get("workflow_name", "n/a")), str(summary.get("source_label", "active session")), "neutral", icon="gallery")
	with top_metrics[1]:
		render_metric_tile("Objects / events", str(summary.get("object_count_value", 0)), "what the system retained and processed in this run", "neutral", icon="cluster")
	with top_metrics[2]:
		render_metric_tile("Unique components", str(summary.get("unique_components", 0)), "distinct e-waste groups extracted from the run", "neutral", icon="stack")
	with top_metrics[3]:
		render_metric_tile(
			"Highest hazard",
			str(summary.get("highest_hazard", "UNKNOWN")),
			"highest risk band controlling downstream routing",
			"danger" if summary.get("highest_hazard") == "HIGH" else "warning",
			icon="warning",
		)

	second_metrics = st.columns(4, gap="small")
	with second_metrics[0]:
		render_metric_tile("Dominant class", str(summary.get("dominant_component", "n/a")), f"confidence {format_pct(to_float(summary.get('dominant_confidence')))}", "success", icon="cpu")
	with second_metrics[1]:
		render_metric_tile("High-hazard count", str(summary.get("high_hazard_count", 0)), "items or events requiring hazardous handling", "danger", icon="shield")
	with second_metrics[2]:
		render_metric_tile("Route patterns", str(summary.get("route_count", 0)), "distinct downstream streams generated", "neutral", icon="route")
	with second_metrics[3]:
		render_metric_tile(
			"Human review",
			"Required" if summary.get("review_required", True) else "Clear",
			"final plant release gate for the observed run",
			"warning" if summary.get("review_required", True) else "success",
			icon="review",
		)

	st.markdown("#### Executive Dossier View")
	cover_fig = build_workflow_cover_figure(summary)
	st.pyplot(cover_fig, use_container_width=True)
	plt.close(cover_fig)

	st.markdown("#### End-to-End Workflow Blueprint")
	workflow_fig = build_workflow_diagram_figure(summary)
	st.pyplot(workflow_fig, use_container_width=True)
	plt.close(workflow_fig)

	st.markdown("#### Plant Operations Snapshot")
	ops_fig = build_workflow_operations_figure(summary)
	st.pyplot(ops_fig, use_container_width=True)
	plt.close(ops_fig)

	st.markdown("#### Component and Routing Intelligence")
	component_intel_fig = build_component_intelligence_figure(summary)
	st.pyplot(component_intel_fig, use_container_width=True)
	plt.close(component_intel_fig)

	summary_cols = st.columns(2, gap="large")
	with summary_cols[0]:
		render_panel(
			"Plant response summary",
			str(summary.get("decision", {}).get("short_recommendation", "No route available.")),
			str(summary.get("decision", {}).get("explanation", "No decision explanation available.")),
			icon="route",
		)
	with summary_cols[1]:
		render_panel(
			"SDG 12 alignment",
			", ".join(summary.get("sdg_targets", [])[:2]) or "SDG 12.4",
			str(summary.get("sdg_alignment_copy", "No SDG alignment narrative available.")),
			icon="spark",
		)

	st.markdown("#### Operator Checklist")
	render_checklist(summary.get("checklist", []))

	st.markdown("#### Component-to-Disposal Matrix")
	component_frame = summary.get("component_frame")
	if isinstance(component_frame, pd.DataFrame) and not component_frame.empty:
		st.dataframe(
			component_frame[
				[
					"component",
					"count",
					"hazard_level",
					"avg_confidence",
					"material_profile",
					"disposal_pathway",
					"process_further",
					"review_required",
					"sdg_target",
				]
			],
			width="stretch",
			hide_index=True,
		)
		render_mapping_cards(summary.get("component_rows", []))
	else:
		st.info("No component mapping matrix is available for the current workflow.")

	st.markdown("#### Object / Event Evidence")
	object_frame = summary.get("object_frame")
	if isinstance(object_frame, pd.DataFrame) and not object_frame.empty:
		display_columns = [
			"object_id",
			"predicted_class",
			"classifier_confidence",
			"detector_label",
			"detector_score",
			"hazard_level",
			"requires_review",
			"recommended_route",
			"process_further",
		]
		if "frame_id" in object_frame.columns:
			display_columns.append("frame_id")
		if "timestamp_s" in object_frame.columns:
			display_columns.append("timestamp_s")
		st.dataframe(object_frame[display_columns], width="stretch", hide_index=True)
	else:
		st.info("No object-level rows are available for the current workflow.")

	if summary.get("analysis_kind") == "video_belt_review" and summary.get("frame_summaries"):
		st.markdown("#### Frame Summary")
		frame_summary_frame = pd.DataFrame(summary["frame_summaries"])
		if not frame_summary_frame.empty:
			frame_summary_frame["timestamp_s"] = frame_summary_frame["timestamp_s"].map(lambda x: f"{float(x):.2f}")
			frame_summary_frame["requires_review"] = frame_summary_frame["requires_review"].map(lambda x: "Yes" if x else "No")
			st.dataframe(frame_summary_frame, width="stretch", hide_index=True)

	trace_steps = summary.get("trace_steps", [])
	if trace_steps:
		st.markdown("#### Execution Trace")
		render_trace_steps(trace_steps)

	st.markdown("#### Export Workflow Dossier")
	export_panels = st.columns(2, gap="small")
	with export_panels[0]:
		render_panel(
			"Report structure",
			"Multi-page plant dossier",
			"The PDF export now bundles the executive view, workflow blueprint, plant operations snapshot, hazard and routing charts, materials matrix, and evidence tables into one report.",
			icon="chart",
		)
	with export_panels[1]:
		render_panel(
			"Reviewer intent",
			str(summary.get("workflow_name", "Workflow review")),
			"Designed so a faculty evaluator or plant operator can understand the full workflow objective, what the system extracted, and what downstream handling decision was produced.",
			icon="review",
		)

	pdf_bytes = build_workflow_pdf_report(summary)
	export_cols = st.columns(2, gap="small")
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	file_stub = f"ewaste_workflow_{summary.get('analysis_kind', 'report')}_{timestamp}"
	with export_cols[0]:
		st.download_button(
			"Download workflow PDF report",
			data=pdf_bytes,
			file_name=f"{file_stub}.pdf",
			mime="application/pdf",
			key=f"{file_stub}_pdf",
			type="primary",
		)
	with export_cols[1]:
		st.download_button(
			"Download workflow JSON summary",
			data=json.dumps(summary.get("export_payload", {}), indent=2),
			file_name=f"{file_stub}.json",
			mime="application/json",
			key=f"{file_stub}_json",
		)


def render_detected_cluster_report(report: Mapping, key_prefix: str, workflow_value: str = "Detector-assisted cluster review") -> None:
	if not isinstance(report, Mapping):
		st.info("Run detector-assisted cluster review to populate this report.")
		return

	overlay = report.get("overlay_image")
	decision = report.get("decision", {}) if isinstance(report.get("decision"), Mapping) else {}
	diagnostics = report.get("diagnostics", {}) if isinstance(report.get("diagnostics"), Mapping) else {}
	component_frame = pd.DataFrame(report.get("components", []))
	object_frame = build_detected_objects_frame(report.get("objects", []))
	hazard_counts = report.get("hazard_counts", {}) if isinstance(report.get("hazard_counts"), Mapping) else {}

	top_left, top_right = st.columns([1.08, 0.92], gap="large")
	with top_left:
		if isinstance(overlay, Image.Image):
			st.image(overlay, width="stretch")
		else:
			st.info("Overlay image unavailable.")
	with top_right:
		render_panel(
			"Workflow",
			workflow_value,
			f"{report.get('detector_name', 'detector')} | scope: {report.get('detection_scope', 'n/a')}",
			icon="cluster",
		)
		r1, r2 = st.columns(2, gap="small")
		with r1:
			render_metric_tile(
				"Detected objects",
				str(report.get("object_count", 0)),
				"localized proposals retained for crop classification",
				"neutral",
				icon="gallery",
			)
		with r2:
			render_metric_tile(
				"Unique components",
				str(len(report.get("components", []))),
				"grouped after per-crop classification",
				"neutral",
				icon="stack",
			)
		r3, r4 = st.columns(2, gap="small")
		with r3:
			render_metric_tile(
				"Highest hazard",
				str(decision.get("hazard_level", "UNKNOWN")),
				"worst-case band present in this cluster",
				"danger" if decision.get("hazard_level") == "HIGH" else "warning",
				icon="warning",
			)
		with r4:
			render_metric_tile(
				"Human review",
				"Required" if decision.get("requires_human_review", True) else "Clear",
				f"threshold {decision.get('confidence_threshold', 0.0):.0%}",
				"warning" if decision.get("requires_human_review", True) else "success",
				icon="review",
			)
		render_banner(
			str(diagnostics.get("headline", "Cluster summary")),
			str(diagnostics.get("detail", "No diagnostics available.")),
			str(diagnostics.get("tone", "neutral")),
			icon="route",
		)
		render_badge_row(
			[
				f"high hazard {hazard_counts.get('HIGH', 0)}",
				f"medium hazard {hazard_counts.get('MEDIUM', 0)}",
				f"low hazard {hazard_counts.get('LOW', 0)}",
				f"fallback {'yes' if report.get('fallback_to_all') else 'no'}",
			]
		)

	st.markdown("#### Cluster Routing Summary")
	st.write(decision.get("explanation", "No routing summary available."))
	render_checklist(build_cluster_operational_checklist(report))

	if not component_frame.empty:
		display_components = component_frame.copy()
		for column in ("avg_classifier_confidence", "avg_detector_score", "peak_classifier_confidence"):
			if column in display_components.columns:
				display_components[column] = display_components[column].map(lambda x: f"{float(x):.2%}")
		st.markdown("#### Component Summary")
		st.dataframe(display_components, width="stretch", hide_index=True)

	if not object_frame.empty:
		display_objects = object_frame.copy()
		for column in ("classifier_confidence", "detector_score"):
			if column in display_objects.columns:
				display_objects[column] = display_objects[column].map(lambda x: f"{float(x):.2%}")
		if "timestamp_s" in display_objects.columns:
			display_objects["timestamp_s"] = display_objects["timestamp_s"].map(lambda x: f"{float(x):.2f}")
		st.markdown("#### Detected Objects")
		st.dataframe(display_objects, width="stretch", hide_index=True)

		dl1, dl2 = st.columns(2, gap="small")
		with dl1:
			st.download_button(
				"Download cluster report JSON",
				data=json.dumps(build_exportable_detection_report(report), indent=2),
				file_name=f"{key_prefix}_cluster_report.json",
				mime="application/json",
				key=f"{key_prefix}_json_download",
			)
		with dl2:
			st.download_button(
				"Download object table CSV",
				data=object_frame.to_csv(index=False),
				file_name=f"{key_prefix}_objects.csv",
				mime="text/csv",
				key=f"{key_prefix}_csv_download",
			)

	if report.get("objects"):
		st.markdown("#### Object Crop Gallery")
		gallery_columns = st.columns(3, gap="small")
		for idx, obj in enumerate(report["objects"]):
			caption = (
				f"{obj['object_id']} | {obj['prediction']['class_name']} | "
				f"{obj['prediction']['confidence']:.1%} | {obj['hazard_level']}"
			)
			with gallery_columns[idx % 3]:
				st.image(obj["crop"], caption=caption, width="stretch")


def render_video_belt_report(report: Mapping, video_name: str | None = None) -> None:
	if not isinstance(report, Mapping):
		st.info("Run video belt review to generate frame-level reports.")
		return

	video_meta = report.get("video_meta", {}) if isinstance(report.get("video_meta"), Mapping) else {}
	frame_summaries = pd.DataFrame(report.get("frame_summaries", []))
	frame_reports = report.get("frame_reports", []) if isinstance(report.get("frame_reports"), list) else []

	render_banner(
		"Video-mode caveat",
		"This workflow samples frames, localizes objects with a pretrained detector, classifies each crop, and aggregates the results. It does not yet track the same physical item across consecutive frames.",
		"warning",
		icon="warning",
	)
	render_detected_cluster_report(
		report,
		key_prefix="video_belt",
		workflow_value=(
			f"Sampled video belt review | {video_name or 'uploaded video'} | "
			f"every {format_float(to_float(video_meta.get('sample_every_seconds')), 2)} s"
		),
	)

	if not frame_summaries.empty:
		display_frames = frame_summaries.copy()
		display_frames["timestamp_s"] = display_frames["timestamp_s"].map(lambda x: f"{float(x):.2f}")
		st.markdown("#### Frame Summary")
		st.dataframe(display_frames, width="stretch", hide_index=True)

	if frame_reports:
		st.markdown("#### Sampled Frame Gallery")
		frame_columns = st.columns(2, gap="large")
		for idx, frame in enumerate(frame_reports):
			caption = (
				f"{frame['frame_id']} | t={float(frame['timestamp_s']):.2f}s | "
				f"objects={frame['report'].get('object_count', 0)}"
			)
			with frame_columns[idx % 2]:
				st.image(frame.get("overlay_image"), caption=caption, width="stretch")


def render_single_item_operations(
	*,
	model: nn.Module,
	class_names: list[str],
	device: torch.device,
	confidence_threshold: float,
	enable_scene_scan: bool,
	scene_grid: int,
	scene_tile_floor: float,
	data_dir: Path,
) -> None:
	render_section_intro(
		"Inference",
		"Operational inference review",
		"Submit an image, inspect confidence distribution, and review composite-scene cues before any downstream routing decision.",
		icon="gallery",
	)
	render_badge_row(["Intake", "Single-label inference", "Confidence gating", "Composite-scene review"])

	left, right = st.columns([1.06, 0.94], gap="large")
	with left:
		uploaded = st.file_uploader(
			"Image intake",
			type=["jpg", "jpeg", "png", "webp"],
			accept_multiple_files=False,
			help="Best performance comes from a single dominant component in frame.",
			key="single_item_upload",
		)
		upload_token = f"{uploaded.name}:{uploaded.size}" if uploaded is not None else None
		a1, a2, a3 = st.columns(3)
		with a1:
			sample_btn = st.button("Load test image", key="single_load_sample")
		with a2:
			clear_btn = st.button("Clear", key="single_clear")
		with a3:
			run_btn = st.button("Run inference", type="primary", key="single_run")

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
			render_panel("Intake queue", "No active image", "Load a test image or upload a sample to start an operator review cycle.", icon="gallery")

	image = st.session_state.get("pending_image")
	if image is not None and run_btn:
		start = time.time()
		prediction = infer_image(model=model, image=image, class_names=class_names, device=device)
		elapsed_ms = int((time.time() - start) * 1000)
		diagnostics = analyze_prediction(prediction, confidence_threshold)
		decision = get_agent_decision(prediction["class_name"], prediction["confidence"], confidence_threshold)
		scene_analysis = None
		if enable_scene_scan:
			scene_analysis = analyze_scene_tiles(
				model=model,
				image=image,
				class_names=class_names,
				device=device,
				grid_size=scene_grid,
				min_confidence=scene_tile_floor,
			)
		st.session_state["last_result"] = {
			"analysis_kind": "single_item_review",
			"prediction": prediction,
			"decision": decision,
			"diagnostics": diagnostics,
			"scene_analysis": scene_analysis,
			"elapsed_ms": elapsed_ms,
		}

	with right:
		result = st.session_state.get("last_result")
		if not result or result.get("analysis_kind") != "single_item_review":
			render_panel("Inference state", "Awaiting input", "Run inference to populate confidence, hazard, and scene evidence.", icon="review")
		else:
			prediction = result["prediction"]
			decision = result["decision"]
			diagnostics = result["diagnostics"]
			r1, r2 = st.columns(2, gap="small")
			with r1:
				render_panel("Predicted class", prediction["class_name"], "single-label top prediction", icon="stack")
			with r2:
				render_panel("Confidence", format_pct(prediction["confidence"]), "measured on this input", icon="gauge")
			render_panel("Latency", f"{result['elapsed_ms']} ms", "single-image forward pass", icon="bolt")
			render_badge_row(
				[
					f"hazard band {decision.get('hazard_level', 'UNKNOWN')}",
					f"human review {'required' if decision.get('requires_human_review', True) else 'clear'}",
					f"explanation {decision.get('explanation_source', 'rule-based')}",
				]
			)
			render_banner(diagnostics["headline"], diagnostics["detail"], diagnostics["tone"], icon="review")
			st.markdown("#### Top-5 Class Scores")
			render_probability_rows(prediction["top_predictions"])

	result = st.session_state.get("last_result")
	if result and result.get("analysis_kind") == "single_item_review":
		scene_analysis = result.get("scene_analysis")
		st.markdown("### Composite Scene Review")
		if enable_scene_scan and scene_analysis:
			render_banner(
				scene_analysis["headline"],
				f"Tile scan uses a {scene_analysis['grid_size']}x{scene_analysis['grid_size']} grid and reports tiles above {scene_analysis['tile_floor']:.2%}. This is a review aid, not object detection.",
				"neutral",
				icon="cluster",
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
				h1, h2 = st.columns(2, gap="small")
				with h1:
					render_metric_tile("High-risk tiles", str(hazard_counts.get("HIGH", 0)), "tile-level hazard evidence", "danger", icon="warning")
				with h2:
					render_metric_tile("Medium-risk tiles", str(hazard_counts.get("MEDIUM", 0)), "tile-level hazard evidence", "warning", icon="shield")
				render_metric_tile("Low-risk tiles", str(hazard_counts.get("LOW", 0)), "tile-level hazard evidence", "success", icon="spark")
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


def render_cluster_image_operations(
	*,
	classifier_model: nn.Module,
	class_names: list[str],
	device: torch.device,
	confidence_threshold: float,
	data_dir: Path,
	detector_model_key: str,
	detector_score_threshold: float,
	max_objects: int,
	min_area_fraction: float,
	relevant_only: bool,
	crop_padding: float,
) -> None:
	render_section_intro(
		"Cluster Review",
		"Detector-assisted cluster intake",
		"Use a pretrained detector to localize multiple objects in one scene, classify each crop with the research model, and generate a cluster-level hazard routing report.",
		icon="cluster",
	)
	render_badge_row(["Pretrained object detector", "Per-object crop classification", "Hazard aggregation", "Cluster routing report"])

	left, right = st.columns([1.02, 0.98], gap="large")
	with left:
		uploaded = st.file_uploader(
			"Cluster image intake",
			type=["jpg", "jpeg", "png", "webp"],
			accept_multiple_files=False,
			help="Designed for mixed scenes on a conveyor or staging tray.",
			key="cluster_image_upload",
		)
		upload_token = f"{uploaded.name}:{uploaded.size}" if uploaded is not None else None
		b1, b2, b3 = st.columns(3)
		with b1:
			sample_btn = st.button("Load test image", key="cluster_sample")
		with b2:
			clear_btn = st.button("Clear", key="cluster_clear")
		with b3:
			run_btn = st.button("Run cluster review", type="primary", key="cluster_run")

		if uploaded is not None and upload_token != st.session_state.get("cluster_last_upload_token"):
			st.session_state["cluster_pending_image"] = Image.open(uploaded).convert("RGB")
			st.session_state["cluster_pending_label"] = uploaded.name
			st.session_state["cluster_last_upload_token"] = upload_token
			st.session_state.pop("last_cluster_result", None)
		elif sample_btn:
			sample_image, sample_name = pick_random_test_image(data_dir)
			st.session_state["cluster_pending_image"] = sample_image
			st.session_state["cluster_pending_label"] = sample_name
			st.session_state.pop("last_cluster_result", None)
		elif clear_btn:
			st.session_state["cluster_pending_image"] = None
			st.session_state["cluster_pending_label"] = None
			st.session_state["cluster_last_upload_token"] = None
			st.session_state.pop("last_cluster_result", None)

		cluster_image: Image.Image | None = st.session_state.get("cluster_pending_image")
		if cluster_image is not None:
			st.image(cluster_image, width="stretch")
			render_badge_row(
				[
					f"source: {st.session_state.get('cluster_pending_label') or 'uploaded image'}",
					f"canvas: {cluster_image.width} x {cluster_image.height}",
					"multi-object review lane",
				]
			)
		else:
			render_panel("Cluster queue", "No active image", "Upload a mixed scene to run detector-assisted cluster analysis.", icon="gallery")

	with right:
		render_panel(
			"Detector engine",
			"Faster R-CNN MobileNetV3",
			"Pretrained detector used only for localization. Every retained crop is then classified by the e-waste model.",
			icon="review",
		)
		r1, r2 = st.columns(2, gap="small")
		with r1:
			render_metric_tile("Detector threshold", f"{detector_score_threshold:.0%}", "minimum box score", "neutral", icon="gauge")
		with r2:
			render_metric_tile("Max retained objects", str(max_objects), "post-filter detection cap", "neutral", icon="stack")
		r3, r4 = st.columns(2, gap="small")
		with r3:
			render_metric_tile("Detector scope", "Priority labels" if relevant_only else "All proposals", "COCO filter mode", "neutral", icon="cluster")
		with r4:
			render_metric_tile("Min area", f"{min_area_fraction:.1%}", "small-box suppression", "neutral", icon="review")
		render_banner(
			"How this lane works",
			"First the detector finds candidate objects, then each crop goes through the classifier, and finally the dashboard aggregates hazard counts and routing actions for the whole cluster.",
			"neutral",
			icon="route",
		)

	cluster_image = st.session_state.get("cluster_pending_image")
	if cluster_image is not None and run_btn:
		try:
			with st.spinner("Loading pretrained detector and analyzing the cluster..."):
				detector_payload = load_detection_model(detector_model_key, device.type)
				start = time.time()
				report = analyze_detected_cluster(
					detector_payload=detector_payload,
					classifier_model=classifier_model,
					image=cluster_image,
					class_names=class_names,
					classifier_device=device,
					confidence_threshold=confidence_threshold,
					detector_score_threshold=detector_score_threshold,
					max_objects=max_objects,
					min_area_fraction=min_area_fraction,
					relevant_only=relevant_only,
					crop_padding=crop_padding,
				)
				report["elapsed_ms"] = int((time.time() - start) * 1000)
				st.session_state["last_cluster_result"] = report
				st.session_state["last_result"] = {
					"analysis_kind": "cluster_image_review",
					"prediction": report["prediction"],
					"decision": report["decision"],
					"diagnostics": report["diagnostics"],
					"cluster_report": report,
					"elapsed_ms": report["elapsed_ms"],
				}
		except Exception as exc:
			st.error(f"Detector-assisted cluster review failed: {exc}")

	report = st.session_state.get("last_cluster_result")
	if report:
		st.markdown("### Cluster Report")
		render_detected_cluster_report(report, key_prefix="cluster_image")
		render_panel("Processing time", f"{report.get('elapsed_ms', 'n/a')} ms", "localization + crop classification + aggregation", icon="bolt")
	else:
		st.info("Run cluster review to produce object localization, per-object classification, and aggregated routing output.")


def render_video_belt_operations(
	*,
	classifier_model: nn.Module,
	class_names: list[str],
	device: torch.device,
	confidence_threshold: float,
	detector_model_key: str,
	detector_score_threshold: float,
	max_objects_per_frame: int,
	min_area_fraction: float,
	relevant_only: bool,
	crop_padding: float,
	sample_every_seconds: float,
	max_frames: int,
) -> None:
	render_section_intro(
		"Video Review",
		"Sampled belt-camera review",
		"Upload conveyor footage, sample frames at a controlled interval, detect objects in each frame, classify the crops, and aggregate a report for the observed belt segment.",
		icon="gallery",
	)
	render_badge_row(["Video intake", "Frame sampling", "Detector-assisted localization", "Belt-segment summary"])

	left, right = st.columns([1.02, 0.98], gap="large")
	with left:
		uploaded_video = st.file_uploader(
			"Video intake",
			type=[ext.lstrip(".") for ext in sorted(VIDEO_EXTS)],
			accept_multiple_files=False,
			help="Recommended for short belt captures where a few sampled frames can summarize the segment.",
			key="belt_video_upload",
		)
		upload_token = f"{uploaded_video.name}:{uploaded_video.size}" if uploaded_video is not None else None
		c1, c2 = st.columns(2)
		with c1:
			clear_btn = st.button("Clear video", key="video_clear")
		with c2:
			run_btn = st.button("Run video review", type="primary", key="video_run")

		if uploaded_video is not None and upload_token != st.session_state.get("video_last_upload_token"):
			st.session_state["pending_video_bytes"] = uploaded_video.getvalue()
			st.session_state["pending_video_name"] = uploaded_video.name
			st.session_state["video_last_upload_token"] = upload_token
			st.session_state.pop("last_video_result", None)
		elif clear_btn:
			st.session_state["pending_video_bytes"] = None
			st.session_state["pending_video_name"] = None
			st.session_state["video_last_upload_token"] = None
			st.session_state.pop("last_video_result", None)

		video_bytes = st.session_state.get("pending_video_bytes")
		if video_bytes:
			st.video(video_bytes)
			render_badge_row(
				[
					f"source: {st.session_state.get('pending_video_name') or 'uploaded video'}",
					f"sample every {sample_every_seconds:.2f}s",
					f"max sampled frames {max_frames}",
				]
			)
		else:
			render_panel("Video queue", "No active video", "Upload conveyor footage to run sampled frame review.", icon="gallery")

	with right:
		render_panel(
			"Video review engine",
			"Detector + crop classifier",
			"Each sampled frame uses the pretrained detector for localization and the e-waste classifier for per-object labeling.",
			icon="review",
		)
		r1, r2 = st.columns(2, gap="small")
		with r1:
			render_metric_tile("Sample interval", f"{sample_every_seconds:.2f} s", "frame extraction cadence", "neutral", icon="bolt")
		with r2:
			render_metric_tile("Max sampled frames", str(max_frames), "review budget per clip", "neutral", icon="stack")
		r3, r4 = st.columns(2, gap="small")
		with r3:
			render_metric_tile("Detector threshold", f"{detector_score_threshold:.0%}", "minimum frame-level box score", "neutral", icon="gauge")
		with r4:
			render_metric_tile("Objects per frame", str(max_objects_per_frame), "retained detections cap", "neutral", icon="cluster")
		render_banner(
			"Current scope",
			"The video workflow summarizes sampled frames from a belt segment. It is a strong operational step forward, but it is still frame-wise analysis and not full multi-object tracking across time.",
			"warning",
			icon="warning",
		)

	video_bytes = st.session_state.get("pending_video_bytes")
	video_name = st.session_state.get("pending_video_name")
	if video_bytes and run_btn:
		try:
			with st.spinner("Sampling frames and running detector-assisted video review..."):
				video_suffix = Path(video_name or "belt_review.mp4").suffix.lower() or ".mp4"
				video_payload = sample_video_frames(
					video_bytes=video_bytes,
					file_suffix=video_suffix,
					sample_every_seconds=sample_every_seconds,
					max_frames=max_frames,
				)
				detector_payload = load_detection_model(detector_model_key, device.type)
				start = time.time()
				report = analyze_video_belt(
					detector_payload=detector_payload,
					classifier_model=classifier_model,
					video_payload=video_payload,
					class_names=class_names,
					classifier_device=device,
					confidence_threshold=confidence_threshold,
					detector_score_threshold=detector_score_threshold,
					max_objects_per_frame=max_objects_per_frame,
					min_area_fraction=min_area_fraction,
					relevant_only=relevant_only,
					crop_padding=crop_padding,
				)
				report["elapsed_ms"] = int((time.time() - start) * 1000)
				st.session_state["last_video_result"] = report
				st.session_state["last_result"] = {
					"analysis_kind": "video_belt_review",
					"prediction": report["prediction"],
					"decision": report["decision"],
					"diagnostics": report["diagnostics"],
					"video_report": report,
					"elapsed_ms": report["elapsed_ms"],
				}
		except Exception as exc:
			st.error(f"Video belt review failed: {exc}")

	report = st.session_state.get("last_video_result")
	if report:
		st.markdown("### Video Belt Report")
		render_video_belt_report(report, video_name=video_name)
		render_panel("Processing time", f"{report.get('elapsed_ms', 'n/a')} ms", "frame sampling + detection + crop classification + aggregation", icon="bolt")
	else:
		st.info("Run video review to generate sampled frame localization, per-object classification, and belt-segment reporting.")


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
	primary_row: pd.Series | None,
	primary_source: str | None,
	secondary_row: pd.Series | None,
	secondary_source: str | None,
	discrepancy_note: str | None,
) -> str:
	parts = [
		f"Best deployed architecture: {best_arch}.",
		f"Primary benchmark ({primary_source or 'n/a'}) accuracy: {format_pct(float(primary_row['accuracy'])) if primary_row is not None else 'n/a'}.",
	]
	if secondary_row is not None and secondary_source:
		parts.append(
			f"Secondary benchmark ({secondary_source}) accuracy: {format_pct(float(secondary_row['accuracy']))}."
		)
	else:
		parts.append("No secondary benchmark source is currently available for this model.")
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
	ann_cls = ann_results_18cls.get("classification_metrics") if isinstance(ann_results_18cls.get("classification_metrics"), Mapping) else {}
	hazard_acc = to_float(ann_results_18cls.get("hazard_class_accuracy"))
	if hazard_acc is None:
		hazard_acc = to_float(ann_cls.get("hazard_class_accuracy"))

	nmi = to_float(clustering_results.get("normalized_mutual_info"))
	cluster_count = clustering_results.get("n_clusters", "n/a")
	return (
		f"Hazard ANN accuracy: {format_pct(hazard_acc)}.\n"
		f"Clustering NMI: {format_float(nmi)}.\n"
		f"Cluster count: {cluster_count}.\n"
		"Write a short discussion paragraph for a research paper that explains what these supporting analyses add beyond the classifier, "
		"and mention limitations without overselling the results."
	)


def build_system_chat_context(
	*,
	active_arch: str,
	class_names: list[str],
	primary_row: pd.Series | None,
	ann_results: Mapping | None,
	ann_backprop_results: Mapping | None,
	clustering_metrics: Mapping | None,
	kmedoids_metrics: Mapping | None,
	clustering_comparison: Mapping | None,
	last_result: Mapping | None,
) -> str:
	benchmark_line = (
		f"{active_arch} accuracy {format_pct(float(primary_row['accuracy']))} and macro-F1 {format_float(float(primary_row['macro_f1']), 4)}"
		if primary_row is not None
		else f"{active_arch} benchmark snapshot unavailable"
	)
	ann_cls = ann_results.get("classification_metrics") if isinstance(ann_results, Mapping) and isinstance(ann_results.get("classification_metrics"), Mapping) else {}
	ann_acc = to_float(ann_cls.get("hazard_class_accuracy")) if ann_cls else to_float(ann_results.get("hazard_class_accuracy")) if isinstance(ann_results, Mapping) else None
	ann_r2 = to_float(ann_results.get("regression_metrics", {}).get("r2")) if isinstance(ann_results, Mapping) and isinstance(ann_results.get("regression_metrics"), Mapping) else None
	kmeans_sil = to_float(clustering_metrics.get("silhouette_score")) if isinstance(clustering_metrics, Mapping) else None
	kmedoids_sil = to_float(kmedoids_metrics.get("silhouette_score")) if isinstance(kmedoids_metrics, Mapping) else None
	comparison_keys = ", ".join(
		name.upper()
		for name in ("kmeans", "kmedoids")
		if isinstance(clustering_comparison, Mapping) and isinstance(clustering_comparison.get(name), Mapping)
	)
	if not comparison_keys:
		comparison_keys = "comparison pending"

	lines = [
		"E-Waste system scope: single-label image classification, hazard-aware routing, ANN hazard scoring, clustering analytics, research dashboard workflows, and registered disposal policies.",
		f"Active architecture: {active_arch}.",
		f"Known classes ({len(class_names)}): {', '.join(class_names[:8])}{' ...' if len(class_names) > 8 else ''}.",
		f"Benchmark: {benchmark_line}.",
		f"Hazard ANN accuracy: {format_pct(ann_acc)} | regression R2: {format_float(ann_r2)}.",
		f"K-Means silhouette: {format_float(kmeans_sil)} | K-Medoids silhouette: {format_float(kmedoids_sil)}.",
		f"Clustering comparison status: {comparison_keys}.",
	]

	if isinstance(ann_backprop_results, Mapping) and ann_backprop_results:
		summary = ann_backprop_results.get("backprop_summary") if isinstance(ann_backprop_results.get("backprop_summary"), Mapping) else {}
		lines.append(
			f"ANN backprop tracking: best epoch {summary.get('best_epoch', 'n/a')} | peak grad norm {format_float(to_float(summary.get('peak_total_grad_norm')))}."
		)

	if isinstance(last_result, Mapping) and last_result:
		prediction = last_result.get("prediction", {})
		decision = last_result.get("decision", {})
		analysis_kind = str(last_result.get("analysis_kind", "single_item_review"))
		if isinstance(prediction, Mapping) and isinstance(decision, Mapping):
			if analysis_kind == "video_belt_review":
				lines.append(
					f"Current inference context: video belt review with {decision.get('detected_objects', 0)} detected object events across {decision.get('sampled_frames', 'n/a')} sampled frames. Dominant component {prediction.get('class_name', 'n/a')} at {format_pct(to_float(prediction.get('confidence')))} and highest hazard {decision.get('hazard_level', 'UNKNOWN')}."
				)
			elif analysis_kind in {"cluster_image_review", "detector_assisted_cluster_review"}:
				lines.append(
					f"Current inference context: detector-assisted cluster review with {decision.get('detected_objects', 0)} localized objects and {decision.get('unique_components', 0)} predicted component groups. Dominant component {prediction.get('class_name', 'n/a')} at {format_pct(to_float(prediction.get('confidence')))} and highest hazard {decision.get('hazard_level', 'UNKNOWN')}."
				)
			else:
				lines.append(
					f"Current inference context: predicted {prediction.get('class_name', 'n/a')} at {format_pct(to_float(prediction.get('confidence')))} with hazard {decision.get('hazard_level', 'UNKNOWN')}."
				)

	lines.append(
		"Stay inside project scope. If the user asks about unrelated topics, politely refuse and redirect to system-related questions."
	)
	return "\n".join(lines)


def build_local_chatbot_reply(
	question: str,
	*,
	active_arch: str,
	primary_row: pd.Series | None,
	ann_results: Mapping | None,
	ann_backprop_results: Mapping | None,
	clustering_metrics: Mapping | None,
	kmedoids_metrics: Mapping | None,
	clustering_comparison: Mapping | None,
	last_result: Mapping | None,
) -> str:
	q = question.lower()

	if any(token in q for token in {"weather", "movie", "song", "sports", "news", "joke", "travel"}):
		return "I’m scoped only to this e-waste system: model behavior, hazard routing, ANN, clustering, dashboard workflow, and the latest inference context."

	if any(token in q for token in {"last result", "last inference", "prediction", "confidence", "uploaded image"}):
		if isinstance(last_result, Mapping) and last_result:
			prediction = last_result.get("prediction", {})
			decision = last_result.get("decision", {})
			analysis_kind = str(last_result.get("analysis_kind", "single_item_review"))
			if isinstance(prediction, Mapping):
				if analysis_kind == "video_belt_review":
					return (
						f"The latest workflow was a video belt review. It aggregated `{decision.get('detected_objects', 0)}` object events across "
						f"`{decision.get('sampled_frames', 'n/a')}` sampled frames. The dominant component evidence was "
						f"`{prediction.get('class_name', 'n/a')}` at {format_pct(to_float(prediction.get('confidence')))}, and the highest observed hazard was "
						f"`{decision.get('hazard_level', 'UNKNOWN')}`."
					)
				if analysis_kind in {"cluster_image_review", "detector_assisted_cluster_review"}:
					return (
						f"The latest workflow was detector-assisted cluster review. It localized `{decision.get('detected_objects', 0)}` objects "
						f"across `{decision.get('unique_components', 0)}` predicted component groups. Dominant evidence was "
						f"`{prediction.get('class_name', 'n/a')}` at {format_pct(to_float(prediction.get('confidence')))}, and the aggregated routing hazard band was "
						f"`{decision.get('hazard_level', 'UNKNOWN')}`."
					)
				return (
					f"The latest inference predicted `{prediction.get('class_name', 'n/a')}` at "
					f"{format_pct(to_float(prediction.get('confidence')))}. The policy layer mapped it to "
					f"`{decision.get('hazard_level', 'UNKNOWN')}` risk and recommends `{decision.get('short_recommendation', 'manual review')}`."
				)
		return "There is no stored inference in session yet. Run an image through the Operations view first."

	if any(token in q for token in {"best model", "benchmark", "accuracy", "f1"}):
		return (
			f"The active benchmark model is `{active_arch}`. Its current primary benchmark snapshot is "
			f"accuracy {format_pct(float(primary_row['accuracy'])) if primary_row is not None else 'n/a'} "
			f"and macro-F1 {format_float(float(primary_row['macro_f1']), 4) if primary_row is not None else 'n/a'}."
		)

	if any(token in q for token in {"ann", "hazard model", "backprop", "gradient"}):
		ann_cls = ann_results.get("classification_metrics") if isinstance(ann_results, Mapping) and isinstance(ann_results.get("classification_metrics"), Mapping) else {}
		ann_acc = to_float(ann_cls.get("hazard_class_accuracy")) if ann_cls else to_float(ann_results.get("hazard_class_accuracy")) if isinstance(ann_results, Mapping) else None
		ann_r2 = to_float(ann_results.get("regression_metrics", {}).get("r2")) if isinstance(ann_results, Mapping) and isinstance(ann_results.get("regression_metrics"), Mapping) else None
		if isinstance(ann_backprop_results, Mapping) and ann_backprop_results:
			summary = ann_backprop_results.get("backprop_summary") if isinstance(ann_backprop_results.get("backprop_summary"), Mapping) else {}
			return (
				f"The hazard ANN currently reports class accuracy {format_pct(ann_acc)} and regression R2 {format_float(ann_r2)}. "
				f"The supplementary backprop notebook tracks gradient norms, with best epoch `{summary.get('best_epoch', 'n/a')}` "
				f"and peak total gradient norm {format_float(to_float(summary.get('peak_total_grad_norm')))}."
			)
		return (
			f"The hazard ANN currently reports class accuracy {format_pct(ann_acc)} and regression R2 {format_float(ann_r2)}. "
			"Backprop diagnostics will appear after running the supplementary ANN notebook."
		)

	if any(token in q for token in {"cluster", "clustering", "kmeans", "k-means", "kmedoid", "medoid"}):
		kmeans_sil = to_float(clustering_metrics.get("silhouette_score")) if isinstance(clustering_metrics, Mapping) else None
		kmedoids_sil = to_float(kmedoids_metrics.get("silhouette_score")) if isinstance(kmedoids_metrics, Mapping) else None
		if isinstance(clustering_comparison, Mapping) and clustering_comparison:
			return (
				f"The dashboard compares K-Means and K-Medoids on the saved embedding space. "
				f"K-Means silhouette is {format_float(kmeans_sil)} and K-Medoids silhouette is {format_float(kmedoids_sil)}. "
				"Use the comparison panel in Analytics to inspect algorithm-level quality metrics and the t-SNE cluster view."
			)
		return (
			f"The current production clustering artifact is K-Means with silhouette {format_float(kmeans_sil)}. "
			"K-Medoids artifacts will become visible after running the supplementary clustering notebook."
		)

	if any(token in q for token in {"hazard", "battery", "pcb", "printer", "disposal", "route"}):
		for component in HAZARD_MAP:
			if component.lower() in q:
				return (
					f"`{component}` is mapped to `{HAZARD_MAP[component]}` risk. "
					f"Material profile: {MATERIAL_MAP[component]}. "
					f"Recommended routing: {DISPOSAL_MAP[component]}."
				)
		return "Ask about a registered component name like Battery, PCB, Mobile, Printer, or Refrigerator and I can return its hazard level and routing policy."

	return (
		"I can help with this project’s classifier workflow, hazard policy engine, ANN hazard model, "
		"K-Means/K-Medoids clustering, benchmark interpretation, dashboard usage, and the current inference result."
	)


def answer_system_copilot(
	question: str,
	*,
	context: str,
	active_arch: str,
	primary_row: pd.Series | None,
	ann_results: Mapping | None,
	ann_backprop_results: Mapping | None,
	clustering_metrics: Mapping | None,
	kmedoids_metrics: Mapping | None,
	clustering_comparison: Mapping | None,
	last_result: Mapping | None,
) -> str:
	if os.getenv("GROQ_API_KEY"):
		system_prompt = (
			"You are the E-Waste System Copilot for a research dashboard. "
			"Answer only about this specific project: classifier workflow, hazard routing, ANN hazard modeling, "
			"clustering analytics, dashboard interpretation, saved artifacts, and the user’s current inference result. "
			"If the question is outside this scope, refuse briefly and redirect to project-related topics. "
			"Be concise, factual, and avoid making up metrics that are not in the provided context.\n\n"
			f"System context:\n{context}"
		)
		return call_groq_text(prompt=question, system_prompt=system_prompt)

	return build_local_chatbot_reply(
		question,
		active_arch=active_arch,
		primary_row=primary_row,
		ann_results=ann_results,
		ann_backprop_results=ann_backprop_results,
		clustering_metrics=clustering_metrics,
		kmedoids_metrics=kmedoids_metrics,
		clustering_comparison=clustering_comparison,
		last_result=last_result,
	)


def render_system_copilot(
	*,
	input_key: str,
	active_arch: str,
	class_names: list[str],
	primary_row: pd.Series | None,
	ann_results: Mapping | None,
	ann_backprop_results: Mapping | None,
	clustering_metrics: Mapping | None,
	kmedoids_metrics: Mapping | None,
	clustering_comparison: Mapping | None,
	last_result: Mapping | None,
) -> None:
	st.caption("Ask about this system only: benchmark interpretation, hazard routing, ANN analytics, clustering comparison, or the latest inference result.")
	copilot_context = build_system_chat_context(
		active_arch=active_arch,
		class_names=class_names,
		primary_row=primary_row,
		ann_results=ann_results,
		ann_backprop_results=ann_backprop_results,
		clustering_metrics=clustering_metrics,
		kmedoids_metrics=kmedoids_metrics,
		clustering_comparison=clustering_comparison,
		last_result=last_result,
	)
	if "copilot_messages" not in st.session_state:
		st.session_state["copilot_messages"] = [
			{
				"role": "assistant",
				"content": "I can explain this project’s workflow, benchmark results, hazard policies, ANN backprop diagnostics, clustering comparison, and the latest inference result.",
			}
		]

	for message in st.session_state["copilot_messages"]:
		with st.chat_message(message["role"]):
			st.markdown(message["content"])

	prompt = st.chat_input("Ask the system copilot about this project", key=input_key)
	if prompt:
		st.session_state["copilot_messages"].append({"role": "user", "content": prompt})
		with st.chat_message("user"):
			st.markdown(prompt)
		with st.chat_message("assistant"):
			with st.spinner("Thinking..."):
				try:
					reply = answer_system_copilot(
						prompt,
						context=copilot_context,
						active_arch=active_arch,
						primary_row=primary_row,
						ann_results=ann_results,
						ann_backprop_results=ann_backprop_results,
						clustering_metrics=clustering_metrics,
						kmedoids_metrics=kmedoids_metrics,
						clustering_comparison=clustering_comparison,
						last_result=last_result,
					)
				except Exception as exc:
					reply = f"Copilot could not answer the question: {exc}"
			st.markdown(reply)
		st.session_state["copilot_messages"].append({"role": "assistant", "content": reply})


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
	models_dir = PROJECT_ROOT / "models"
	classification_dir = PROJECT_ROOT / "models" / "classification"
	data_dir = PROJECT_ROOT / "data"
	clustering_dir = PROJECT_ROOT / "models" / "clustering"

	try:
		assets = discover_dashboard_assets(str(classification_dir), str(data_dir))
	except Exception as exc:
		st.error(f"failed to load model assets: {exc}")
		st.stop()

	checkpoints = assets["checkpoints"]
	available_arches = [arch for arch in SUPPORTED_ARCHES if arch in checkpoints]
	if not available_arches:
		available_arches = sorted(checkpoints.keys())
	default_arch = "vit_b16" if "vit_b16" in available_arches else assets["best_arch"]

	with st.sidebar:
		st.markdown("### Operations Console")
		st.caption("Classification, detector-assisted cluster routing, sampled video review, benchmark records, and data registry for the current research system.")
		active_arch = st.selectbox("Active model", options=available_arches, index=available_arches.index(default_arch))
		confidence_threshold = st.slider("Human review threshold", 0.50, 0.95, 0.70, 0.01)
		enable_scene_scan = st.checkbox("Enable composite scene scan", value=True)
		scene_grid = st.selectbox("Scene scan grid", options=[2, 3], index=1)
		scene_tile_floor = st.slider("Tile evidence floor", 0.10, 0.80, 0.30, 0.05)
		with st.expander("Detector-assisted review", expanded=False):
			detector_model_key = "fasterrcnn_mobilenet_v3_large_320_fpn"
			st.caption("Pretrained detector for cluster images and sampled video frames.")
			detector_scope_label = st.selectbox(
				"Detector scope",
				options=["E-waste priority COCO labels", "All COCO proposals"],
				index=0,
			)
			detector_score_threshold = st.slider("Detector confidence", 0.20, 0.95, 0.45, 0.05)
			max_detected_objects = st.slider("Max objects / image", 1, 12, 8, 1)
			max_objects_per_frame = st.slider("Max objects / frame", 1, 12, 6, 1)
			min_box_area_percent = st.slider("Minimum box area (%)", 0.2, 20.0, 1.0, 0.2)
			crop_padding_percent = st.slider("Crop expansion (%)", 0.0, 25.0, 8.0, 1.0)
		with st.expander("Video sampling", expanded=False):
			video_sample_every = st.slider("Sample every (seconds)", 0.25, 5.0, 1.00, 0.25)
			video_max_frames = st.slider("Max sampled frames", 1, 16, 6, 1)
		workspace = st.selectbox("View", options=WORKSPACES, index=0)
		st.markdown("### Runtime")
		st.caption(f"Device: {runtime.device.type}")
		st.caption(f"GPU: {runtime.gpu_name or 'not detected'}")
		st.caption(f"VRAM: {f'{runtime.vram_gb:.2f} GB' if runtime.vram_gb is not None else 'n/a'}")
		st.info("Single-item triage remains the benchmarked classifier lane. Cluster and video modes use a pretrained detector for localization and then classify each retained crop.")

	detector_scope_relevant_only = detector_scope_label.startswith("E-waste")
	min_box_area_fraction = float(min_box_area_percent / 100.0)
	crop_padding = float(crop_padding_percent / 100.0)

	class_names = assets["class_names"]
	best_arch = assets["best_arch"]
	active_checkpoint = checkpoints[active_arch]
	metric_catalog = pd.DataFrame(assets["metric_catalog"])
	primary_metrics = pd.DataFrame(assets["primary_metrics"])
	architecture_payloads = load_architecture_payloads(str(classification_dir))
	active_arch_payload = architecture_payloads.get(active_arch, {})
	model_inventory_payload = load_models_inventory(str(models_dir))
	inventory_frame = pd.DataFrame(model_inventory_payload["files"])
	inventory_by_category = pd.DataFrame(model_inventory_payload["by_category"])
	inventory_by_section = pd.DataFrame(model_inventory_payload["by_section"])
	dataset_profile_payload = load_dataset_profile(str(data_dir))
	dataset_profile = pd.DataFrame(dataset_profile_payload["rows"])
	supporting = load_supporting_metrics(str(PROJECT_ROOT))
	primary_benchmark_row, primary_benchmark_source, secondary_benchmark_row, secondary_benchmark_source = resolve_benchmark_rows(
		metric_catalog,
		active_arch,
	)
	discrepancy_note = build_metric_discrepancy_note(metric_catalog, active_arch)

	model = load_classifier(
		checkpoint_path=active_checkpoint,
		arch=active_arch,
		class_names=tuple(class_names),
		device_type=runtime.device.type,
	)

	ann_results = supporting["ann_results"]
	ann_results_18cls = supporting["ann_results_18cls"]
	ann_backprop_results = supporting["ann_backprop_results"]
	clustering_metrics = supporting["clustering_metrics"]
	clustering_results = supporting["clustering_results"]
	kmedoids_metrics = supporting["kmedoids_metrics"]
	kmedoids_results = supporting["kmedoids_results"]
	clustering_comparison = supporting["clustering_comparison"]
	competition_leaderboard = supporting["competition_leaderboard"]
	competition_all_players = supporting["competition_all_players"]
	per_class_f1_scores = supporting["per_class_f1_scores"]

	ann_cls_metrics = ann_results_18cls.get("classification_metrics") if isinstance(ann_results_18cls.get("classification_metrics"), Mapping) else {}
	ann_hazard_accuracy = to_float(ann_results_18cls.get("hazard_class_accuracy"))
	if ann_hazard_accuracy is None:
		ann_hazard_accuracy = to_float(ann_cls_metrics.get("hazard_class_accuracy"))

	cluster_groups = clustering_metrics.get("n_clusters", clustering_results.get("n_clusters", "n/a"))

	render_hero(
		title="E-Waste Operations Console",
		copy=(
			"A premium operator surface for the current research stack: benchmarked single-item classification, "
			"detector-assisted cluster review for mixed scenes, sampled video-belt analysis, hazard-aware routing, "
			"benchmark evidence, and analytical support layers."
		),
		chips=[
			f"active backbone {active_arch}",
			f"benchmark leader {best_arch}",
			f"human review threshold {confidence_threshold:.0%}",
			f"scene scan {'enabled' if enable_scene_scan else 'disabled'}",
			"detector-assisted cluster routing",
			f"video sampling every {video_sample_every:.2f}s",
			f"llm {'groq online' if os.getenv('GROQ_API_KEY') else 'deterministic mode'}",
		],
		side_notes=[
			(
				"Primary benchmark",
				format_pct(float(primary_benchmark_row["accuracy"])) if primary_benchmark_row is not None else "n/a",
				f"Active source: {primary_benchmark_source or 'unavailable'}.",
			),
			(
				"Hazard ANN",
				format_pct(ann_hazard_accuracy),
				"Tabular hazard model accuracy for downstream severity support.",
			),
			(
				"Runtime",
				runtime.device.type.upper(),
				f"{runtime.gpu_name or 'CPU execution'} | {f'{runtime.vram_gb:.2f} GB VRAM' if runtime.vram_gb is not None else 'memory profile unavailable'}",
			),
		],
	)

	if discrepancy_note:
		render_banner(
			"Benchmark context",
			discrepancy_note + " The collage example is a mixed-object scene, so a low live confidence is an uncertainty signal rather than a reliable single-class decision.",
			tone="warning",
			icon="warning",
		)

	top_row = st.columns(3)
	with top_row[0]:
		render_metric_tile("Active architecture", active_arch, f"checkpoint: {Path(active_checkpoint).name}", "neutral", icon="cpu")
	with top_row[1]:
		render_metric_tile(
			"Primary benchmark",
			format_pct(float(primary_benchmark_row["accuracy"])) if primary_benchmark_row is not None else "n/a",
			f"source: {primary_benchmark_source or 'unavailable'}",
			"success",
			icon="chart",
		)
	with top_row[2]:
		if secondary_benchmark_row is not None and secondary_benchmark_source:
			render_metric_tile(
				"Secondary benchmark",
				format_pct(float(secondary_benchmark_row["accuracy"])),
				f"source: {secondary_benchmark_source}",
				"warning",
				icon="review",
			)
		else:
			render_metric_tile(
				"Secondary benchmark",
				format_pct(float(primary_benchmark_row["accuracy"])) if primary_benchmark_row is not None else "n/a",
				f"single-source mode ({primary_benchmark_source or 'none'})",
				"neutral",
				icon="review",
			)

	second_row = st.columns(2)
	with second_row[0]:
		render_metric_tile("Hazard ANN accuracy", format_pct(ann_hazard_accuracy), "18-class hazard snapshot", "success", icon="shield")
	with second_row[1]:
		render_metric_tile("Cluster groups", str(cluster_groups), "unsupervised structure available", "neutral", icon="cluster")

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
	if "cluster_pending_image" not in st.session_state:
		st.session_state["cluster_pending_image"] = None
	if "cluster_pending_label" not in st.session_state:
		st.session_state["cluster_pending_label"] = None
	if "cluster_last_upload_token" not in st.session_state:
		st.session_state["cluster_last_upload_token"] = None
	if "last_cluster_result" not in st.session_state:
		st.session_state["last_cluster_result"] = None
	if "pending_video_bytes" not in st.session_state:
		st.session_state["pending_video_bytes"] = None
	if "pending_video_name" not in st.session_state:
		st.session_state["pending_video_name"] = None
	if "video_last_upload_token" not in st.session_state:
		st.session_state["video_last_upload_token"] = None
	if "last_video_result" not in st.session_state:
		st.session_state["last_video_result"] = None

	if workspace == "Operations":
		ops_mode = st.radio(
			"Operational mode",
			options=["Single-item triage", "Cluster image review", "Video belt review"],
			horizontal=True,
			key="operations_mode_selector",
		)
		if ops_mode == "Single-item triage":
			render_single_item_operations(
				model=model,
				class_names=class_names,
				device=runtime.device,
				confidence_threshold=confidence_threshold,
				enable_scene_scan=enable_scene_scan,
				scene_grid=scene_grid,
				scene_tile_floor=scene_tile_floor,
				data_dir=data_dir,
			)
		elif ops_mode == "Cluster image review":
			render_cluster_image_operations(
				classifier_model=model,
				class_names=class_names,
				device=runtime.device,
				confidence_threshold=confidence_threshold,
				data_dir=data_dir,
				detector_model_key=detector_model_key,
				detector_score_threshold=detector_score_threshold,
				max_objects=max_detected_objects,
				min_area_fraction=min_box_area_fraction,
				relevant_only=detector_scope_relevant_only,
				crop_padding=crop_padding,
			)
		else:
			render_video_belt_operations(
				classifier_model=model,
				class_names=class_names,
				device=runtime.device,
				confidence_threshold=confidence_threshold,
				detector_model_key=detector_model_key,
				detector_score_threshold=detector_score_threshold,
				max_objects_per_frame=max_objects_per_frame,
				min_area_fraction=min_box_area_fraction,
				relevant_only=detector_scope_relevant_only,
				crop_padding=crop_padding,
				sample_every_seconds=video_sample_every,
				max_frames=video_max_frames,
			)

	if workspace == "Workflow":
		render_workflow_workspace(st.session_state.get("last_result"))

	if workspace == "Policy":
		render_section_intro(
			"Decision",
			"Policy and routing review",
			"This panel exposes the actual decision output: hazard lookup, compliance signal, recommendation mode, and the trace used to assemble the final response.",
			icon="route",
		)
		result = st.session_state.get("last_result")
		if not result:
			render_panel("Policy queue", "Awaiting inference", "Run inference first to generate hazard and routing guidance.", icon="route")
		else:
			decision = result["decision"]
			diagnostics = result["diagnostics"]
			analysis_kind = str(result.get("analysis_kind", "single_item_review"))
			workflow_report = result.get("cluster_report") if analysis_kind == "cluster_image_review" else result.get("video_report") if analysis_kind == "video_belt_review" else None
			p_top = st.columns(2, gap="large")
			p_bottom = st.columns(2, gap="large")
			with p_top[0]:
				render_metric_tile("Hazard level", decision.get("hazard_level", "n/a"), "mapped from policy taxonomy", "danger" if decision.get("hazard_level") == "HIGH" else "warning", icon="shield")
			with p_top[1]:
				render_metric_tile("SDG target", decision.get("sdg_target", "n/a"), "policy alignment output", "neutral", icon="spark")
			with p_bottom[0]:
				render_metric_tile("Human review", "Required" if decision.get("requires_human_review", True) else "Not required", f"threshold: {decision.get('confidence_threshold', confidence_threshold):.2%}", "warning" if decision.get("requires_human_review", True) else "success", icon="review")
			with p_bottom[1]:
				render_metric_tile("Decision mode", decision.get("agent_mode", "n/a"), f"provider: {decision.get('llm_provider', 'none')} | source: {decision.get('explanation_source', 'n/a')}", "neutral", icon="route")
			l1, l2 = st.columns([1.05, 0.95], gap="large")
			with l1:
				render_panel("Recommended pathway", decision.get("short_recommendation", "n/a"), "routing output after tool execution", tone="success" if not decision.get("requires_human_review", True) else "warning", icon="route")
				st.markdown("#### Material Profile")
				st.write(decision.get("material_profile", "n/a"))
				st.markdown("#### Decision Rationale")
				st.write(decision.get("explanation", "n/a"))
				if decision.get("llm_error"):
					st.warning(f"LLM augmentation failed and the system fell back to deterministic reasoning: {decision['llm_error']}")
			with l2:
				interpretation_copy = (
					"Tie the routing decision to confidence. Confident single-component inputs can proceed automatically; ambiguous or composite scenes should stop at triage."
					if analysis_kind == "single_item_review"
					else "Detector-assisted cluster and video lanes localize multiple objects, classify each crop, and then aggregate a cluster-level routing action. Use the highest hazard band and review flags to decide whether the belt segment can proceed."
				)
				render_banner("Operating interpretation", interpretation_copy, "neutral", icon="route")
				render_badge_row(
					[
						f"compliance: {'ready' if decision.get('compliance_flag', False) else 'escalate'}",
						f"mode: {decision.get('agent_mode', 'n/a')}",
						f"provider: {decision.get('llm_provider', 'none')}",
						f"source: {decision.get('explanation_source', 'n/a')}",
					]
				)
				st.markdown("#### Operational Checklist")
				checklist_items = (
					build_cluster_operational_checklist(workflow_report or result)
					if analysis_kind != "single_item_review"
					else build_operational_checklist(decision, diagnostics)
				)
				render_checklist(checklist_items)
			trace_steps = decision.get("tool_trace", [])
			if trace_steps:
				st.markdown("#### Tool Execution Trace")
				render_trace_steps(trace_steps)
			with st.expander("Raw decision payload"):
				st.json(decision)

	if workspace == "Benchmarks":
		render_section_intro(
			"Benchmarks",
			"Evaluation records",
			"This section surfaces architecture-level evaluation insights from the models folder, with charts generated from metrics payloads instead of raw JSON dumps.",
			icon="chart",
		)
		if discrepancy_note:
			render_banner("Benchmark caution", discrepancy_note + " Do not present the archived and script metrics as if they were one experiment.", "warning", icon="warning")
		b1, b2 = st.columns([0.95, 1.05], gap="large")
		with b1:
			if primary_metrics.empty:
				render_panel("Benchmark records", "Unavailable", "No benchmark metrics JSON found in models/classification.", icon="chart")
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
			selected_metrics = active_arch_payload.get("metrics") if isinstance(active_arch_payload.get("metrics"), Mapping) else {}
			selected_acc = to_float(selected_metrics.get("accuracy"))
			if selected_acc is None:
				selected_acc = to_float(active_arch_payload.get("test_accuracy"))
			selected_macro_f1 = to_float(selected_metrics.get("macro_f1"))
			if selected_macro_f1 is None:
				selected_macro_f1 = to_float(active_arch_payload.get("macro_f1"))
			selected_loss = to_float(active_arch_payload.get("test_loss"))
			render_panel(
				"Selected model",
				f"{active_arch} | acc {format_pct(selected_acc)} | macro-F1 {format_float(selected_macro_f1, 4)}",
				f"Test loss {format_float(selected_loss, 4)}. Insights below are sourced from models/classification/{active_arch}/results.json and companion artifacts.",
				icon="review",
			)
			render_badge_row(["Single-label benchmark", "Held-out evaluation", "Model-specific diagnostics", "No raw JSON rendering"])

		bench_tabs = st.tabs(["Selected Model", "Per-Class F1", "Saved Graphs", "Competition"])
		with bench_tabs[0]:
			col_left, col_right = st.columns(2, gap="large")
			with col_left:
				conf_fig = build_confusion_figure(
					payload=active_arch_payload,
					fallback_class_names=class_names,
					title=f"{active_arch} confusion matrix",
					normalized=False,
				)
				if conf_fig is not None:
					st.pyplot(conf_fig, use_container_width=True)
				else:
					st.info("Confusion matrix payload unavailable for the selected model.")
			with col_right:
				history_fig = build_history_figure(active_arch_payload.get("history"), f"{active_arch} training history")
				if history_fig is not None:
					st.pyplot(history_fig, use_container_width=True)
				else:
					st.info("Training history is unavailable for the selected model.")
			report_fig = build_class_report_heatmap_figure(active_arch_payload, f"{active_arch} class report")
			if report_fig is not None:
				st.pyplot(report_fig, use_container_width=True)
			else:
				st.info("Classification report heatmap is unavailable for the selected model.")
		with bench_tabs[1]:
			f1_fig = build_per_class_f1_heatmap_figure(per_class_f1_scores)
			if f1_fig is not None:
				st.pyplot(f1_fig, use_container_width=True)
			else:
				fallback_f1 = classification_dir / "graphs" / "per_class_f1_comparison.png"
				if fallback_f1.exists():
					st.image(str(fallback_f1), width="stretch")
				else:
					st.info("Per-class F1 payload unavailable.")
		with bench_tabs[2]:
			graph_dir = classification_dir / "graphs"
			graph_images = [
				path
				for path in sorted(graph_dir.glob("*"), key=lambda p: p.name.lower())
				if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
			]
			if not graph_images:
				st.info("No saved graph images were found in models/classification/graphs.")
			else:
				grid = st.columns(2, gap="large")
				for idx, graph_path in enumerate(graph_images):
					with grid[idx % 2]:
						st.image(str(graph_path), caption=graph_path.name, width="stretch")
		with bench_tabs[3]:
			ranking = competition_leaderboard.get("ranking") if isinstance(competition_leaderboard, Mapping) else None
			if isinstance(ranking, list) and ranking:
				comp_frame = pd.DataFrame(ranking)
				st.dataframe(comp_frame[["rank", "model", "model_type", "accuracy", "macro_f1"]], width="stretch", hide_index=True)
				st.bar_chart(comp_frame.set_index("model")[["accuracy", "macro_f1"]], width="stretch")
			elif isinstance(competition_all_players, Mapping) and competition_all_players:
				rows = []
				for model_name, payload in competition_all_players.items():
					if not isinstance(payload, Mapping):
						continue
					metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else payload
					acc = to_float(metrics.get("accuracy")) if isinstance(metrics, Mapping) else None
					macro_f1 = to_float(metrics.get("macro_f1")) if isinstance(metrics, Mapping) else None
					if acc is None or macro_f1 is None:
						continue
					rows.append({"model": model_name, "accuracy": acc, "macro_f1": macro_f1})
				if rows:
					frame = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
					st.dataframe(frame, width="stretch", hide_index=True)
					st.bar_chart(frame.set_index("model")[["accuracy", "macro_f1"]], width="stretch")
				else:
					st.info("Competition payload found but no plottable metrics were detected.")
			else:
				st.info("Competition benchmark payload unavailable.")

		st.markdown("#### Results Drafting")
		if st.button("Draft results paragraph", key="generate_results_paragraph"):
			try:
				with st.spinner("Generating results paragraph..."):
					st.session_state["llm_results_summary"] = call_groq_text(
						prompt=build_results_summary_prompt(
							best_arch=active_arch,
							primary_row=primary_benchmark_row,
							primary_source=primary_benchmark_source,
							secondary_row=secondary_benchmark_row,
							secondary_source=secondary_benchmark_source,
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
			"These panels derive ANN and clustering insights directly from the current models artifacts with chart-first views, including supplementary backpropagation and K-Means vs K-Medoids analysis when those notebooks have been executed.",
			icon="cluster",
		)
		a_top = st.columns(2, gap="large")
		a_bottom = st.columns(2, gap="large")
		kmeans_silhouette = to_float(clustering_metrics.get("silhouette_score")) if isinstance(clustering_metrics, Mapping) else None
		kmedoids_silhouette = to_float(kmedoids_metrics.get("silhouette_score")) if isinstance(kmedoids_metrics, Mapping) else None
		with a_top[0]:
			render_metric_tile("Hazard class accuracy", format_pct(ann_hazard_accuracy), "18-class ANN hazard snapshot", "success", icon="shield")
		with a_top[1]:
			regression_r2 = to_float(ann_results.get("regression_metrics", {}).get("r2")) if isinstance(ann_results, Mapping) else None
			render_metric_tile("Regression R2", format_float(regression_r2), "hazard severity regression", "neutral", icon="chart")
		with a_bottom[0]:
			render_metric_tile("K-Means silhouette", format_float(kmeans_silhouette), "baseline cluster separation score", "warning", icon="cluster")
		with a_bottom[1]:
			render_metric_tile(
				"K-Medoids silhouette" if kmedoids_silhouette is not None else "NMI",
				format_float(kmedoids_silhouette) if kmedoids_silhouette is not None else format_float(to_float(clustering_metrics.get("normalized_mutual_info")) if isinstance(clustering_metrics, Mapping) else None),
				"medoid-based cluster separation" if kmedoids_silhouette is not None else "alignment with known labels",
				"neutral",
				icon="spark",
			)

		analytics_tabs = st.tabs(["Hazard ANN", "Clustering", "Competition", "System Copilot"])
		with analytics_tabs[0]:
			ann_fig = build_ann_overview_figure(ann_results_18cls or ann_results)
			if ann_fig is not None:
				st.pyplot(ann_fig, use_container_width=True)
			else:
				st.info("ANN insights are unavailable because ann_results_18cls.json could not be parsed.")
			backprop_fig = build_ann_backprop_figure(ann_backprop_results)
			if backprop_fig is not None:
				st.pyplot(backprop_fig, use_container_width=True)
				backprop_summary = ann_backprop_results.get("backprop_summary") if isinstance(ann_backprop_results.get("backprop_summary"), Mapping) else {}
				if backprop_summary:
					st.dataframe(pd.DataFrame([backprop_summary]), width="stretch", hide_index=True)
			else:
				st.info("Run the supplementary ANN backpropagation notebook to surface gradient-flow diagnostics here.")
		with analytics_tabs[1]:
			cluster_overview_fig = build_clustering_overview_figure(clustering_metrics)
			if cluster_overview_fig is not None:
				st.pyplot(cluster_overview_fig, use_container_width=True)
			else:
				st.info("Clustering metrics payload unavailable.")

			cluster_compare_fig = build_clustering_comparison_figure(clustering_comparison)
			if cluster_compare_fig is not None:
				st.pyplot(cluster_compare_fig, use_container_width=True)
			else:
				st.info("Run the supplementary K-Means vs K-Medoids notebook to populate the algorithm comparison view.")

			cluster_algo = st.selectbox(
				"Cluster algorithm view",
				options=["K-Means", "K-Medoids"] if kmedoids_metrics else ["K-Means"],
				index=0,
				key="cluster_algo_view",
			)
			cluster_labels_name = "kmedoids_cluster_labels.npy" if cluster_algo == "K-Medoids" else "cluster_labels.npy"
			tsne_clusters_name = "kmedoids_tsne_clusters.npy" if cluster_algo == "K-Medoids" else "tsne_clusters.npy"
			active_clustering_results = kmedoids_results if cluster_algo == "K-Medoids" else clustering_results
			class_names_from_clustering = active_clustering_results.get("class_names") if isinstance(active_clustering_results.get("class_names"), list) else class_names
			composition_fig = build_cluster_composition_figure(
				clustering_dir,
				class_names=class_names_from_clustering,
				cluster_labels_name=cluster_labels_name,
			)
			if composition_fig is not None:
				st.pyplot(composition_fig, use_container_width=True)
			else:
				st.info(f"Class-vs-cluster composition could not be generated for {cluster_algo}.")

			tsne_fig = build_tsne_cluster_figure(clustering_dir, tsne_clusters_name=tsne_clusters_name)
			if tsne_fig is not None:
				st.pyplot(tsne_fig, use_container_width=True)
			else:
				st.info(f"t-SNE scatter could not be generated for {cluster_algo}.")
		with analytics_tabs[2]:
			ranking = competition_leaderboard.get("ranking") if isinstance(competition_leaderboard, Mapping) else None
			if isinstance(ranking, list) and ranking:
				comp_frame = pd.DataFrame(ranking)
				st.dataframe(comp_frame[["rank", "model", "model_type", "accuracy", "macro_f1"]], width="stretch", hide_index=True)
				st.bar_chart(comp_frame.set_index("model")[["accuracy", "macro_f1"]], width="stretch")
			else:
				st.info("Competition leaderboard is unavailable.")
		with analytics_tabs[3]:
			render_system_copilot(
				input_key="analytics_copilot_input",
				active_arch=active_arch,
				class_names=class_names,
				primary_row=primary_benchmark_row,
				ann_results=ann_results_18cls or ann_results,
				ann_backprop_results=ann_backprop_results,
				clustering_metrics=clustering_metrics,
				kmedoids_metrics=kmedoids_metrics,
				clustering_comparison=clustering_comparison,
				last_result=st.session_state.get("last_result"),
			)

		st.markdown("#### Discussion Drafting")
		if st.button("Draft discussion paragraph", key="generate_discussion_paragraph"):
			try:
				with st.spinner("Generating discussion paragraph..."):
					st.session_state["llm_discussion_summary"] = call_groq_text(
						prompt=build_discussion_prompt(
							ann_results_18cls=ann_results_18cls,
							clustering_results=clustering_metrics or clustering_results,
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

	if workspace == "Copilot":
		render_section_intro(
			"Copilot",
			"System-only research assistant",
			"This assistant is constrained to the current project. Use it to explain the dashboard workflow, interpret benchmark records, compare K-Means and K-Medoids, summarize ANN backprop diagnostics, or answer questions about the latest inference result.",
			icon="chat",
		)
		render_system_copilot(
			input_key="copilot_workspace_input",
			active_arch=active_arch,
			class_names=class_names,
			primary_row=primary_benchmark_row,
			ann_results=ann_results_18cls or ann_results,
			ann_backprop_results=ann_backprop_results,
			clustering_metrics=clustering_metrics,
			kmedoids_metrics=kmedoids_metrics,
			clustering_comparison=clustering_comparison,
			last_result=st.session_state.get("last_result"),
		)

	if workspace == "Registry":
		render_section_intro(
			"Registry",
			"Model and data inventory",
			"This view inventories checkpoints, dataset balance, taxonomy, and every artifact under models/ with visual summaries.",
			icon="database",
		)
		registry_rows = []
		for arch, path in assets["checkpoints"].items():
			registry_rows.append(
				{
					"architecture": arch,
					"checkpoint_path": str(path).replace("\\", "/"),
					"status": "active" if arch == active_arch else "available",
				}
			)
		registry_tabs = st.tabs(["Checkpoints", "Dataset", "Taxonomy", "Artifacts"])
		with registry_tabs[0]:
			st.dataframe(pd.DataFrame(registry_rows), width="stretch", hide_index=True)
		with registry_tabs[1]:
			if not dataset_profile.empty:
				split_choice = st.selectbox("Dataset split", options=sorted(dataset_profile["split"].unique().tolist()))
				split_frame = dataset_profile.loc[dataset_profile["split"] == split_choice].copy()
				st.dataframe(split_frame.sort_values("class_name"), width="stretch", hide_index=True)
				st.bar_chart(split_frame.set_index("class_name")[["count"]], width="stretch")
				t1, t2, t3 = st.columns(3)
				with t1:
					render_metric_tile("Train images", str(dataset_profile_payload["totals"].get("train", "n/a")), "single-label foldered data", "neutral", icon="database")
				with t2:
					render_metric_tile("Validation images", str(dataset_profile_payload["totals"].get("val", "n/a")), "held-out tuning split", "neutral", icon="review")
				with t3:
					render_metric_tile("Test images", str(dataset_profile_payload["totals"].get("test", "n/a")), "held-out evaluation split", "neutral", icon="chart")
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
		with registry_tabs[2]:
			st.markdown("#### Hazard Taxonomy")
			st.dataframe(pd.DataFrame(taxonomy_rows), width="stretch", hide_index=True)
		with registry_tabs[3]:
			total_files = int(len(inventory_frame)) if not inventory_frame.empty else 0
			total_size_mb = float(inventory_frame["size_mb"].sum()) if not inventory_frame.empty else 0.0
			cat_row = st.columns(3, gap="large")
			with cat_row[0]:
				render_metric_tile("Model artifacts", str(total_files), "all files under models/", "neutral", icon="database")
			with cat_row[1]:
				render_metric_tile("Total artifact size", f"{total_size_mb:.2f} MB", "combined payload footprint", "neutral", icon="stack")
			with cat_row[2]:
				visual_count = int((inventory_frame["category"] == "visual").sum()) if not inventory_frame.empty else 0
				render_metric_tile("Visual artifacts", str(visual_count), "ready for direct dashboard rendering", "success", icon="gallery")

			chart_left, chart_right = st.columns(2, gap="large")
			with chart_left:
				if not inventory_by_section.empty:
					section_chart = inventory_by_section.set_index("section")[["file_count"]]
					st.bar_chart(section_chart, width="stretch")
			with chart_right:
				if not inventory_by_category.empty:
					category_chart = inventory_by_category.set_index("category")[["file_count"]]
					st.bar_chart(category_chart, width="stretch")

			if inventory_frame.empty:
				st.info("No files were discovered under models/.")
			else:
				section_options = sorted(inventory_frame["section"].unique().tolist())
				category_options = sorted(inventory_frame["category"].unique().tolist())
				selected_sections = st.multiselect("Sections", options=section_options, default=section_options)
				selected_categories = st.multiselect("Categories", options=category_options, default=category_options)

				filtered = inventory_frame.loc[
					inventory_frame["section"].isin(selected_sections)
					& inventory_frame["category"].isin(selected_categories)
				].copy()
				filtered = filtered.sort_values(["section", "category", "path"])
				st.dataframe(
					filtered[["section", "path", "category", "extension", "size", "modified", "insight"]],
					width="stretch",
					hide_index=True,
				)

				preview_candidates = filtered.loc[filtered["category"] == "visual", "path"].tolist()
				if preview_candidates:
					selected_preview = st.selectbox("Visual artifact preview", options=preview_candidates)
					st.image(str(models_dir / selected_preview), caption=selected_preview, width="stretch")
				else:
					st.caption("No visual artifact in the current filter selection.")
		render_banner(
			"Deployment scope",
			"The current vision stack is a single-label component classifier. Present the scene scan as composite-image triage and reserve true detection or multi-label classification as future work.",
			"warning",
			icon="warning",
		)


if __name__ == "__main__":
	if not _is_streamlit_runtime_active():
		print("This is a Streamlit app. Launch it with: streamlit run dashboard/app.py")
		sys.exit(0)
	main()
