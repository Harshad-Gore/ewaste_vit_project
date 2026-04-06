from __future__ import annotations

from collections.abc import Mapping
import html
from pathlib import Path
import json
import os
import random
import sys
import time
from textwrap import dedent
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

EVAL_TRANSFORM = build_eval_transform(224)

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

	render_hero(
		title="E-Waste Operations Console",
		copy=(
			"A premium operator surface for the current research stack: single-label classification, "
			"confidence-gated review, hazard-aware routing, benchmark evidence, and analytical support layers. "
			"Mixed scenes are surfaced honestly as triage cases rather than being overstated as true detection."
		),
		chips=[
			f"active backbone {best_arch}",
			f"human review threshold {confidence_threshold:.0%}",
			f"scene scan {'enabled' if enable_scene_scan else 'disabled'}",
			f"llm {'groq online' if os.getenv('GROQ_API_KEY') else 'deterministic mode'}",
		],
		side_notes=[
			(
				"Archived benchmark",
				format_pct(float(best_archived_row["accuracy"])) if best_archived_row is not None else "n/a",
				"Best archived single-label benchmark currently available in the repository.",
			),
			(
				"Hazard ANN",
				format_pct(float(ann_results_18cls.get("hazard_class_accuracy"))) if ann_results_18cls else "n/a",
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
		render_metric_tile("Deployed architecture", best_arch, f"checkpoint: {Path(best_checkpoint).name}", "neutral", icon="cpu")
	with top_row[1]:
		render_metric_tile("Archived benchmark", format_pct(float(best_archived_row["accuracy"])) if best_archived_row is not None else "n/a", "source: dl_results.json", "success", icon="chart")
	with top_row[2]:
		render_metric_tile("Script benchmark", format_pct(float(best_script_row["accuracy"])) if best_script_row is not None else "n/a", "source: test_results.json", "warning", icon="review")

	second_row = st.columns(2)
	with second_row[0]:
		render_metric_tile("Hazard ANN accuracy", format_pct(float(ann_results_18cls.get("hazard_class_accuracy"))) if ann_results_18cls else "n/a", "18-class hazard snapshot", "success", icon="shield")
	with second_row[1]:
		render_metric_tile("Cluster groups", str(clustering_results.get("n_clusters", "n/a")), "unsupervised structure available", "neutral", icon="cluster")

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
				render_panel("Intake queue", "No active image", "Load a test image or upload a sample to start an operator review cycle.", icon="gallery")

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
		if result:
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
				render_banner("Operating interpretation", "Tie the routing decision to confidence. Confident single-component inputs can proceed automatically; ambiguous or composite scenes should stop at triage.", "neutral", icon="route")
				render_badge_row(
					[
						f"compliance: {'ready' if decision.get('compliance_flag', False) else 'escalate'}",
						f"mode: {decision.get('agent_mode', 'n/a')}",
						f"provider: {decision.get('llm_provider', 'none')}",
						f"source: {decision.get('explanation_source', 'n/a')}",
					]
				)
				st.markdown("#### Operational Checklist")
				render_checklist(build_operational_checklist(decision, diagnostics))
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
			"This section surfaces benchmark tables, confusion matrices, training curves, and interpretability artifacts already present in the repository.",
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
			render_panel(
				"Evidence interpretation",
				"Read the benchmark in scope",
				"`dl_results.json` is the stronger archived snapshot. `test_results.json` is the current script-generated benchmark when present. Mixed-scene failure is expected because the dataset is single-label by design.",
				icon="review",
			)
			render_badge_row(["Single-label benchmark", "Held-out evaluation", "Interpretability available", "Do not overclaim scene performance"])
		bench_tabs = st.tabs(["Confusion", "Per-Class F1", "Curves", "Grad-CAM"])
		with bench_tabs[0]:
			st.image(str(classification_dir / "graphs" / "confusion_matrices_18cls.png"), width="stretch")
		with bench_tabs[1]:
			st.image(str(classification_dir / "graphs" / "per_class_f1_comparison.png"), width="stretch")
		with bench_tabs[2]:
			st.image(str(classification_dir / "graphs" / "training_curves_18cls.png"), width="stretch")
		with bench_tabs[3]:
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
			icon="cluster",
		)
		a_top = st.columns(2, gap="large")
		a_bottom = st.columns(2, gap="large")
		with a_top[0]:
			render_metric_tile("Hazard class accuracy", format_pct(float(ann_results_18cls.get("hazard_class_accuracy"))) if ann_results_18cls else "n/a", "18-class ANN hazard snapshot", "success", icon="shield")
		with a_top[1]:
			render_metric_tile("Regression R2", format_float(float(ann_results.get("regression_metrics", {}).get("r2"))) if ann_results else "n/a", "hazard severity regression", "neutral", icon="chart")
		with a_bottom[0]:
			render_metric_tile("Silhouette", format_float(float(clustering_metrics.get("silhouette_score"))) if clustering_metrics else "n/a", "cluster separation score", "warning", icon="cluster")
		with a_bottom[1]:
			render_metric_tile("NMI", format_float(float(clustering_results.get("normalized_mutual_info"))) if clustering_results else "n/a", "alignment with known labels", "neutral", icon="spark")
		analytics_tabs = st.tabs(["Hazard Model", "Clustering"])
		with analytics_tabs[0]:
			a_img1, a_img2 = st.columns(2, gap="large")
			with a_img1:
				st.image(str(ann_dir / "feature_importance.png"), width="stretch")
			with a_img2:
				st.image(str(ann_dir / "hazard_class_confusion.png"), width="stretch")
		with analytics_tabs[1]:
			c_img1, c_img2 = st.columns(2, gap="large")
			with c_img1:
				st.image(str(clustering_dir / "tsne_by_class.png"), width="stretch")
			with c_img2:
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
			icon="database",
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
		registry_tabs = st.tabs(["Checkpoints", "Dataset", "Taxonomy"])
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
