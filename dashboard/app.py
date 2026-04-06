from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import html
from pathlib import Path
import json
import os
import random
import sys
import time
from textwrap import dedent
from urllib import request as urllib_request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
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


def build_cluster_composition_figure(clustering_dir: Path, class_names: list[str] | None = None) -> plt.Figure | None:
	class_labels = load_npy_array(clustering_dir / "labels.npy")
	cluster_labels = load_npy_array(clustering_dir / "cluster_labels.npy")
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


def build_tsne_cluster_figure(clustering_dir: Path, max_points: int = 8000) -> plt.Figure | None:
	tsne = load_npy_array(clustering_dir / "tsne_result.npy")
	clusters = load_npy_array(clustering_dir / "tsne_clusters.npy")
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
		"clustering_metrics": load_json(project_root / "models" / "clustering" / "clustering_metrics.json") or {},
		"clustering_results": load_json(project_root / "models" / "clustering" / "clustering_results.json") or {},
		"competition_leaderboard": load_json(project_root / "models" / "competition" / "leaderboard.json") or {},
		"competition_all_players": load_json(project_root / "models" / "competition" / "all_players_results.json") or {},
		"competition_deep": load_json(project_root / "models" / "competition" / "deep_results.json") or {},
		"competition_traditional": load_json(project_root / "models" / "competition" / "traditional_ml_results.json") or {},
		"per_class_f1_scores": load_json(project_root / "models" / "classification" / "per_class_f1_scores.json") or {},
		"benchmark_summary": load_json(project_root / "models" / "classification" / "benchmark_summary.json") or {},
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
		st.caption("Classification, review routing, benchmark records, and data registry for the current research system.")
		active_arch = st.selectbox("Active model", options=available_arches, index=available_arches.index(default_arch))
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
	clustering_metrics = supporting["clustering_metrics"]
	clustering_results = supporting["clustering_results"]
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
			"A premium operator surface for the current research stack: single-label classification, "
			"confidence-gated review, hazard-aware routing, benchmark evidence, and analytical support layers. "
			"Mixed scenes are surfaced honestly as triage cases rather than being overstated as true detection."
		),
		chips=[
			f"active backbone {active_arch}",
			f"benchmark leader {best_arch}",
			f"human review threshold {confidence_threshold:.0%}",
			f"scene scan {'enabled' if enable_scene_scan else 'disabled'}",
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
			"These panels derive ANN and clustering insights directly from the current models artifacts with chart-first views.",
			icon="cluster",
		)
		a_top = st.columns(2, gap="large")
		a_bottom = st.columns(2, gap="large")
		with a_top[0]:
			render_metric_tile("Hazard class accuracy", format_pct(ann_hazard_accuracy), "18-class ANN hazard snapshot", "success", icon="shield")
		with a_top[1]:
			regression_r2 = to_float(ann_results.get("regression_metrics", {}).get("r2")) if isinstance(ann_results, Mapping) else None
			render_metric_tile("Regression R2", format_float(regression_r2), "hazard severity regression", "neutral", icon="chart")
		with a_bottom[0]:
			render_metric_tile("Silhouette", format_float(to_float(clustering_metrics.get("silhouette_score")) if isinstance(clustering_metrics, Mapping) else None), "cluster separation score", "warning", icon="cluster")
		with a_bottom[1]:
			render_metric_tile("NMI", format_float(to_float(clustering_metrics.get("normalized_mutual_info")) if isinstance(clustering_metrics, Mapping) else None), "alignment with known labels", "neutral", icon="spark")

		analytics_tabs = st.tabs(["Hazard ANN", "Clustering", "Competition"])
		with analytics_tabs[0]:
			ann_fig = build_ann_overview_figure(ann_results_18cls or ann_results)
			if ann_fig is not None:
				st.pyplot(ann_fig, use_container_width=True)
			else:
				st.info("ANN insights are unavailable because ann_results_18cls.json could not be parsed.")
		with analytics_tabs[1]:
			cluster_overview_fig = build_clustering_overview_figure(clustering_metrics)
			if cluster_overview_fig is not None:
				st.pyplot(cluster_overview_fig, use_container_width=True)
			else:
				st.info("Clustering metrics payload unavailable.")

			class_names_from_clustering = clustering_results.get("class_names") if isinstance(clustering_results.get("class_names"), list) else class_names
			composition_fig = build_cluster_composition_figure(clustering_dir, class_names=class_names_from_clustering)
			if composition_fig is not None:
				st.pyplot(composition_fig, use_container_width=True)
			else:
				st.info("Class-vs-cluster composition could not be generated from labels.npy and cluster_labels.npy.")

			tsne_fig = build_tsne_cluster_figure(clustering_dir)
			if tsne_fig is not None:
				st.pyplot(tsne_fig, use_container_width=True)
			else:
				st.info("t-SNE scatter could not be generated from tsne_result.npy and tsne_clusters.npy.")
		with analytics_tabs[2]:
			ranking = competition_leaderboard.get("ranking") if isinstance(competition_leaderboard, Mapping) else None
			if isinstance(ranking, list) and ranking:
				comp_frame = pd.DataFrame(ranking)
				st.dataframe(comp_frame[["rank", "model", "model_type", "accuracy", "macro_f1"]], width="stretch", hide_index=True)
				st.bar_chart(comp_frame.set_index("model")[["accuracy", "macro_f1"]], width="stretch")
			else:
				st.info("Competition leaderboard is unavailable.")

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
