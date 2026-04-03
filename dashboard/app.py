from __future__ import annotations

from pathlib import Path
import json
import sys

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent import AgentInput, EwasteDecisionAgent  # noqa: E402
from agent.tools import HAZARD_MAP  # noqa: E402


def load_best_model_info() -> dict | None:
	best_path = PROJECT_ROOT / "models" / "classification" / "best_model.json"
	if not best_path.exists():
		return None
	with best_path.open("r", encoding="utf-8") as fp:
		return json.load(fp)


def main() -> None:
	st.set_page_config(page_title="ewaste decision support", layout="wide")
	st.title("ewaste classification + disposal support")
	st.caption("sdg 12.4 aligned decision layer")

	model_info = load_best_model_info()
	if model_info is None:
		st.warning("best model summary not found at models/classification/best_model.json")
	else:
		c1, c2, c3 = st.columns(3)
		c1.metric("best architecture", model_info.get("best_arch", "n/a"))
		metrics = model_info.get("results", {})
		c2.metric("test accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
		c3.metric("macro f1", f"{metrics.get('macro_f1', 0):.4f}")

	st.divider()
	st.subheader("agentic disposal recommendation")

	components = sorted(HAZARD_MAP.keys())
	selected_component = st.selectbox("component class", components)
	confidence_pct = st.slider("model confidence (%)", min_value=1, max_value=100, value=85)

	if st.button("generate recommendation", type="primary"):
		agent = EwasteDecisionAgent(confidence_threshold=0.70)
		decision = agent.run(
			AgentInput(
				component=selected_component,
				confidence=confidence_pct / 100.0,
			)
		)

		d1, d2, d3 = st.columns(3)
		d1.metric("hazard level", decision["hazard_level"])
		d2.metric("sdg target", decision["sdg_target"])
		d3.metric("human review", "yes" if decision["requires_human_review"] else "no")

		st.markdown("**recommended pathway**")
		st.write(decision["short_recommendation"])

		st.markdown("**rationale**")
		st.write(decision["explanation"])

		st.markdown("**material profile**")
		st.write(decision["material_profile"])

		with st.expander("raw decision payload"):
			st.json(decision)


if __name__ == "__main__":
	main()

