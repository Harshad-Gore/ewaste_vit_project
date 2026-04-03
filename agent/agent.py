from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
	as_payload,
	disposal_recommendation,
	hazard_lookup,
	regulation_check,
)


@dataclass
class AgentInput:
	component: str
	confidence: float


class EwasteDecisionAgent:
	"""rule-first decision agent with optional llm explanation expansion."""

	def __init__(self, llm_fn: Callable[[str], str] | None = None, confidence_threshold: float = 0.70):
		self.llm_fn = llm_fn
		self.confidence_threshold = confidence_threshold
		self.system_prompt = SYSTEM_PROMPT

	def run(self, payload: AgentInput) -> dict:
		lookup = hazard_lookup(payload.component)
		compliance = regulation_check(
			hazard_level=lookup.hazard_level,
			confidence=payload.confidence,
			confidence_threshold=self.confidence_threshold,
		)
		recommendation = disposal_recommendation(
			lookup=lookup,
			confidence=payload.confidence,
			llm_fn=self.llm_fn,
		)

		out = as_payload(lookup, compliance, recommendation)
		out["confidence"] = payload.confidence
		out["system_prompt_version"] = "v1"
		return out

