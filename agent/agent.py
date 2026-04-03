from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request
from typing import Callable

from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
	as_payload,
	disposal_recommendation,
	hazard_lookup,
	regulation_check,
)

try:
	from langchain_anthropic import ChatAnthropic
except Exception:  # pragma: no cover - optional runtime dependency
	ChatAnthropic = None


def _load_local_env() -> None:
	"""Load key=value pairs from project .env if process env is missing them."""
	project_root = Path(__file__).resolve().parents[1]
	env_path = project_root / ".env"
	if not env_path.exists():
		return

	try:
		lines = env_path.read_text(encoding="utf-8").splitlines()
	except OSError:
		return

	for raw in lines:
		line = raw.strip()
		if not line or line.startswith("#"):
			continue
		if line.lower().startswith("export "):
			line = line[7:].strip()
		if "=" not in line:
			continue

		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip()
		if not key:
			continue

		if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
			value = value[1:-1]

		if not os.getenv(key):
			os.environ[key] = value


_load_local_env()


@dataclass
class AgentInput:
	component: str
	confidence: float


@dataclass
class AgentTraceStep:
	step: str
	status: str
	summary: str


class EwasteDecisionAgent:
	"""Tool-driven decision agent with optional LLM explanation expansion."""

	def __init__(self, llm_fn: Callable[[str], str] | None = None, confidence_threshold: float = 0.70):
		self.llm_fn = llm_fn or self._build_default_llm_fn()
		self.confidence_threshold = confidence_threshold
		self.system_prompt = SYSTEM_PROMPT
		self.llm_provider = self._detect_provider()

	def _detect_provider(self) -> str:
		if os.getenv("GROQ_API_KEY"):
			return "groq"
		if os.getenv("ANTHROPIC_API_KEY"):
			return "anthropic"
		return "none"

	def _build_default_llm_fn(self) -> Callable[[str], str] | None:
		groq_llm = self._build_groq_llm_fn()
		if groq_llm is not None:
			return groq_llm

		return self._build_anthropic_llm_fn()

	def _build_groq_llm_fn(self) -> Callable[[str], str] | None:
		api_key = os.getenv("GROQ_API_KEY")
		if not api_key:
			return None

		model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

		def _call(prompt: str) -> str:
			payload = {
				"model": model_name,
				"temperature": 0.2,
				"messages": [
					{
						"role": "system",
						"content": self.system_prompt,
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
					"User-Agent": "ewaste-vit-project/1.0 (+streamlit-agent)",
				},
				method="POST",
			)

			try:
				with urllib_request.urlopen(req, timeout=30) as response:
					body = json.loads(response.read().decode("utf-8"))
			except urllib_error.HTTPError as exc:
				error_body = exc.read().decode("utf-8", errors="replace")
				raise RuntimeError(f"groq api error {exc.code}: {error_body}") from exc

			choices = body.get("choices", [])
			if not choices:
				raise RuntimeError("groq returned no choices")

			message = choices[0].get("message", {})
			content = message.get("content", "")
			if not content:
				raise RuntimeError("groq returned empty content")
			return str(content).strip()

		return _call

	def _build_anthropic_llm_fn(self) -> Callable[[str], str] | None:
		api_key = os.getenv("ANTHROPIC_API_KEY")
		if not api_key or ChatAnthropic is None:
			return None

		model = ChatAnthropic(
			model="claude-3-5-sonnet-latest",
			temperature=0.1,
			anthropic_api_key=api_key,
		)

		def _call(prompt: str) -> str:
			response = model.invoke(prompt)
			content = getattr(response, "content", response)
			if isinstance(content, str):
				return content
			if isinstance(content, list):
				chunks: list[str] = []
				for item in content:
					text = getattr(item, "text", None)
					if text:
						chunks.append(str(text))
					elif isinstance(item, dict) and "text" in item:
						chunks.append(str(item["text"]))
					else:
						chunks.append(str(item))
				return "\n".join(chunks).strip()
			return str(content).strip()

		return _call

	def run(self, payload: AgentInput) -> dict:
		lookup = hazard_lookup(payload.component)
		compliance = regulation_check(
			hazard_level=lookup.hazard_level,
			confidence=payload.confidence,
			confidence_threshold=self.confidence_threshold,
		)
		explanation_source = "rule-based"
		agent_mode = "deterministic_tool_agent"
		llm_error: str | None = None

		try:
			recommendation = disposal_recommendation(
				lookup=lookup,
				confidence=payload.confidence,
				llm_fn=self.llm_fn,
			)
			if self.llm_fn is not None:
				explanation_source = "llm"
				agent_mode = "llm_augmented_tool_agent"
		except Exception as exc:  # pragma: no cover - defensive fallback for remote LLM failures
			llm_error = str(exc)
			recommendation = disposal_recommendation(
				lookup=lookup,
				confidence=payload.confidence,
				llm_fn=None,
			)

		trace = [
			asdict(
				AgentTraceStep(
					step="hazard_lookup",
					status="completed",
					summary=f"{lookup.component} mapped to {lookup.hazard_level} risk with material profile {lookup.material_profile}.",
				)
			),
			asdict(
				AgentTraceStep(
					step="regulation_check",
					status="completed",
					summary=(
						f"Confidence {payload.confidence:.2%} evaluated against threshold {self.confidence_threshold:.2%}; "
						f"human review = {'yes' if compliance.requires_human_review else 'no'}."
					),
				)
			),
			asdict(
				AgentTraceStep(
					step="disposal_recommendation",
					status="completed",
					summary=f"Recommended pathway: {recommendation.short_recommendation}. Explanation source: {explanation_source}.",
				)
			),
		]

		out = as_payload(lookup, compliance, recommendation)
		out["confidence"] = payload.confidence
		out["agent_mode"] = agent_mode
		out["explanation_source"] = explanation_source
		out["llm_provider"] = self.llm_provider
		out["tool_trace"] = trace
		out["system_prompt_version"] = "v2"
		if llm_error:
			out["llm_error"] = llm_error
		return out

