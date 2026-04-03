from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable


HAZARD_MAP = {
	"Battery": "HIGH",
	"PCB": "HIGH",
	"Mobile": "HIGH",
	"Television": "HIGH",
	"Laptop": "HIGH",
	"light bulbs": "HIGH",
	"Refrigerator": "HIGH",
	"Air-Conditioner": "HIGH",
	"Microwave": "MEDIUM",
	"Washing Machine": "MEDIUM",
	"Printer": "MEDIUM",
	"Microchip-IC": "MEDIUM",
	"Keyboard": "LOW",
	"Mouse": "LOW",
	"Resistor": "LOW",
	"transistor": "LOW",
	"heat-sink": "LOW",
	"Passive-Component": "LOW",
}


MATERIAL_MAP = {
	"Battery": "lithium/cadmium/lead acid",
	"PCB": "fr4 composite with lead solder and metals",
	"Mobile": "lithium battery, rare earth metals, mixed glass/plastic",
	"Television": "lead glass, mercury-bearing parts, mixed plastic",
	"Laptop": "battery, soldered electronics, mixed metals",
	"light bulbs": "glass with possible mercury content",
	"Refrigerator": "steel/copper with refrigerant gas",
	"Air-Conditioner": "aluminium/copper with refrigerant gas",
	"Microwave": "steel/copper and magnetron assembly",
	"Washing Machine": "steel body, motor copper, mixed polymers",
	"Printer": "plastic shell, toner residue, pcb",
	"Microchip-IC": "silicon package with metal traces",
	"Keyboard": "abs plastic, membrane, copper traces",
	"Mouse": "abs plastic with small pcb",
	"Resistor": "ceramic and carbon/metal film",
	"transistor": "semiconductor package",
	"heat-sink": "aluminium/copper thermal dissipation block",
	"Passive-Component": "mixed passive electronics",
}


RISK_DRIVER_MAP = {
	"Battery": "lithium, cadmium, lead, and electrolyte leakage",
	"PCB": "lead solder, brominated resins, and mixed heavy metals",
	"Mobile": "integrated battery packs, mixed metals, and bonded electronics",
	"Television": "lead-bearing glass, mercury-bearing parts, and mixed plastics",
	"Laptop": "embedded batteries, soldered boards, and mixed alloys",
	"light bulbs": "mercury exposure risk and fragile glass breakage",
	"Refrigerator": "refrigerants, compressor oils, and heavy appliance metals",
	"Air-Conditioner": "refrigerants, compressor assemblies, and copper-aluminium systems",
	"Microwave": "magnetron assemblies, capacitors, and appliance metals",
	"Washing Machine": "motors, metal chassis, and mixed polymers",
	"Printer": "toner residue, embedded pcb assemblies, and mixed plastics",
	"Microchip-IC": "semiconductor packaging and concentrated metal traces",
	"Keyboard": "membrane layers, copper traces, and mixed polymers",
	"Mouse": "small pcb assemblies and mixed plastics",
	"Resistor": "ceramic bodies and metal or carbon films",
	"transistor": "semiconductor packaging and small metal leads",
	"heat-sink": "metal recovery handling rather than toxic content",
	"Passive-Component": "small electronic assemblies requiring controlled sorting",
}


DISPOSAL_MAP = {
	"Battery": "send to hazardous battery recycling facility",
	"PCB": "send to certified ewaste recycler for metal recovery",
	"Mobile": "send to certified electronics recycler",
	"Television": "send to hazardous ewaste channel with mercury/lead handling",
	"Laptop": "remove battery then send to certified ewaste recycler",
	"light bulbs": "send to mercury-safe lamp collection stream",
	"Refrigerator": "recover refrigerant first, then dismantle in certified facility",
	"Air-Conditioner": "recover refrigerant first, then dismantle in certified facility",
	"Microwave": "route to appliance metals recovery stream",
	"Washing Machine": "route to appliance metals recovery stream",
	"Printer": "route to ewaste stream with toner-safe handling",
	"Microchip-IC": "route to component-level ewaste recovery",
	"Keyboard": "route to plastics and small-ewaste stream",
	"Mouse": "route to plastics and small-ewaste stream",
	"Resistor": "route to small component ewaste stream",
	"transistor": "route to small component ewaste stream",
	"heat-sink": "route to clean metals recycling stream",
	"Passive-Component": "route to component-level ewaste recovery",
}


@dataclass
class HazardLookupResult:
	component: str
	hazard_level: str
	material_profile: str
	disposal_pathway: str


@dataclass
class RegulationCheckResult:
	sdg_target: str
	compliance_flag: bool
	requires_human_review: bool
	confidence_threshold: float


@dataclass
class RecommendationResult:
	short_recommendation: str
	explanation: str


def hazard_lookup(component: str) -> HazardLookupResult:
	if component not in HAZARD_MAP:
		raise ValueError(f"unknown component: {component}")

	return HazardLookupResult(
		component=component,
		hazard_level=HAZARD_MAP[component],
		material_profile=MATERIAL_MAP[component],
		disposal_pathway=DISPOSAL_MAP[component],
	)


def regulation_check(
	hazard_level: str,
	confidence: float,
	confidence_threshold: float = 0.70,
) -> RegulationCheckResult:
	if hazard_level not in {"HIGH", "MEDIUM", "LOW"}:
		raise ValueError(f"invalid hazard level: {hazard_level}")

	# high and medium hazard are mapped to sdg 12.4, low to sdg 12.5 support.
	sdg_target = "SDG 12.4" if hazard_level in {"HIGH", "MEDIUM"} else "SDG 12.5"

	requires_human_review = confidence < confidence_threshold
	return RegulationCheckResult(
		sdg_target=sdg_target,
		compliance_flag=True,
		requires_human_review=requires_human_review,
		confidence_threshold=confidence_threshold,
	)


def disposal_recommendation(
	lookup: HazardLookupResult,
	confidence: float,
	llm_fn: Callable[[str], str] | None = None,
) -> RecommendationResult:
	risk_drivers = RISK_DRIVER_MAP.get(lookup.component, lookup.material_profile)
	if llm_fn is None:
		if confidence < 0.70:
			explanation = (
				f"{lookup.component} maps to {lookup.hazard_level} risk because the main risk drivers are "
				f"{risk_drivers}. confidence is {confidence:.2%}, below the operating threshold, so final routing "
				f"should be held for human review. provisional pathway: {lookup.disposal_pathway}."
			)
		else:
			explanation = (
				f"{lookup.component} maps to {lookup.hazard_level} risk because the main risk drivers are "
				f"{risk_drivers}. the material profile is {lookup.material_profile}. recommended action: "
				f"{lookup.disposal_pathway}."
			)
		return RecommendationResult(
			short_recommendation=lookup.disposal_pathway,
			explanation=explanation,
		)

	prompt = (
		f"component: {lookup.component}\n"
		f"hazard: {lookup.hazard_level}\n"
		f"materials: {lookup.material_profile}\n"
		f"confidence: {confidence:.2%}\n"
		f"default pathway: {lookup.disposal_pathway}\n"
		"write a concise disposal recommendation with compliance framing."
	)
	response = llm_fn(prompt)
	return RecommendationResult(
		short_recommendation=lookup.disposal_pathway,
		explanation=response.strip(),
	)


def as_payload(*items) -> dict:
	payload = {}
	for item in items:
		payload.update(asdict(item))
	return payload

