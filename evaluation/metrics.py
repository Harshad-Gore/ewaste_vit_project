from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)


@dataclass
class ClassificationMetrics:
	accuracy: float
	macro_f1: float
	weighted_f1: float
	macro_precision: float
	macro_recall: float

	def to_dict(self) -> dict[str, float]:
		return asdict(self)


def compute_classification_metrics(
	y_true: np.ndarray,
	y_pred: np.ndarray,
) -> ClassificationMetrics:
	"""Compute standard aggregate metrics for multi-class classification."""
	return ClassificationMetrics(
		accuracy=float(accuracy_score(y_true, y_pred)),
		macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
		weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
		macro_precision=float(
			precision_score(y_true, y_pred, average="macro", zero_division=0)
		),
		macro_recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
	)


def build_classification_artifacts(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	class_names: list[str],
) -> dict[str, Any]:
	"""Generate detailed report and confusion matrices for experiment outputs."""
	labels = list(range(len(class_names)))
	cm = confusion_matrix(y_true, y_pred, labels=labels)

	# avoid division by zero for classes that might not appear in a split
	row_sums = cm.sum(axis=1, keepdims=True)
	cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

	report = classification_report(
		y_true,
		y_pred,
		labels=labels,
		target_names=class_names,
		output_dict=True,
		zero_division=0,
	)

	return {
		"classification_report": report,
		"confusion_matrix": cm.tolist(),
		"confusion_matrix_normalized": cm_norm.tolist(),
	}


def write_metrics_json(output_path: Path, payload: dict[str, Any]) -> None:
	"""Write metrics payload as pretty JSON."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as fp:
		json.dump(payload, fp, indent=2)

