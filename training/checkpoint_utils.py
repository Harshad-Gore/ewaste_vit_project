from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT_VERSION = 1


def build_rich_checkpoint(
    *,
    arch: str,
    class_names: list[str],
    model_state_dict: dict[str, torch.Tensor],
    config: dict[str, Any],
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "checkpoint_type": "classification_model",
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "arch": arch,
        "class_names": class_names,
        "num_classes": len(class_names),
        "config": config,
        "model_state_dict": model_state_dict,
    }

    if metrics is not None:
        payload["metrics"] = metrics
    if extra:
        payload.update(extra)

    return payload


def extract_model_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    # preferred format from this project
    if isinstance(checkpoint_obj, Mapping) and "model_state_dict" in checkpoint_obj:
        state = checkpoint_obj["model_state_dict"]
        if isinstance(state, Mapping):
            return dict(state)

    # common training-checkpoint variants
    if isinstance(checkpoint_obj, Mapping) and "state_dict" in checkpoint_obj:
        state = checkpoint_obj["state_dict"]
        if isinstance(state, Mapping):
            return dict(state)

    # legacy behavior: raw state_dict saved directly
    if isinstance(checkpoint_obj, Mapping) and checkpoint_obj:
        if all(torch.is_tensor(value) for value in checkpoint_obj.values()):
            return dict(checkpoint_obj)

    raise TypeError(
        "unsupported checkpoint format. expected a raw state_dict or a dict with model_state_dict/state_dict."
    )


def discover_classification_checkpoints(
    classification_dir: Path,
    supported_arches: list[str],
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not classification_dir.exists():
        return out

    for arch in supported_arches:
        direct_candidates = [
            classification_dir / arch / f"{arch}_best.pth",
            classification_dir / f"{arch}_best.pth",
        ]

        selected: Path | None = None
        for candidate in direct_candidates:
            if candidate.exists():
                selected = candidate
                break

        if selected is None:
            recursive_matches = sorted(
                classification_dir.rglob(f"{arch}_best.pth"),
                key=lambda p: (len(p.parts), str(p).lower()),
            )
            if recursive_matches:
                selected = recursive_matches[0]

        if selected is not None:
            out[arch] = selected

    return out


def has_any_classification_checkpoint(classification_dir: Path) -> bool:
    if not classification_dir.exists():
        return False
    return any(classification_dir.rglob("*_best.pth"))


def read_best_model_info(classification_dir: Path) -> dict[str, Any] | None:
    path = classification_dir / "best_model.json"
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError):
        return None

    if isinstance(payload, Mapping):
        return dict(payload)
    return None
