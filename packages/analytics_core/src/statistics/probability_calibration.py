"""Empirical probability calibration authority for AA-OS.

Raw Bayesian posteriors are model beliefs.  They become *calibrated probabilities*
only after an empirical calibration profile has been fit on out-of-sample labeled
investigations.  This module implements a dependency-light monotone isotonic map
using the pool-adjacent-violators algorithm (PAVA).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib

import numpy as np


@dataclass(frozen=True)
class CalibrationProfile:
    """Immutable monotone mapping from raw model belief to empirical probability."""

    x: tuple[float, ...]
    y: tuple[float, ...]
    sample_size: int
    scope: str = "unknown"
    method: str = "isotonic_pava"
    model_id: str = "AAOS_BELIEF_V1"
    score_semantics: str = "posterior_probability"
    population_scope: str = "unknown"
    out_of_sample_verified: bool = False
    calibration_source_fingerprint: str = ""
    fitting_protocol: str = ""
    profile_id: str = ""

    def __post_init__(self):
        canonical = {
            "x": [float(v) for v in self.x],
            "y": [float(v) for v in self.y],
            "sample_size": int(self.sample_size),
            "scope": str(self.scope),
            "method": str(self.method),
            "model_id": str(self.model_id),
            "score_semantics": str(self.score_semantics),
            "population_scope": str(self.population_scope),
            "out_of_sample_verified": bool(self.out_of_sample_verified),
            "calibration_source_fingerprint": str(self.calibration_source_fingerprint),
            "fitting_protocol": str(self.fitting_protocol),
        }
        expected = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if self.profile_id and self.profile_id != expected:
            raise ValueError("Calibration profile integrity hash mismatch.")
        object.__setattr__(self, "profile_id", expected)

    @property
    def is_usable(self) -> bool:
        return (
            len(self.x) >= 2
            and self.sample_size >= 20
            and self.scope not in {"", "unknown"}
            and self.model_id not in {"", "unknown"}
            and self.score_semantics == "posterior_probability"
            and self.population_scope not in {"", "unknown"}
            and self.out_of_sample_verified is True
            and len(self.calibration_source_fingerprint) == 64
            and all(ch in "0123456789abcdef" for ch in self.calibration_source_fingerprint.lower())
            and self.fitting_protocol == "labeled_holdout_out_of_sample_v1"
            and all(0.0 <= float(v) <= 1.0 for v in self.x)
            and all(0.0 <= float(v) <= 1.0 for v in self.y)
            and list(self.x) == sorted(self.x)
            and list(self.y) == sorted(self.y)
        )

    def is_authorized_for(
        self,
        *,
        model_id: str,
        scope: str,
        population_scope: str,
    ) -> bool:
        """Return True only when the profile is valid and exactly matches the active runtime context."""
        return bool(
            self.is_usable
            and str(self.model_id) == str(model_id)
            and str(self.scope) == str(scope)
            and str(self.population_scope) == str(population_scope)
        )

    def calibrate(self, probability: float) -> float:
        if not self.is_usable:
            raise ValueError("Calibration profile is not empirically sufficient for probability calibration.")
        p = float(np.clip(probability, 0.0, 1.0))
        return float(np.interp(p, self.x, self.y))

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": list(self.x),
            "y": list(self.y),
            "sample_size": self.sample_size,
            "scope": self.scope,
            "method": self.method,
            "model_id": self.model_id,
            "score_semantics": self.score_semantics,
            "population_scope": self.population_scope,
            "out_of_sample_verified": self.out_of_sample_verified,
            "calibration_source_fingerprint": self.calibration_source_fingerprint,
            "fitting_protocol": self.fitting_protocol,
            "profile_id": self.profile_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationProfile":
        oos_flag = value.get("out_of_sample_verified", False)
        if not isinstance(oos_flag, bool):
            raise ValueError("out_of_sample_verified must be a JSON boolean, not a string or numeric value.")
        return cls(
            x=tuple(float(v) for v in value.get("x", [])),
            y=tuple(float(v) for v in value.get("y", [])),
            sample_size=int(value.get("sample_size", 0)),
            scope=str(value.get("scope", "unknown")),
            method=str(value.get("method", "isotonic_pava")),
            model_id=str(value.get("model_id", "AAOS_BELIEF_V1")),
            score_semantics=str(value.get("score_semantics", "posterior_probability")),
            population_scope=str(value.get("population_scope", "unknown")),
            out_of_sample_verified=oos_flag,
            calibration_source_fingerprint=str(value.get("calibration_source_fingerprint", "")),
            fitting_protocol=str(value.get("fitting_protocol", "")),
            profile_id=str(value.get("profile_id", "")),
        )


def _pava(values: Sequence[float], weights: Sequence[float]) -> list[float]:
    means: list[float] = []
    block_weights: list[float] = []
    for value, weight in zip(values, weights):
        means.append(float(value))
        block_weights.append(float(weight))
        while len(means) >= 2 and means[-2] > means[-1]:
            total_w = block_weights[-2] + block_weights[-1]
            merged = (means[-2] * block_weights[-2] + means[-1] * block_weights[-1]) / total_w
            means[-2:] = [merged]
            block_weights[-2:] = [total_w]
    expanded: list[float] = []
    for mean, weight in zip(means, block_weights):
        expanded.extend([float(mean)] * int(round(weight)))
    return expanded


def fit_isotonic_calibration(
    y_true: Iterable[int | bool],
    probabilities: Iterable[float],
    *,
    scope: str = "unknown",
    min_sample_size: int = 20,
    model_id: str = "AAOS_BELIEF_V1",
    score_semantics: str = "posterior_probability",
    population_scope: str = "unknown",
    out_of_sample_verified: bool = False,
    calibration_source_fingerprint: str = "",
    fitting_protocol: str = "",
) -> CalibrationProfile:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(probabilities), dtype=float)
    if y.shape != p.shape:
        raise ValueError("y_true and probabilities must have identical lengths.")
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], np.clip(p[mask], 0.0, 1.0)
    if len(y) < min_sample_size:
        raise ValueError(f"At least {min_sample_size} labeled predictions are required; got {len(y)}.")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("Calibration labels must be binary 0/1 outcomes.")
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    ys = y[order]
    fitted = _pava(ys.tolist(), [1.0] * len(ys))
    # Compress repeated probability values into monotone knot points.  Ensure
    # endpoints exist so interpolation remains defined for the full [0, 1] range.
    knots: dict[float, list[float]] = {}
    for raw_p, fit_y in zip(ps.tolist(), fitted):
        knots.setdefault(float(raw_p), []).append(float(fit_y))
    x = list(knots.keys())
    yk = [float(np.mean(knots[k])) for k in x]
    # Monotone by construction; clamp to valid probability range.
    yk = [float(np.clip(v, 0.0, 1.0)) for v in yk]
    if x[0] > 0.0:
        x.insert(0, 0.0)
        yk.insert(0, yk[0])
    if x[-1] < 1.0:
        x.append(1.0)
        yk.append(yk[-1])
    return CalibrationProfile(
        tuple(x), tuple(yk), len(y), scope=scope, model_id=model_id,
        score_semantics=score_semantics, population_scope=population_scope,
        out_of_sample_verified=out_of_sample_verified,
        calibration_source_fingerprint=calibration_source_fingerprint,
        fitting_protocol=fitting_protocol,
    )


def load_calibration_profile(path: str | Path | None) -> CalibrationProfile | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as handle:
        return CalibrationProfile.from_dict(json.load(handle))


def save_calibration_profile(profile: CalibrationProfile, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(profile.to_dict(), handle, indent=2, sort_keys=True)
