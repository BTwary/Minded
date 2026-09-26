"""EpistemicCalibrationEngine: Disentangles confidence into 8 calibrated epistemic vectors."""
from dataclasses import dataclass
from typing import Any, Dict, Optional
import os
from packages.analytics_core.src.statistics.probability_calibration import CalibrationProfile, load_calibration_profile
import numpy as np
import pandas as pd


@dataclass
class EpistemicVectors:
    """8 Disentangled Epistemic Vectors replacing monolithic confidence scores."""
    evidence_strength: float       # Effect size and sample coverage (0.0 .. 1.0)
    data_quality: float            # Completeness, null rate, schema stability (0.0 .. 1.0)
    statistical_uncertainty: float # p-value / confidence interval width (0.0 .. 1.0)
    model_uncertainty: float       # Dual-engine agreement / delta (0.0 .. 1.0)
    hypothesis_belief: float       # Exact Bayesian posterior probability (0.0 .. 1.0)
    prediction_uncertainty: float  # Residual error / backtest variance (0.0 .. 1.0)
    causal_uncertainty: float      # Observational confounding bounds (0.0 .. 1.0)
    multiverse_robustness_pct: Optional[float] # Specification curve agreement when applicable (0.0 .. 100.0%)
    overall_epistemic_grade: str = "UNCALIBRATED"
    probability_semantics: str = "MODEL_BASED_BELIEF"
    calibrated_probability: Optional[float] = None
    calibration_status: str = "UNAVAILABLE"
    calibration_profile_id: Optional[str] = None
    calibration_scope: Optional[str] = None
    calibration_model_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class EpistemicCalibrationEngine:
    """Calibrates and calculates explicit multi-vector uncertainty metrics."""

    @staticmethod
    def _resolve_profile(
        profile: Optional[CalibrationProfile] = None,
        *,
        runtime_model_id: Optional[str] = None,
        runtime_scope: Optional[str] = None,
        runtime_population_scope: Optional[str] = None,
    ) -> Optional[CalibrationProfile]:
        """Resolve only an explicitly supplied profile or an environment profile
        that is authorized for the active model/scope.

        Calibration is a claim about a specific model and population. Loading an
        arbitrary local JSON profile must therefore never silently upgrade a raw
        Bayesian belief into an empirically calibrated probability.
        """
        expected_model = runtime_model_id or os.getenv("AAOS_CALIBRATION_MODEL", "AAOS_BELIEF_V1")
        expected_scope = runtime_scope if runtime_scope is not None else os.getenv("AAOS_CALIBRATION_SCOPE")
        expected_population = runtime_population_scope if runtime_population_scope is not None else os.getenv("AAOS_CALIBRATION_POPULATION_SCOPE")

        if profile is not None:
            if not profile.is_usable or not expected_scope or not expected_population:
                return None
            if not profile.is_authorized_for(
                model_id=expected_model, scope=expected_scope, population_scope=expected_population
            ):
                return None
            return profile

        candidate = load_calibration_profile(os.getenv("AAOS_CALIBRATION_PROFILE"))
        if candidate is None or not candidate.is_usable:
            return None
        if not expected_scope or not expected_population:
            return None
        if not candidate.is_authorized_for(
            model_id=expected_model, scope=expected_scope, population_scope=expected_population
        ):
            return None
        return candidate

    @classmethod
    def _apply_calibration(
        cls,
        posterior_probability: float,
        profile: Optional[CalibrationProfile] = None,
        *,
        runtime_model_id: Optional[str] = None,
        runtime_scope: Optional[str] = None,
        runtime_population_scope: Optional[str] = None,
    ):
        prof = cls._resolve_profile(
            profile,
            runtime_model_id=runtime_model_id,
            runtime_scope=runtime_scope,
            runtime_population_scope=runtime_population_scope,
        )
        if prof is not None and prof.is_usable:
            return prof.calibrate(posterior_probability), "CALIBRATED", "EMPIRICALLY_CALIBRATED"
        return None, "UNAVAILABLE", "MODEL_BASED_BELIEF"

    @classmethod
    def calibrate_vectors(
        cls,
        df: pd.DataFrame,
        posterior_probability: float,
        verification_delta_pct: float,
        variance_explained_pct: Optional[float],
        multiverse_robustness_pct: Optional[float],
        calibration_profile: Optional[CalibrationProfile] = None,
        causal_identifiable: Optional[bool] = None,
        runtime_model_id: Optional[str] = None,
        runtime_scope: Optional[str] = None,
        runtime_population_scope: Optional[str] = None,
    ) -> EpistemicVectors:
        null_ratio = df.isnull().mean().mean() if not df.empty else 0.0
        data_quality = float(np.clip(1.0 - null_ratio, 0.0, 1.0))
        if variance_explained_pct is not None:
            ev_strength = float(np.clip(variance_explained_pct / 100.0, 0.0, 1.0))
        else:
            ev_strength = 0.0
        # Verification agreement is a binary admissibility gate, not a confidence
        # score. Model uncertainty measures only disagreement magnitude.
        model_uncertainty = float(np.clip(verification_delta_pct / 100.0, 0.0, 1.0))
        n = max(1, len(df))
        stat_uncertainty = float(np.clip(1.0 / np.sqrt(n), 0.0, 1.0))
        pred_uncertainty = float(np.clip(1.0 - posterior_probability, 0.0, 1.0))
        # Derived from actual causal-identification evidence, matching the
        # semantics already used in calibrate_epistemic_state below. Previously
        # this was an unconditional `0.0 if False else 1.0` tautology, so every
        # investigation reported maximal causal uncertainty regardless of
        # whether identifiability had actually been evaluated or established.
        # When the caller hasn't run the causal gate yet (causal_identifiable
        # is None), stay conservative rather than silently claiming identified.
        causal_uncertainty = 0.0 if causal_identifiable else 1.0
        calibrated_probability, calibration_status, semantics = cls._apply_calibration(
            posterior_probability, calibration_profile,
            runtime_model_id=runtime_model_id, runtime_scope=runtime_scope,
            runtime_population_scope=runtime_population_scope,
        )
        grade = (
            "A_CALIBRATED" if calibrated_probability is not None and calibrated_probability >= 0.80 and model_uncertainty < 0.05 else
            "B_CALIBRATED" if calibrated_probability is not None and calibrated_probability >= 0.65 else
            "UNCALIBRATED"
        )
        return EpistemicVectors(
            evidence_strength=ev_strength, data_quality=data_quality,
            statistical_uncertainty=stat_uncertainty, model_uncertainty=model_uncertainty,
            hypothesis_belief=float(posterior_probability), prediction_uncertainty=pred_uncertainty,
            causal_uncertainty=causal_uncertainty, multiverse_robustness_pct=(float(multiverse_robustness_pct) if multiverse_robustness_pct is not None else None),
            overall_epistemic_grade=grade, probability_semantics=semantics,
            calibrated_probability=calibrated_probability, calibration_status=calibration_status,
            calibration_profile_id=(calibration_profile.profile_id if calibrated_probability is not None and calibration_profile is not None else None),
            calibration_scope=(calibration_profile.scope if calibrated_probability is not None and calibration_profile is not None else None),
            calibration_model_id=(calibration_profile.model_id if calibrated_probability is not None and calibration_profile is not None else None),
        )

    @classmethod
    def calibrate_epistemic_state(
        cls,
        posterior_probability: float,
        evidence_count: int = 1,
        all_verifications_passed: bool = True,
        multiverse_robustness: Optional[float] = 1.0,
        adversarial_survived: bool = True,
        sample_size: int = 100,
        causal_identifiable: bool = False,
        calibration_profile: Optional[CalibrationProfile] = None,
        runtime_model_id: Optional[str] = None,
        runtime_scope: Optional[str] = None,
        runtime_population_scope: Optional[str] = None,
    ) -> EpistemicVectors:
        # Evidence count is descriptive strength, not probability calibration.
        ev_strength = float(np.clip(0.1 * evidence_count, 0.0, 1.0))
        data_quality = 0.0 if sample_size <= 0 else float(np.clip(1.0 - 1.0 / np.sqrt(sample_size), 0.0, 1.0))
        model_uncertainty = 0.0 if all_verifications_passed else 1.0
        stat_uncertainty = float(np.clip(1.0 / np.sqrt(max(1, sample_size)), 0.0, 1.0))
        pred_uncertainty = float(np.clip(1.0 - posterior_probability, 0.0, 1.0))
        causal_uncertainty = 0.0 if causal_identifiable else 1.0
        robustness_pct = (float(multiverse_robustness * 100.0) if multiverse_robustness is not None and multiverse_robustness <= 1.0 else (float(multiverse_robustness) if multiverse_robustness is not None else None))
        calibrated_probability, calibration_status, semantics = cls._apply_calibration(
            posterior_probability, calibration_profile,
            runtime_model_id=runtime_model_id, runtime_scope=runtime_scope,
            runtime_population_scope=runtime_population_scope,
        )
        grade = (
            "A_CALIBRATED" if calibrated_probability is not None and calibrated_probability >= 0.80 and all_verifications_passed else
            "B_CALIBRATED" if calibrated_probability is not None and calibrated_probability >= 0.65 else
            "UNCALIBRATED"
        )
        return EpistemicVectors(
            evidence_strength=ev_strength, data_quality=data_quality, statistical_uncertainty=stat_uncertainty,
            model_uncertainty=model_uncertainty, hypothesis_belief=float(posterior_probability),
            prediction_uncertainty=pred_uncertainty, causal_uncertainty=causal_uncertainty,
            multiverse_robustness_pct=robustness_pct, overall_epistemic_grade=grade,
            probability_semantics=semantics, calibrated_probability=calibrated_probability,
            calibration_status=calibration_status,
            calibration_profile_id=(calibration_profile.profile_id if calibrated_probability is not None and calibration_profile is not None else None),
            calibration_scope=(calibration_profile.scope if calibrated_probability is not None and calibration_profile is not None else None),
            calibration_model_id=(calibration_profile.model_id if calibrated_probability is not None and calibration_profile is not None else None),
        )
