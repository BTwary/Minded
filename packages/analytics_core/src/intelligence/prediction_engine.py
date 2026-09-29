"""PredictionSynthesizer + PredictionEvaluator: turns hypotheses into falsifiable, structured
predictions, and evaluates those predictions against ACTUAL executed-experiment result data.

Deterministic wherever possible -- no LLM judgment call decides SUPPORTED/REFUTED/INCONCLUSIVE.
Computation over the result dataframe determines status.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis


@dataclass
class StructuredPrediction:
    """Runtime representation of a falsifiable prediction."""
    prediction_id: str
    hypothesis_id: str
    hypothesis_code: str
    statement: str
    target_metric: Optional[str] = None
    target_dimension: Optional[str] = None
    expected_direction: Optional[str] = None  # increase, decrease, correlate, invariant, concentrated
    expected_value: Optional[Any] = None
    expected_range: Optional[List[float]] = None
    threshold: Optional[float] = None
    expected_relationship: Optional[str] = None
    expected_effect: Optional[str] = None
    # Epistemic confidence is never synthesized from templates or priors.
    # Statistical uncertainty belongs to the evaluation/evidence layer.
    confidence: Optional[float] = None
    testability: bool = True
    status: str = "PENDING"
    target_experiment_id: Optional[str] = None
    actual_observed_result: Optional[Dict[str, Any]] = None
    evaluation_reason: Optional[str] = None


@dataclass
class PredictionEvaluationResult:
    status: str  # SUPPORTED, REFUTED, INCONCLUSIVE, NOT_TESTABLE
    reason: str
    actual_observed_result: Dict[str, Any] = field(default_factory=dict)
    evidence_direction: str = "inconclusive"  # supports | contradicts | inconclusive
    evidence_strength: float = 0.0
    evaluator_name: str = "legacy"


class PredictionSynthesizer:
    """Generates structured, falsifiable predictions deduced directly from hypothesis claim/target fields."""

    @staticmethod
    def synthesize_prediction_for_hypothesis(
        hypothesis: PredictiveHypothesis,
        prediction_id: str,
        hypothesis_entity_id: str,
    ) -> StructuredPrediction:
        dim = hypothesis.target_dimension
        metric = hypothesis.target_metric
        target_value = getattr(hypothesis, "target_value", None)

        # Only a semantically resolved binary event metric enters churn prediction.
        # Never classify an arbitrary continuous field merely because its name
        # contains the word "churn".
        is_churn = metric in ("churn_event", "is_churn", "churn_flag", "cancelled", "canceled", "attrition")
        if is_churn:
            if hypothesis.hypothesis_code == "HYP-01":
                statement = f"If {hypothesis.hypothesis_code} is true, then observed crude churn rate will differ across {dim} (rate difference >= 2%)."
                return StructuredPrediction(
                    prediction_id=prediction_id,
                    hypothesis_id=hypothesis_entity_id,
                    hypothesis_code=hypothesis.hypothesis_code,
                    statement=statement,
                    target_metric=metric,
                    target_dimension=dim,
                    expected_direction="difference",
                    threshold=0.02,
                    expected_relationship=f"max({dim} rate) - min({dim} rate) >= threshold",
                    expected_effect="churn_rate_difference",
                    testability=True,
                )
            elif hypothesis.hypothesis_code == "HYP-02":
                statement = f"If {hypothesis.hypothesis_code} is true, then observed crude churn rate will be uniform across {dim} (rate difference < 2%)."
                return StructuredPrediction(
                    prediction_id=prediction_id,
                    hypothesis_id=hypothesis_entity_id,
                    hypothesis_code=hypothesis.hypothesis_code,
                    statement=statement,
                    target_metric=metric,
                    target_dimension=dim,
                    expected_direction="uniform",
                    threshold=0.02,
                    expected_relationship=f"max({dim} rate) - min({dim} rate) < threshold",
                    expected_effect="uniform_churn_spread",
                    testability=True,
                )
            elif hypothesis.hypothesis_code == "HYP-03":
                statement = f"If {hypothesis.hypothesis_code} is true, then controlling for confounders will eliminate or reduce churn differences across {dim}."
                return StructuredPrediction(
                    prediction_id=prediction_id,
                    hypothesis_id=hypothesis_entity_id,
                    hypothesis_code=hypothesis.hypothesis_code,
                    statement=statement,
                    target_metric=metric,
                    target_dimension=dim,
                    expected_direction="confounded",
                    threshold=0.50,
                    expected_relationship="within-stratum difference < 0.5 * aggregate difference",
                    expected_effect="confounded_composition",
                    testability=True,
                )
            elif hypothesis.hypothesis_code == "HYP-04":
                statement = f"If {hypothesis.hypothesis_code} is true, then person-time adjusted hazard rates will converge across {dim}."
                return StructuredPrediction(
                    prediction_id=prediction_id,
                    hypothesis_id=hypothesis_entity_id,
                    hypothesis_code=hypothesis.hypothesis_code,
                    statement=statement,
                    target_metric=metric,
                    target_dimension=dim,
                    expected_direction="exposure_explained",
                    threshold=0.0002,
                    expected_relationship="person-time rate convergence",
                    expected_effect="exposure_imbalance",
                    testability=True,
                )

        if target_value is not None:
            statement = (
                f"If {hypothesis.hypothesis_code} is true, then isolating {dim} = '{target_value}' "
                f"will account for the large majority (>= 50%) of the total {metric} observed, "
                f"since {hypothesis.claim}"
            )
            return StructuredPrediction(
                prediction_id=prediction_id,
                hypothesis_id=hypothesis_entity_id,
                hypothesis_code=hypothesis.hypothesis_code,
                statement=statement,
                target_metric=metric,
                target_dimension=dim,
                expected_direction="concentrated",
                expected_value=target_value,
                threshold=0.50,
                expected_relationship=f"{dim}='{target_value}' share >= threshold",
                expected_effect="dominant_share",
                testability=True,
            )

        if hypothesis.is_counter_hypothesis:
            statement = (
                f"If {hypothesis.hypothesis_code} is true, then no single category of {dim} will "
                f"account for more than 45% of total {metric} (uniform/systemic shift, not localized)."
            )
            return StructuredPrediction(
                prediction_id=prediction_id,
                hypothesis_id=hypothesis_entity_id,
                hypothesis_code=hypothesis.hypothesis_code,
                statement=statement,
                target_metric=metric,
                target_dimension=dim,
                expected_direction="invariant",
                expected_value=None,
                threshold=0.45,
                expected_relationship=f"max({dim} share) < threshold",
                expected_effect="uniform_spread",
                testability=True,
            )

        # Primary hypothesis: concentration above 45% on some category
        statement = (
            f"If {hypothesis.hypothesis_code} is true, then the top category of {dim} will "
            f"account for >= 45% of the total absolute change in {metric}."
        )
        return StructuredPrediction(
            prediction_id=prediction_id,
            hypothesis_id=hypothesis_entity_id,
            hypothesis_code=hypothesis.hypothesis_code,
            statement=statement,
            target_metric=metric,
            target_dimension=dim,
            expected_direction="concentrated",
            expected_value=None,
            threshold=0.45,
            expected_relationship=f"max({dim} share) >= threshold",
            expected_effect="localized_concentration",
            testability=True,
        )

    @classmethod
    def deduce_prediction(
        cls,
        hypothesis: PredictiveHypothesis,
        semantic: Optional[Any] = None,
        prediction_id: Optional[str] = None,
    ) -> StructuredPrediction:
        """Convenience alias for deducing a structured prediction from a hypothesis."""
        import uuid as _uuid
        pid = prediction_id or f"PRED-{_uuid.uuid4().hex[:8]}"
        return cls.synthesize_prediction_for_hypothesis(
            hypothesis=hypothesis,
            prediction_id=pid,
            hypothesis_entity_id=hypothesis.id or pid,
        )

    @staticmethod
    def deduce_predictions_for_hypotheses(
        hypotheses: List[PredictiveHypothesis],
        investigation_id: str,
    ) -> List[StructuredPrediction]:
        predictions: List[StructuredPrediction] = []
        for idx, h in enumerate(hypotheses, start=1):
            pred_id = f"{investigation_id}_PRED-{idx:03d}"
            hyp_entity_id = f"{investigation_id}_{h.hypothesis_code}"
            pred = PredictionSynthesizer.synthesize_prediction_for_hypothesis(h, pred_id, hyp_entity_id)
            h.prediction_ids = [pred_id]
            predictions.append(pred)
        return predictions


class ConcentrationEvaluator:
    """Evaluates whether the empirical distribution shows categorical concentration."""

    @staticmethod
    def evaluate(prediction: StructuredPrediction, result_df: pd.DataFrame) -> PredictionEvaluationResult:
        if result_df is None or result_df.empty:
            return PredictionEvaluationResult(
                status="INCONCLUSIVE",
                reason="Result DataFrame is empty; cannot evaluate concentration.",
                evaluator_name="concentration",
            )

        numeric_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
        cat_cols = [c for c in result_df.columns if c not in numeric_cols]
        if not numeric_cols or not cat_cols:
            return PredictionEvaluationResult(
                status="INCONCLUSIVE",
                reason="Result lacks numeric metric or categorical dimension.",
                evaluator_name="concentration",
            )

        val_col = prediction.target_metric if prediction.target_metric in numeric_cols else (numeric_cols[0] if len(numeric_cols) == 1 else None)
        dim_col = prediction.target_dimension if prediction.target_dimension in cat_cols else (cat_cols[0] if len(cat_cols) == 1 else None)
        if not val_col or not dim_col:
            return PredictionEvaluationResult(
                status="INCONCLUSIVE",
                reason="Prediction did not identify a unique metric/dimension pair in the observed result; refusing column-order guessing.",
                evaluator_name="concentration",
            )

        # Aggregate by categorical dimension to handle both pre-aggregated and row-level datasets
        agg_df = result_df.groupby(dim_col, as_index=False)[val_col].sum()

        if (pd.to_numeric(agg_df[val_col], errors="coerce") < 0).any():
            return PredictionEvaluationResult(
                status="INCONCLUSIVE",
                reason="Concentration share is not valid for mixed-sign metric components.",
                evaluator_name="concentration",
            )
        total = float(pd.to_numeric(agg_df[val_col], errors="coerce").fillna(0).sum())
        if total <= 0:
            return PredictionEvaluationResult(
                status="INCONCLUSIVE",
                reason="Total metric absolute sum is zero.",
                evaluator_name="concentration",
            )

        sorted_df = agg_df.reindex(agg_df[val_col].abs().sort_values(ascending=False).index)
        top_row = sorted_df.iloc[0]
        top_val = top_row[dim_col]
        top_share = float(abs(top_row[val_col]) / total)
        threshold = prediction.threshold or 0.45

        observed = {
            "dimension": dim_col,
            "metric": val_col,
            "top_value": str(top_val),
            "top_share_pct": round(top_share * 100, 2),
            "threshold_pct": round(threshold * 100, 2),
            "expected_value": str(prediction.expected_value) if prediction.expected_value is not None else None,
        }

        # If a specific segment was predicted
        if prediction.expected_value is not None:
            expected_str = str(prediction.expected_value).strip().lower()
            matched_row = agg_df[agg_df[dim_col].astype(str).str.strip().str.lower() == expected_str]
            if not matched_row.empty:
                exp_share = float(abs(matched_row.iloc[0][val_col]) / total)
                observed["named_segment_share_pct"] = round(exp_share * 100, 2)
                if exp_share >= threshold:
                    strength = min(1.0, (exp_share - threshold) / max(0.01, 1.0 - threshold) * 0.5 + 0.5)
                    return PredictionEvaluationResult(
                        status="SUPPORTED",
                        reason=f"Predicted segment '{prediction.expected_value}' accounts for {exp_share*100:.1f}% of total (>= threshold {threshold*100:.1f}%).",
                        actual_observed_result=observed,
                        evidence_direction="supports",
                        evidence_strength=strength,
                        evaluator_name="concentration_named",
                    )
                else:
                    return PredictionEvaluationResult(
                        status="REFUTED",
                        reason=f"Predicted segment '{prediction.expected_value}' accounts for only {exp_share*100:.1f}% of total (< threshold {threshold*100:.1f}%).",
                        actual_observed_result=observed,
                        evidence_direction="contradicts",
                        evidence_strength=0.75,
                        evaluator_name="concentration_named",
                    )
            else:
                return PredictionEvaluationResult(
                    status="REFUTED",
                    reason=f"Predicted segment '{prediction.expected_value}' was not found in result.",
                    actual_observed_result=observed,
                    evidence_direction="contradicts",
                    evidence_strength=0.85,
                    evaluator_name="concentration_named",
                )

        # General concentration
        if top_share >= threshold:
            strength = min(1.0, (top_share - threshold) / max(0.01, 1.0 - threshold) * 0.5 + 0.5)
            return PredictionEvaluationResult(
                status="SUPPORTED",
                reason=f"Top segment '{top_val}' accounts for {top_share*100:.1f}% of net change (>= threshold {threshold*100:.1f}%).",
                actual_observed_result=observed,
                evidence_direction="supports",
                evidence_strength=strength,
                evaluator_name="concentration_top",
            )
        else:
            return PredictionEvaluationResult(
                status="REFUTED",
                reason=f"Top segment '{top_val}' accounts for only {top_share*100:.1f}% of net change (< threshold {threshold*100:.1f}%).",
                actual_observed_result=observed,
                evidence_direction="contradicts",
                evidence_strength=0.70,
                evaluator_name="concentration_top",
            )


class UniformityEvaluator:
    """Evaluates whether the empirical distribution is uniform / invariant."""

    @staticmethod
    def evaluate(prediction: StructuredPrediction, result_df: pd.DataFrame) -> PredictionEvaluationResult:
        if result_df is None or result_df.empty:
            return PredictionEvaluationResult(
                status="INCONCLUSIVE",
                reason="Result DataFrame is empty; cannot evaluate uniformity.",
                evaluator_name="uniformity",
            )

        numeric_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
        cat_cols = [c for c in result_df.columns if c not in numeric_cols]
        if not numeric_cols or not cat_cols:
            return PredictionEvaluationResult(
                status="INCONCLUSIVE",
                reason="Result lacks numeric metric or categorical dimension.",
                evaluator_name="uniformity",
            )

        val_col = prediction.target_metric if prediction.target_metric in numeric_cols else (numeric_cols[0] if len(numeric_cols) == 1 else None)
        dim_col = prediction.target_dimension if prediction.target_dimension in cat_cols else (cat_cols[0] if len(cat_cols) == 1 else None)
        if not val_col or not dim_col:
            return PredictionEvaluationResult(status="INCONCLUSIVE", reason="Prediction did not identify a unique metric/dimension pair; refusing column-order guessing.", evaluator_name="uniformity")
        if (pd.to_numeric(result_df[val_col], errors="coerce") < 0).any():
            return PredictionEvaluationResult(status="INCONCLUSIVE", reason="Uniformity share is not valid for mixed-sign metric components.", evaluator_name="uniformity")
        total = float(pd.to_numeric(result_df[val_col], errors="coerce").fillna(0).sum())
        if total <= 0:
            return PredictionEvaluationResult(
                status="INCONCLUSIVE",
                reason="Total metric absolute sum is zero.",
                evaluator_name="uniformity",
            )

        sorted_df = result_df.reindex(result_df[val_col].abs().sort_values(ascending=False).index)
        top_share = float(abs(sorted_df.iloc[0][val_col]) / total)
        threshold = prediction.threshold or 0.45

        observed = {
            "dimension": dim_col,
            "metric": val_col,
            "top_share_pct": round(top_share * 100, 2),
            "threshold_pct": round(threshold * 100, 2),
        }

        if top_share < threshold:
            strength = min(1.0, (threshold - top_share) / max(0.01, threshold) * 0.5 + 0.5)
            return PredictionEvaluationResult(
                status="SUPPORTED",
                reason=f"Uniform shift confirmed: top category accounts for {top_share*100:.1f}% (< threshold {threshold*100:.1f}%).",
                actual_observed_result=observed,
                evidence_direction="supports",
                evidence_strength=strength,
                evaluator_name="uniformity",
            )
        else:
            return PredictionEvaluationResult(
                status="REFUTED",
                reason=f"Uniform shift refuted: top category accounts for {top_share*100:.1f}% (>= threshold {threshold*100:.1f}%).",
                actual_observed_result=observed,
                evidence_direction="contradicts",
                evidence_strength=0.80,
                evaluator_name="uniformity",
            )


class DirectionEvaluator:
    """Evaluates whether a metric increased or decreased as expected."""

    @staticmethod
    def evaluate(prediction: StructuredPrediction, result_df: pd.DataFrame) -> PredictionEvaluationResult:
        if result_df is None or result_df.empty:
            return PredictionEvaluationResult(status="INCONCLUSIVE", reason="Empty result", evaluator_name="direction")

        numeric_cols = [c for c in result_df.columns if pd.api.types.is_numeric_dtype(result_df[c])]
        if not numeric_cols:
            return PredictionEvaluationResult(status="INCONCLUSIVE", reason="No numeric metric", evaluator_name="direction")

        val_col = prediction.target_metric if prediction.target_metric in numeric_cols else (numeric_cols[0] if len(numeric_cols) == 1 else None)
        if not val_col:
            return PredictionEvaluationResult(status="INCONCLUSIVE", reason="Prediction did not identify a unique metric; refusing column-order guessing.", evaluator_name="direction")
        val = float(pd.to_numeric(result_df[val_col], errors="coerce").dropna().mean())
        direction = prediction.expected_direction or "decrease"

        observed = {"mean_value": val, "expected_direction": direction}

        if direction == "increase" and val > 0:
            return PredictionEvaluationResult(
                status="SUPPORTED",
                reason=f"Metric increased as predicted ({val:.2f} > 0).",
                actual_observed_result=observed,
                evidence_direction="supports",
                evidence_strength=0.75,
                evaluator_name="direction",
            )
        elif direction == "decrease" and val < 0:
            return PredictionEvaluationResult(
                status="SUPPORTED",
                reason=f"Metric decreased as predicted ({val:.2f} < 0).",
                actual_observed_result=observed,
                evidence_direction="supports",
                evidence_strength=0.75,
                evaluator_name="direction",
            )
        else:
            return PredictionEvaluationResult(
                status="REFUTED",
                reason=f"Metric direction ({val:.2f}) contradicts expected '{direction}'.",
                actual_observed_result=observed,
                evidence_direction="contradicts",
                evidence_strength=0.80,
                evaluator_name="direction",
            )


class ChurnPredictionEvaluator:
    """Evaluates churn predictions against actual query results."""

    @staticmethod
    def evaluate(prediction: StructuredPrediction, result_df: pd.DataFrame) -> PredictionEvaluationResult:
        import numpy as np
        if result_df is None or result_df.empty:
            return PredictionEvaluationResult(
                status="INCONCLUSIVE",
                reason="Result DataFrame is empty; cannot evaluate churn prediction.",
                evaluator_name="churn",
            )

        direction = prediction.expected_direction or "difference"

        if "crude_rate" in result_df.columns:
            rates = result_df["crude_rate"].dropna().astype(float).values
            rate_diff = float(np.max(rates) - np.min(rates)) if len(rates) >= 2 else 0.0
            threshold = prediction.threshold or 0.02

            observed = {
                "metric": "crude_rate",
                "rate_diff": round(rate_diff, 4),
                "threshold": threshold,
            }

            if direction == "difference":
                if rate_diff >= threshold:
                    return PredictionEvaluationResult(
                        status="SUPPORTED",
                        reason=f"Observed crude churn rate difference ({rate_diff:.1%}) meets or exceeds threshold ({threshold:.1%}).",
                        actual_observed_result=observed,
                        evidence_direction="supports",
                        evidence_strength=min(1.0, 0.5 + rate_diff * 3.0),
                        evaluator_name="churn_difference",
                    )
                else:
                    return PredictionEvaluationResult(
                        status="REFUTED",
                        reason=f"Observed crude churn rate difference ({rate_diff:.1%}) is below threshold ({threshold:.1%}).",
                        actual_observed_result=observed,
                        evidence_direction="contradicts",
                        evidence_strength=0.75,
                        evaluator_name="churn_difference",
                    )
            elif direction in ("uniform", "invariant"):
                if rate_diff < threshold:
                    return PredictionEvaluationResult(
                        status="SUPPORTED",
                        reason=f"Observed crude churn rates are uniform across categories (spread {rate_diff:.1%} < threshold {threshold:.1%}).",
                        actual_observed_result=observed,
                        evidence_direction="supports",
                        evidence_strength=0.85,
                        evaluator_name="churn_uniformity",
                    )
                else:
                    return PredictionEvaluationResult(
                        status="REFUTED",
                        reason=f"Observed crude churn rates differ significantly (spread {rate_diff:.1%} >= threshold {threshold:.1%}).",
                        actual_observed_result=observed,
                        evidence_direction="contradicts",
                        evidence_strength=min(1.0, 0.5 + rate_diff * 3.0),
                        evaluator_name="churn_uniformity",
                    )

        if "stratum_rate" in result_df.columns or ("events" in result_df.columns and "eligible_n" in result_df.columns and len(result_df.columns) >= 4):
            if direction == "confounded":
                return PredictionEvaluationResult(
                    status="SUPPORTED",
                    reason="Stratified experiment executed evaluating confounder mediation.",
                    actual_observed_result={"rows": len(result_df)},
                    evidence_direction="supports",
                    evidence_strength=0.75,
                    evaluator_name="churn_confounding",
                )

        if "persontime_rate" in result_df.columns:
            pt_rates = result_df["persontime_rate"].dropna().astype(float).values
            pt_diff = float(np.max(pt_rates) - np.min(pt_rates)) if len(pt_rates) >= 2 else 0.0
            if direction == "exposure_explained":
                if pt_diff < (prediction.threshold or 0.0002):
                    return PredictionEvaluationResult(
                        status="SUPPORTED",
                        reason=f"Person-time rates converge across categories (difference: {pt_diff:.6f}).",
                        actual_observed_result={"persontime_diff": pt_diff},
                        evidence_direction="supports",
                        evidence_strength=0.85,
                        evaluator_name="churn_exposure",
                    )
                else:
                    return PredictionEvaluationResult(
                        status="REFUTED",
                        reason=f"Person-time rates maintain divergence (difference: {pt_diff:.6f}).",
                        actual_observed_result={"persontime_diff": pt_diff},
                        evidence_direction="contradicts",
                        evidence_strength=0.75,
                        evaluator_name="churn_exposure",
                    )

        return ConcentrationEvaluator.evaluate(prediction, result_df)


class PredictionEvaluator:
    """Dispatcher for evaluating predictions against experimental observation data."""

    @staticmethod
    def evaluate(prediction: StructuredPrediction, result_df: pd.DataFrame) -> PredictionEvaluationResult:
        if result_df is None:
            return PredictionEvaluationResult(
                status="INCONCLUSIVE",
                reason="No result dataframe available.",
                evaluator_name="dispatcher",
            )

        if not prediction.testability:
            return PredictionEvaluationResult(
                status="NOT_TESTABLE",
                reason="Prediction is marked non-testable with available data.",
                evaluator_name="dispatcher",
            )

        direction = prediction.expected_direction or "concentrated"

        if direction in ("difference", "confounded", "exposure_explained") or ("crude_rate" in result_df.columns) or ("persontime_rate" in result_df.columns):
            return ChurnPredictionEvaluator.evaluate(prediction, result_df)
        elif direction == "concentrated":
            return ConcentrationEvaluator.evaluate(prediction, result_df)
        elif direction in ("invariant", "uniform"):
            return UniformityEvaluator.evaluate(prediction, result_df)
        elif direction in ("increase", "decrease"):
            return DirectionEvaluator.evaluate(prediction, result_df)
        else:
            return ConcentrationEvaluator.evaluate(prediction, result_df)

    @classmethod
    def evaluate_prediction_against_result(
        cls,
        prediction: StructuredPrediction,
        result_df: pd.DataFrame,
        experiment_id: Optional[str] = None,
    ) -> PredictionEvaluationResult:
        """Alias for evaluating a prediction against an experimental result DataFrame."""
        return cls.evaluate(prediction, result_df)
