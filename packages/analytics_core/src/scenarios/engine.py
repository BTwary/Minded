"""Deterministic Causal-Aware Scenario and Sensitivity Engine."""
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class ScenarioType(str, Enum):
    TYPE_A_ARITHMETIC = "Arithmetic Scaling"
    TYPE_B_BUSINESS_ACCOUNTING = "Deterministic Business Logic"
    TYPE_C_PREDICTIVE = "Statistical / Predictive Extrapolation"
    TYPE_D_CAUSAL_INTERVENTION = "Causal Counterfactual Intervention"


class ScenarioSimulator:
    """Rigorous Counterfactual Simulation and Sensitivity Engine."""

    def simulate_discount_cap(
        self,
        df: pd.DataFrame,
        target_discount_cap: float = 0.10,
        filter_col: Optional[str] = None,
        filter_val: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Type B Deterministic Scenario: Cap promotional discounts and recompute ground-truth net revenue and margin."""
        if "revenue" not in df.columns or "discount_rate" not in df.columns:
            return {
                "status": "unavailable",
                "scenario_type": ScenarioType.TYPE_B_BUSINESS_ACCOUNTING.value,
                "reason": "Simulation unavailable: Required columns ('revenue', 'discount_rate') not found in dataset.",
            }

        sim_df = df.copy()
        mask = pd.Series(True, index=sim_df.index)
        if filter_col and filter_val and filter_col in sim_df.columns:
            mask = sim_df[filter_col] == filter_val

        baseline_revenue = float(sim_df["revenue"].sum())
        baseline_avg_discount = float(sim_df.loc[mask, "discount_rate"].mean()) if mask.sum() > 0 else 0.0

        # Calculate unit price / gross value before discount
        # revenue = gross * (1 - discount) => gross = revenue / (1 - discount)
        gross_series = sim_df.loc[mask, "revenue"] / (1.0 - sim_df.loc[mask, "discount_rate"].clip(upper=0.99))
        
        # Apply intervention: discount capped at target_discount_cap
        sim_discounts = sim_df.loc[mask, "discount_rate"].clip(upper=target_discount_cap)
        sim_revenue_series = gross_series * (1.0 - sim_discounts)

        counterfactual_total_revenue = baseline_revenue - float(sim_df.loc[mask, "revenue"].sum()) + float(sim_revenue_series.sum())
        delta_revenue = counterfactual_total_revenue - baseline_revenue
        delta_pct = (delta_revenue / baseline_revenue) * 100.0 if baseline_revenue > 0 else 0.0

        return {
            "status": "computed",
            "scenario_name": f"Discount Optimization (Capped at {int(target_discount_cap * 100)}%)",
            "scenario_type": ScenarioType.TYPE_B_BUSINESS_ACCOUNTING.value,
            "scope": f"{filter_col} = '{filter_val}' ({mask.sum():,} affected rows)" if filter_col else f"Entire dataset ({len(df):,} rows)",
            "baseline_revenue": round(baseline_revenue, 2),
            "projected_revenue": round(counterfactual_total_revenue, 2),
            "net_revenue_expansion": round(delta_revenue, 2),
            "net_growth_pct": round(delta_pct, 2),
            "baseline_avg_discount_pct": round(baseline_avg_discount * 100, 1),
            "target_discount_cap_pct": round(target_discount_cap * 100, 1),
            "uncertainty_note": "Deterministic arithmetic calculation. Assumes customer demand elasticity is inelastic to discount reduction.",
        }

    def simulate_volume_recovery(
        self,
        df: pd.DataFrame,
        dimension_col: str,
        lagging_value: str,
        benchmark_target_growth_pct: float = 15.0,
    ) -> Dict[str, Any]:
        """Type B Deterministic Scenario: Project portfolio recovery if a lagging segment returns to benchmark performance."""
        if dimension_col not in df.columns or "revenue" not in df.columns:
            return {
                "status": "unavailable",
                "scenario_type": ScenarioType.TYPE_B_BUSINESS_ACCOUNTING.value,
                "reason": "Simulation unavailable: Required columns not present.",
            }

        baseline_total = float(df["revenue"].sum())
        lagging_mask = df[dimension_col] == lagging_value
        lagging_baseline = float(df.loc[lagging_mask, "revenue"].sum())

        projected_lagging = lagging_baseline * (1.0 + (benchmark_target_growth_pct / 100.0))
        counterfactual_total = baseline_total - lagging_baseline + projected_lagging
        delta_abs = counterfactual_total - baseline_total
        delta_pct = (delta_abs / baseline_total) * 100.0 if baseline_total > 0 else 0.0

        return {
            "status": "computed",
            "scenario_name": f"Segment Volume Recovery ({lagging_value} +{benchmark_target_growth_pct}%)",
            "scenario_type": ScenarioType.TYPE_B_BUSINESS_ACCOUNTING.value,
            "scope": f"{dimension_col} = '{lagging_value}' ({lagging_mask.sum():,} transactions)",
            "baseline_total_revenue": round(baseline_total, 2),
            "segment_baseline_revenue": round(lagging_baseline, 2),
            "projected_total_revenue": round(counterfactual_total, 2),
            "net_impact_absolute": round(delta_abs, 2),
            "net_impact_percentage": round(delta_pct, 2),
            "uncertainty_note": "Deterministic counterfactual projection based on specified target recovery rate.",
        }
