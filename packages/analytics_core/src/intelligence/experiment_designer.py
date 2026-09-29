"""ExperimentDesignAgent: Pre-flight statistical power and A/B test parameter design."""
import math
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    from statsmodels.stats.power import TTestIndPower
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


@dataclass
class ABDesignSpecification:
    """Complete specification for a prescribed A/B test."""
    required_n_per_arm: int
    total_sample_size: int
    estimated_duration_days: int
    target_mde_pct: float
    baseline_variance: float
    statistical_power: float


class ExperimentDesignAgent:
    """Pre-flight statistical power and sample size calculator for Prescriptive Interventions."""

    @staticmethod
    def calculate_ab_parameters(
        baseline_mean: float,
        baseline_std: float,
        daily_traffic: int,
        target_mde_pct: float = 0.05,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> ABDesignSpecification:
        """Calculates required sample size and duration for an A/B test based on empirical baseline variance."""
        if baseline_std <= 0 or daily_traffic <= 0:
            raise ValueError("Cannot calculate power: zero variance or zero daily traffic.")

        # Calculate Cohen's d (Effect Size)
        absolute_mde = baseline_mean * target_mde_pct
        effect_size = absolute_mde / baseline_std

        if _HAS_STATSMODELS:
            power_analysis = TTestIndPower()
            required_n_per_arm = power_analysis.solve_power(
                effect_size=effect_size,
                alpha=alpha,
                power=power,
                ratio=1.0,
                alternative="two-sided",
            )
        else:
            # High-precision analytical normal approximation
            # Z_{1 - alpha/2} for alpha=0.05 is 1.95996; Z_{power} for 0.80 is 0.84162
            z_alpha = 1.95996 if abs(alpha - 0.05) < 0.01 else 2.57583
            z_power = 0.84162 if abs(power - 0.80) < 0.01 else 1.28155
            required_n_per_arm = 2.0 * ((z_alpha + z_power) / effect_size) ** 2

        n_arm = int(math.ceil(required_n_per_arm))
        total_n = n_arm * 2
        duration_days = int(math.ceil(total_n / daily_traffic))

        return ABDesignSpecification(
            required_n_per_arm=n_arm,
            total_sample_size=total_n,
            estimated_duration_days=duration_days,
            target_mde_pct=float(target_mde_pct),
            baseline_variance=float(baseline_std ** 2),
            statistical_power=float(power),
        )
