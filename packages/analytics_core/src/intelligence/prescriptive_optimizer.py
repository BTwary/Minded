"""PrescriptiveOptimizer: Solves bounded resource and budget allocation optimization using SciPy Linear Programming."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linprog


@dataclass
class OptimizationResult:
    """Outcome of linear resource/budget optimization."""
    optimal_allocation: Dict[str, float]
    max_expected_return: float
    status: str


class PrescriptiveOptimizer:
    """Prescriptive resource/budget allocation solver using SciPy Linear Programming."""

    @staticmethod
    def optimize_budget_allocation(
        channels: List[str],
        expected_rois: List[float],
        total_budget: float,
        min_bounds: List[float],
        max_bounds: List[float],
    ) -> OptimizationResult:
        """Solves the prescriptive budget allocation problem:
        
        Maximize sum(ROI_i * x_i) subject to sum(x_i) <= Budget and Min_i <= x_i <= Max_i
        """
        # linprog minimizes, so we negate expected_rois to maximize
        c = [-roi for roi in expected_rois]

        # Inequality constraint: sum(x_i) <= total_budget
        A_ub = [np.ones(len(channels))]
        b_ub = [total_budget]

        # Bounds for each channel (min/max spend limits)
        bounds = list(zip(min_bounds, max_bounds))

        # Solve Linear Program with HiGHS
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if res.success:
            allocation = {channels[i]: float(res.x[i]) for i in range(len(channels))}
            max_return = float(-res.fun)
            return OptimizationResult(
                optimal_allocation=allocation,
                max_expected_return=max_return,
                status="OPTIMAL_SOLUTION_FOUND",
            )
        else:
            return OptimizationResult(
                optimal_allocation={},
                max_expected_return=0.0,
                status=f"INFEASIBLE_OR_UNBOUNDED: {res.message}",
            )
