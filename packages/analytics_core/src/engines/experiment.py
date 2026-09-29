"""ExperimentEngine: Generates executable experiment specifications and analytical plans."""
from dataclasses import dataclass, field
from typing import Any, Dict, List
from packages.analytics_core.src.engines.hypothesis import CandidateHypothesis
from packages.analytics_core.src.engines.semantic import SemanticResolution


@dataclass
class ExperimentSpecification:
    """Executable analytical experiment specification."""
    code: str
    hypothesis_code: str
    tool_name: str
    query_sql: str
    description: str
    aggregation_type: str  # SUM, AVG, COUNT, VARIANCE


class ExperimentEngine:
    """Synthesizes executable experiment specifications from hypothesis definitions."""

    @staticmethod
    def design_experiments(
        hypotheses: List[CandidateHypothesis],
        semantic: SemanticResolution,
    ) -> List[ExperimentSpecification]:
        metric = semantic.target_metric_col
        specs: List[ExperimentSpecification] = []

        for idx, hyp in enumerate(hypotheses):
            exp_code = f"EXP-{idx+1:02d}"
            if not hyp.is_counter_hypothesis:
                sql = f"SELECT SUM({metric}) AS total_val, COUNT(*) AS row_cnt FROM data_table"
                agg = "SUM"
                desc = f"Dimensional volume concentration of {metric} under {hyp.code}."
            else:
                sql = f"SELECT AVG({metric}) AS mean_val, COUNT(*) AS row_cnt FROM data_table"
                agg = "AVG"
                desc = f"Uniform distribution baseline verification on {metric} under {hyp.code}."

            specs.append(
                ExperimentSpecification(
                    code=exp_code,
                    hypothesis_code=hyp.code,
                    tool_name="duckdb_sql",
                    query_sql=sql,
                    description=desc,
                    aggregation_type=agg,
                )
            )

        return specs
