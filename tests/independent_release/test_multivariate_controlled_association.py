"""Production-chain unit proofs for explicit multivariable association wording."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from packages.analytics_core.src.engines.intent import IntentEngine
from packages.analytics_core.src.engines.method_selection import MethodSelectionEngine
from packages.analytics_core.src.engines.semantic import SemanticEngine
from packages.analytics_core.src.intelligence.semantic_resolution_builder import build_canonical_semantic_resolution
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler
from packages.analytics_core.src.intelligence.predictive_hypothesis import HypothesisSynthesizer
from packages.schemas.src.semantic_resolution_contract import QuestionRoleProposal


def _df(n=120):
    rng = np.random.RandomState(7)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 2.0 * x1 - 1.5 * x2 + 0.4 * z + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "z": z, "y": y})


def _compile(q):
    df = _df()
    intent = IntentEngine.parse_intent(q, available_columns=list(df.columns))
    semantic = SemanticEngine.resolve_schema_static(intent, {"data_table": df})
    plan = UniversalQuestionCompiler.compile(q, semantic=semantic, df=df)
    roles = QuestionRoleProposal.from_plan_semantics(plan.semantics)
    decision = MethodSelectionEngine.decide(
        intent=intent, semantic=semantic, question=q, primary_df=df, question_roles=roles,
    )
    return df, semantic, plan, roles, decision


def test_control_wording_selects_joint_estimand():
    df, semantic, plan, _, decision = _compile("Does y depend on x1 and x2, controlling for z?")
    assert "compound_question_requires_clause_preserving_analysis" not in plan.unresolved_questions
    assert decision.estimand.predictor_columns == ["x1", "x2", "z"]
    assert decision.estimand.joint_predictors is True
    selected = MethodSelectionEngine.select_for_plan(decision, plan, semantic, None, df)
    assert selected.selected_method_code == "association_multivariate_ols"


def test_joint_hypothesis_uses_one_joint_estimand():
    df, semantic, plan, roles, decision = _compile("Does y depend on x1 and x2, controlling for z?")
    decision = MethodSelectionEngine.select_for_plan(decision, plan, semantic, None, df)
    canonical = build_canonical_semantic_resolution(semantic, "CORRELATION", question_roles=roles)
    hyps = HypothesisSynthesizer.synthesize_competing_hypotheses(
        semantic, "Does y depend on x1 and x2, controlling for z?",
        canonical_semantics=canonical, decision=decision,
    )
    assert len(hyps) == 2
    assert all((h.provenance or {}).get("joint_predictors") == ["x1", "x2", "z"] for h in hyps)
    assert ", ".join(["x1", "x2", "z"]) in hyps[0].secondary_metric


def test_pairwise_question_does_not_upgrade_to_joint_model():
    _, _, _, _, decision = _compile("Is x1 associated with y?")
    assert decision.estimand.joint_predictors is False
    assert decision.selected_method_code != "association_multivariate_ols"
