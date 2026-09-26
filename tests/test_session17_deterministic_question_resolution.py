import pandas as pd

from packages.analytics_core.src.engines.semantic import SemanticResolution
from packages.analytics_core.src.intelligence.universal_question_planner import UniversalQuestionCompiler


def _sem(df, *, target=None, group=None):
    return SemanticResolution(
        primary_dataset_name="orders",
        target_metric_col=target,
        group_dimension_col=group,
        time_col="order_date" if "order_date" in df.columns else None,
        table_grain="order_id",
        available_numeric_cols=[c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])],
        available_categorical_cols=[c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])],
    )


def test_partial_relation_keeps_resolved_target_when_exposure_is_unresolved():
    df = pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "quantity": [1, 2, 3],
            "discount_pct": [0, 10, 20],
        }
    )
    plan = UniversalQuestionCompiler.compile(
        "Did discount percentage affect quantity sold?",
        semantic=_sem(df),
        df=df,
    )
    assert plan.semantics.target_column == "quantity"
    assert plan.semantics.explanatory_columns == []
    assert "Explanatory variable could not be resolved confidently." in plan.unresolved_questions


def test_which_grouping_phrase_binds_group_without_by_keyword():
    df = pd.DataFrame(
        {
            "product_category": ["A", "B", "A"],
            "revenue": [10.0, 30.0, 20.0],
        }
    )
    plan = UniversalQuestionCompiler.compile(
        "Which product category has the highest revenue?",
        semantic=_sem(df),
        df=df,
    )
    assert plan.semantics.target_column == "revenue"
    assert plan.semantics.grouping_columns == ["product_category"]
    assert plan.unresolved_questions == []


def test_multiword_metric_alias_is_not_lost_when_trigger_word_is_average():
    df = pd.DataFrame(
        {
            "aov": [100.0, 150.0, 120.0],
            "region": ["N", "S", "N"],
        }
    )
    plan = UniversalQuestionCompiler.compile(
        "What is the average order value by region?",
        semantic=_sem(df),
        df=df,
    )
    assert plan.semantics.target_column == "aov"
    assert plan.semantics.grouping_columns == ["region"]
    assert plan.unresolved_questions == []
