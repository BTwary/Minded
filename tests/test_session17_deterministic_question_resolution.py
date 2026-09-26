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

def test_money_alias_resolves_revenue_across_safe_customer_region_join():
    orders = pd.DataFrame({"order_id":["O1","O2","O3","O4"],"customer_id":["C1","C2","C1","C3"],"revenue":[100.0,200.0,50.0,75.0],"quantity":[1,2,1,3]})
    customers = pd.DataFrame({"customer_id":["C1","C2","C3"],"region":["North","South","North"]})
    from packages.analytics_core.src.engines.intent import IntentEngine
    from packages.analytics_core.src.engines.semantic import SemanticEngine
    intent = IntentEngine.parse_intent("Which region made the most money?", available_columns=list(dict.fromKeys([*orders.columns,*customers.columns])))
    resolution = SemanticEngine.resolve_schema_static(intent, {"orders":orders,"customers":customers})
    assert resolution.primary_dataset_name == "orders"
    assert resolution.target_metric_col == "revenue"
    assert resolution.group_dimension_col == "region"
    assert resolution.relational_access is not None
    assert resolution.relational_access.base_table == "orders"
    assert resolution.relational_access.joined_table == "customers"
    assert resolution.relational_access.metric_column == "revenue"
    assert resolution.relational_access.dim_column == "region"


def test_explicit_missing_correlation_predictor_cannot_fall_back_to_other_numeric_column():
    df = pd.DataFrame({"quantity":[1,2,3,4],"revenue":[10.0,20.0,30.0,40.0]})
    from packages.analytics_core.src.engines.intent import IntentEngine
    from packages.analytics_core.src.engines.semantic import SemanticEngine
    intent = IntentEngine.parse_intent("Did discount percentage affect quantity sold?", available_columns=list(df.columns))
    resolution = SemanticEngine.resolve_schema_static(intent, {"orders":df})
    assert resolution.target_metric_col == "quantity"
    assert resolution.secondary_metric_col is None
    plan = UniversalQuestionCompiler.compile("Did discount percentage affect quantity sold?", semantic=resolution, df=df)
    assert plan.semantics.target_column == "quantity"
    assert plan.semantics.explanatory_columns == []
    assert any("Could not uniquely resolve all sides" in item for item in plan.unresolved_questions)


def test_ranking_direction_is_preserved_in_canonical_estimand():
    df = pd.DataFrame({"region":["N","S","N","S"],"revenue":[10.0,20.0,30.0,5.0]})
    highest = UniversalQuestionCompiler.compile("Which region had the highest revenue?", semantic=_sem(df,target="revenue",group="region"), df=df)
    least = UniversalQuestionCompiler.compile("Which region had the least revenue?", semantic=_sem(df,target="revenue",group="region"), df=df)
    assert highest.estimand["ranking_direction"] == "DESC"
    assert least.estimand["ranking_direction"] == "ASC"
