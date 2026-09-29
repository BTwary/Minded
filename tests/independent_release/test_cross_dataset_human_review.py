from packages.analytics_core.src.runtime.cross_dataset_review import build_cross_dataset_review


def test_verified_cross_dataset_review_requires_relational_plan_and_independent_verification():
    review = build_cross_dataset_review(
        requested_dataset_ids=["c", "o", "p"],
        dataset_name_by_id={"c": "customers", "o": "orders", "p": "products"},
        experiments=[
            {
                "id": "exp-1",
                "test_code": "EXP-RELATIONAL-H1",
                "status": "EXECUTED",
                "arguments_json": {
                    "relational_plan": {
                        "base_table": "orders",
                        "hops": [
                            {"left_table": "orders", "right_table": "customers", "left_key": "customer_id", "right_key": "customer_id"},
                            {"left_table": "orders", "right_table": "products", "left_key": "product_id", "right_key": "product_id"},
                        ],
                    },
                    "sql": "SELECT customers.segment, SUM(orders.revenue) FROM orders ...",
                },
            }
        ],
        evidences=[{"id": "ev-1", "experiment_id": "exp-1", "validation_status": "verified"}],
        verifications=[{"evidence_id": "ev-1", "status": "PASSED"}],
        join_events=[{"join_safety_reports": [{"status": "SAFE"}, {"status": "SAFE"}]}],
    )
    assert review["status"] == "VERIFIED_CROSS_DATASET"
    assert review["review_checks"]["join_plan_recorded"] is True
    assert review["review_checks"]["independent_verification_recorded"] is True
    assert review["review_checks"]["selected_scope_fully_used"] is True
    assert {d["dataset_name"] for d in review["datasets"] if d["used_by_experiments"]} == {"customers", "orders", "products"}


def test_human_review_does_not_call_selected_scope_used_when_only_one_dataset_contributed():
    review = build_cross_dataset_review(
        requested_dataset_ids=["a", "b"],
        dataset_name_by_id={"a": "sales", "b": "support"},
        experiments=[
            {
                "id": "exp-1",
                "test_code": "EXP-SALES",
                "status": "EXECUTED",
                "arguments_json": {"sql": "SELECT SUM(revenue) FROM sales"},
            }
        ],
        evidences=[],
        verifications=[],
        join_events=[],
    )
    assert review["status"] == "SELECTED_BUT_NOT_USED"
    assert review["review_checks"]["selected_scope_fully_used"] is False


def test_human_review_blocks_on_unsafe_join():
    review = build_cross_dataset_review(
        requested_dataset_ids=["a", "b"],
        dataset_name_by_id={"a": "sales", "b": "customers"},
        experiments=[], evidences=[], verifications=[],
        join_events=[{"join_safety_reports": [{"status": "FANOUT_RISK"}]}],
    )
    assert review["status"] == "BLOCKED_JOIN_SAFETY"
    assert review["join_safety_block_count"] == 1
