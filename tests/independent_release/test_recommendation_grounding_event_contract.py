from packages.analytics_core.src.governance.recommendation_grounding import evaluate_recommendation_grounding


def test_grounding_contract_requires_verified_support():
    grounded = evaluate_recommendation_grounding(
        ["EV-1"], [], {"EV-1": "VERIFIED"}, {"EV-1": "VERIFIED"}
    )
    assert grounded.status == "GROUNDED"


def test_grounding_contract_rejects_present_but_unverified_support():
    blocked = evaluate_recommendation_grounding(
        ["EV-1"], [], {"EV-1": "VERIFIED"}, {"EV-1": "FAILED"}
    )
    assert blocked.status == "UNSUPPORTED"
