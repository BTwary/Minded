"""Human-reviewable cross-dataset execution artifact.

The reviewer-facing artifact is derived from durable investigation scope,
relational experiment contracts, join-safety outcomes, and independent
verification records. It deliberately distinguishes selected scope from the
datasets actually used by a verified claim.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


def build_cross_dataset_review(
    *,
    requested_dataset_ids: Iterable[str],
    dataset_name_by_id: Mapping[str, str],
    experiments: Iterable[Mapping[str, Any]],
    evidences: Iterable[Mapping[str, Any]],
    verifications: Iterable[Mapping[str, Any]],
    join_events: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    requested_scope_ids = [str(value) for value in requested_dataset_ids]
    selected_scope_count = len(requested_scope_ids)
    dataset_id_by_name = {str(name): str(dataset_id) for dataset_id, name in dataset_name_by_id.items()}
    dataset_usage = {
        dataset_id: {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name_by_id.get(dataset_id),
            "selected": True,
            "used_by_experiments": [],
            "used_in_verified_relational_claim": False,
        }
        for dataset_id in requested_scope_ids
    }

    blocked_join_events = []
    for event in join_events:
        reports = event.get("join_safety_reports") or []
        if any(
            isinstance(report, Mapping)
            and str(report.get("status", "")).upper() not in {"SAFE", "PASS"}
            for report in reports
        ):
            blocked_join_events.append(dict(event))

    evidence_by_experiment: dict[str, list[Mapping[str, Any]]] = {}
    for evidence in evidences:
        experiment_id = evidence.get("experiment_id")
        if experiment_id:
            evidence_by_experiment.setdefault(str(experiment_id), []).append(evidence)

    verification_by_evidence: dict[str, list[Mapping[str, Any]]] = {}
    for verification in verifications:
        evidence_id = verification.get("evidence_id")
        if evidence_id:
            verification_by_evidence.setdefault(str(evidence_id), []).append(verification)

    relational_reviews = []
    verified_relational_count = 0
    for experiment in experiments:
        args = dict(experiment.get("arguments_json") or {})
        plan = args.get("relational_plan")
        if not isinstance(plan, Mapping):
            continue
        hops = [dict(hop) for hop in (plan.get("hops") or []) if isinstance(hop, Mapping)]
        source_tables = set()
        if plan.get("base_table"):
            source_tables.add(str(plan["base_table"]))
        for hop in hops:
            for key in ("left_table", "right_table"):
                if hop.get(key):
                    source_tables.add(str(hop[key]))
        source_tables = sorted(source_tables)

        experiment_id = str(experiment.get("id"))
        ev_rows = evidence_by_experiment.get(experiment_id, [])
        ev_ids = {str(ev.get("id")) for ev in ev_rows if ev.get("id")}
        ver_rows = [
            verification
            for evidence_id in ev_ids
            for verification in verification_by_evidence.get(evidence_id, [])
        ]
        verification_statuses = [str(v.get("status") or "UNVERIFIED").upper() for v in ver_rows]
        verified = bool(
            ev_rows
            and any(status in {"PASSED", "VERIFIED"} for status in verification_statuses)
            and any(str(ev.get("validation_status") or "").lower() == "verified" for ev in ev_rows)
        )
        if verified:
            verified_relational_count += 1

        experiment_code = str(experiment.get("test_code") or experiment_id)
        for table in source_tables:
            dataset_id = dataset_id_by_name.get(table)
            if dataset_id in dataset_usage:
                dataset_usage[dataset_id]["used_by_experiments"].append(experiment_code)
                dataset_usage[dataset_id]["used_in_verified_relational_claim"] = bool(
                    dataset_usage[dataset_id]["used_in_verified_relational_claim"] or verified
                )

        relational_reviews.append(
            {
                "experiment_id": experiment_id,
                "experiment_code": experiment_code,
                "status": experiment.get("status"),
                "source_tables": source_tables,
                "join_hops": hops,
                "sql": args.get("sql"),
                "evidence_ids": sorted(ev_ids),
                "verification_statuses": verification_statuses,
                "independently_verified": verified,
            }
        )

    if selected_scope_count <= 1:
        status = "NOT_CROSS_DATASET"
    elif blocked_join_events:
        status = "BLOCKED_JOIN_SAFETY"
    elif verified_relational_count > 0:
        status = "VERIFIED_CROSS_DATASET"
    elif relational_reviews:
        status = "CROSS_DATASET_EXECUTED_UNVERIFIED"
    else:
        status = "SELECTED_BUT_NOT_USED"

    review_checks = {
        "dataset_scope_recorded": bool(requested_scope_ids),
        "at_least_two_sources_when_relational": (
            all(len(review["source_tables"]) >= 2 for review in relational_reviews)
            if relational_reviews
            else True
        ),
        "join_plan_recorded": (
            all(bool(review["join_hops"]) for review in relational_reviews)
            if relational_reviews
            else True
        ),
        "independent_verification_recorded": (
            verified_relational_count > 0 if relational_reviews else None
        ),
        "selected_scope_fully_used": (
            selected_scope_count <= 1
            or (
                bool(relational_reviews)
                and all(
                    any(
                        dataset_id in {
                            dataset_id_by_name.get(table)
                            for table in review["source_tables"]
                        }
                        for review in relational_reviews
                    )
                    for dataset_id in dataset_usage
                )
            )
        ),
    }

    if status == "VERIFIED_CROSS_DATASET":
        message = (
            "Cross-dataset evidence is independently verified; review the join path "
            "and contributing datasets before accepting the claim."
        )
    elif status in {"SELECTED_BUT_NOT_USED", "CROSS_DATASET_EXECUTED_UNVERIFIED"}:
        message = (
            "Multiple datasets were selected, but no independently verified cross-dataset "
            "claim exists yet. Do not interpret the result as using all selected datasets."
        )
    elif status == "BLOCKED_JOIN_SAFETY":
        message = "A cross-dataset join was blocked by the join-safety gate; no positive cross-dataset claim should be accepted."
    else:
        message = "Single-dataset analysis."

    return {
        "status": status,
        "selected_dataset_count": selected_scope_count,
        "requested_dataset_ids": requested_scope_ids,
        "datasets": list(dataset_usage.values()),
        "relational_experiments": relational_reviews,
        "join_safety_block_count": len(blocked_join_events),
        "review_checks": review_checks,
        "review_message": message,
    }
