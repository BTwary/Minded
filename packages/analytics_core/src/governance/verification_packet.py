"""Verification packet: the human verifier's worklist for a finished investigation.

Product contract (see VISION.md section 6): AA-OS does the analytical work; the
human analyst is the *verifier*.  A verifier should never have to re-derive the
investigation, and should never be handed a bare "trust me" verdict.  This module
turns the investigation payload (the dict served by ``GET /investigations/{id}``)
into a short, risk-ranked list of things to check, where every item says

  * what the machine already did and how it came out (``machine_status``),
  * the one specific thing only a human can confirm (``you_confirm``),
  * why it matters, and the evidence / SQL needed to check it.

Design rules (each one is pinned by a test):

  1. Deterministic and offline: no AI, no I/O, stdlib only.  The packet is a pure
     function of the payload, so the same investigation always yields the same
     packet and the same fingerprints.
  2. Unknown is never a pass.  A check that did not run, a payload we cannot
     interpret, or a missing record is surfaced as ``NOT_RUN`` / ``NEEDS_HUMAN``
     and asks for human attention; it is never silently green.
  3. Attention goes where the machine cannot self-verify: the interpretation of
     the question, metric semantics, the legitimacy of a stratifying variable,
     unvalidated high-risk assumptions.  Numbers that two independent engines
     already agree on are offered as optional spot checks, not chores.
  4. Sign-off is bound to the exact result.  Decisions carry the fingerprint of
     the item they were made on; if a re-run changes the item, the old decision
     no longer counts (``stale``).
  5. A human can always object.  Only ``VERIFIED`` is gated; ``REJECTED`` and
     ``NEEDS_REWORK`` are always accepted.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = "1"

# --- vocab -----------------------------------------------------------------
BLOCKER = "BLOCKER"        # verification cannot complete without an explicit human decision + comment
REVIEW = "REVIEW"          # needs human judgment the machine cannot supply; decision required
SPOT_CHECK = "SPOT_CHECK"  # machine-verified; optional human sampling
INFO = "INFO"              # context only; no action

PASSED = "PASSED"
FAILED = "FAILED"
WARNING = "WARNING"
NOT_RUN = "NOT_RUN"
NEEDS_HUMAN = "NEEDS_HUMAN"

CONFIRMED = "CONFIRMED"
REJECTED = "REJECTED"
NEEDS_REWORK = "NEEDS_REWORK"
DECISIONS = (CONFIRMED, REJECTED, NEEDS_REWORK)

SIGNOFF_VERIFIED = "VERIFIED"
SIGNOFF_OUTCOMES = (SIGNOFF_VERIFIED, REJECTED, NEEDS_REWORK)

_PRIORITY_ORDER = {BLOCKER: 0, REVIEW: 1, SPOT_CHECK: 2, INFO: 3}
REQUIRED_PRIORITIES = (BLOCKER, REVIEW)

# Policy thresholds (judgement calls, named so they are visible and adjustable).
ROBUSTNESS_OK_PCT = 80.0     # >= : robust enough to be an optional spot check
ROBUSTNESS_LOW_PCT = 50.0    # <  : a blocker; between the two: needs review
MIN_COMMENT_CHARS = 10

_POSITIVE_ATTACK_FAILURES = {"REFUTED", "CONTRADICTED"}


# --- small helpers ---------------------------------------------------------
def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _sha(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(_canon(obj).encode("utf-8")).hexdigest()[:n]


def _g(d: Any, *keys: str, default: Any = None) -> Any:
    """Safe nested get over dicts (returns ``default`` on any non-dict hop)."""
    cur = d
    for k in keys:
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}g}"
    except (TypeError, ValueError):
        return str(x)


def _item(
    item_id: str,
    category: str,
    priority: str,
    machine_status: str,
    title: str,
    machine_did: str,
    you_confirm: str,
    why_it_matters: str,
    evidence: Optional[Dict[str, Any]] = None,
    reproduce: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    core = {
        "id": item_id,
        "category": category,
        "priority": priority,
        "machine_status": machine_status,
        "title": title,
        "evidence": evidence or {},
    }
    return {
        "id": item_id,
        "category": category,
        "priority": priority,
        "machine_status": machine_status,
        "title": title,
        "machine_did": machine_did,
        "you_confirm": you_confirm,
        "why_it_matters": why_it_matters,
        "evidence": evidence or {},
        "reproduce": reproduce,
        "requires_decision": priority in REQUIRED_PRIORITIES,
        "fingerprint": _sha(core),
    }


# --- item builders ---------------------------------------------------------
def _compared_dimension(inv: Mapping[str, Any]) -> Optional[str]:
    return (
        _g(inv, "multiverse", "dimension")
        or _g(inv, "missingness", "grouping_column")
    )


def _primary_dataset(inv: Mapping[str, Any]) -> Optional[str]:
    return _g(inv, "semantic_world_model", "primary_dataset") or _g(
        inv, "semantic_world_model", "contract", "scope", "dataset"
    )


def _interpretation_items(inv: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    contract = _g(inv, "semantic_world_model", "contract") or {}
    miss = inv.get("missingness") or {}
    dataset = _primary_dataset(inv)
    target = _g(contract, "target", "target") or miss.get("metric_name")
    agg = _g(contract, "target", "aggregation_type") or miss.get("aggregation_type")
    dim = _compared_dimension(inv)
    question = inv.get("question")

    reading = []
    if target:
        reading.append(f"measure = '{target}'" + (f" (combined by {str(agg).upper()})" if agg else ""))
    if dim:
        reading.append(f"compared across '{dim}'")
    if dataset:
        reading.append(f"on dataset '{dataset}'")
    reading_txt = "; ".join(reading) if reading else "the system could not state how it read the question"

    out.append(_item(
        "INTERP-QUESTION", "INTERPRETATION", REVIEW, NEEDS_HUMAN,
        "Does this answer the question you asked?",
        "Resolved the question into a measure, a comparison and a dataset (no AI needed; deterministic).",
        f"You asked: \"{question}\". The system read it as: {reading_txt}. Is that what you meant?",
        "Everything downstream is only as right as this reading; a well-verified answer to the wrong question is still wrong.",
        evidence={
            "question": question,
            "measure": target,
            "aggregation": agg,
            "compared_across": dim,
            "dataset": dataset,
            "claim_type": contract.get("claim_type"),
            "problem_class": contract.get("problem_class"),
            "interpretation_notes": list(contract.get("semantic_interpretations") or []),
        },
    ))

    unresolved = list(contract.get("unresolved_questions") or [])
    if unresolved:
        out.append(_item(
            "INTERP-UNRESOLVED", "INTERPRETATION", BLOCKER, NEEDS_HUMAN,
            "The system left questions about your request unresolved",
            "Flagged the points it could not settle from the data and schema.",
            "Answer each open question below (or confirm the assumption made) before relying on the result.",
            "An unresolved interpretation means the reported number may not be the one you wanted.",
            evidence={"unresolved_questions": unresolved},
        ))

    status = str(miss.get("semantic_resolution_status") or "")
    if status.upper().startswith("UNRESOLVED"):
        ledger_validated = None
        for a in _g(inv, "assumption_ledger", "items", default=[]) or []:
            if str(a.get("statement", "")).startswith("[METRIC_SEMANTICS]"):
                ledger_validated = bool(a.get("validated"))
        out.append(_item(
            "METRIC-AGGREGATION", "INTERPRETATION", BLOCKER, NEEDS_HUMAN,
            "The metric was aggregated by default, not by definition",
            f"No metric definition or schema hint fixed how '{target}' should be combined, so "
            f"{str(agg).upper() if agg else 'a default aggregation'} was applied.",
            f"Is {str(agg).upper() if agg else 'this aggregation'} the right way to combine '{target}' across rows? "
            "(A SUM of a score, rate, price or percentage is usually not meaningful; a MEAN or a defined ratio often is.)",
            "Every number that depends on this metric changes if the aggregation changes.",
            evidence={
                "metric": target,
                "aggregation_used": agg,
                "semantic_resolution_status": status,
                "assumption_ledger_marks_this_validated": ledger_validated,
            },
        ))

    grain = _g(contract, "grain") or {}
    if contract and grain.get("grain_proven") is False:
        out.append(_item(
            "GRAIN", "INTERPRETATION", REVIEW, WARNING,
            "What one row represents was not proven",
            "Looked for a key that uniquely identifies a row; none was proven.",
            "Confirm what one row in the data represents (e.g. one order, one customer-month).",
            "If rows are not at the assumed level, counts and sums are inflated or misattributed.",
            evidence={"grain": grain.get("grain"), "primary_key": grain.get("primary_key"), "grain_proven": False},
        ))
    return out


def _readiness_item(inv: Mapping[str, Any]) -> Dict[str, Any]:
    dr = _g(inv, "semantic_world_model", "data_readiness")
    if not dr:
        return _item(
            "DATA-READINESS", "DATA", REVIEW, NOT_RUN,
            "No data-readiness record was found",
            "Nothing was recorded about the fitness of the data for this question.",
            "Inspect the dataset yourself (missing values, duplicates, units) before relying on the result.",
            "Without a readiness check, data problems are undetected rather than absent.",
            evidence={},
        )
    checks = dr.get("checks") or {}
    miss = {k: v for k, v in (checks.get("missingness") or {}).items() if isinstance(v, (int, float))}
    worst_col, worst_rate = (max(miss.items(), key=lambda kv: kv[1]) if miss else (None, 0.0))
    indicators = {
        k: list(checks.get(k) or [])
        for k in ("selection_bias_indicators", "leakage_indicators", "unit_consistency_indicators", "temporal_issues")
    }
    flagged = {k: v for k, v in indicators.items() if v}
    verdict = str(dr.get("fitness_verdict") or "").upper()
    if verdict == "UNFIT":
        pri, ms = BLOCKER, FAILED
    elif verdict == "CAUTION" or flagged:
        pri, ms = REVIEW, WARNING
    elif verdict == "FIT":
        pri, ms = SPOT_CHECK, PASSED
    else:
        pri, ms = REVIEW, NOT_RUN
    fps = dr.get("source_fingerprints") or {}
    return _item(
        "DATA-READINESS", "DATA", pri, ms,
        f"Data fitness: {verdict or 'UNKNOWN'}",
        f"Profiled {dr.get('row_count')} rows x {dr.get('column_count')} columns: missing values, duplicate rows/keys, "
        "outliers, class imbalance, selection-bias, leakage and unit-consistency indicators.",
        "Confirm this is the dataset and snapshot you intended (compare the fingerprint), and read any flagged risks.",
        "A correct analysis of stale or wrong data is still a wrong answer.",
        evidence={
            "dataset": dr.get("dataset_name"),
            "rows": dr.get("row_count"),
            "columns": dr.get("column_count"),
            "fitness_verdict": verdict or None,
            "source_fingerprints": fps,
            "worst_missing_column": worst_col,
            "worst_missing_rate": worst_rate,
            "duplicate_rows": checks.get("duplicates"),
            "flagged_indicators": flagged,
        },
    )


def _evidence_items(inv: Mapping[str, Any]) -> List[Dict[str, Any]]:
    evs = list(inv.get("evidence") or [])
    ver_by_ev: Dict[str, List[Mapping[str, Any]]] = {}
    for v in inv.get("evidence_verifications") or []:
        ver_by_ev.setdefault(str(v.get("evidence_id")), []).append(v)
    dataset = _primary_dataset(inv)

    if not evs:
        return [_item(
            "EVID-NONE", "EVIDENCE", BLOCKER, NOT_RUN,
            "No numerical evidence was recorded",
            "The investigation finished without any stored experiment result.",
            "Do not rely on the answer until you have produced the supporting numbers yourself.",
            "A verdict with no recorded evidence cannot be audited.",
            evidence={"verdict": inv.get("verdict_type")},
        )]

    out: List[Dict[str, Any]] = []
    verified: List[Dict[str, Any]] = []
    for ev in evs:
        eid = str(ev.get("id"))
        vs = ver_by_ev.get(eid, [])
        ok = str(ev.get("validation_status", "")).upper() == "VERIFIED" and any(
            str(v.get("status", "")).upper() == "VERIFIED" for v in vs
        )
        row = {
            "evidence_id": eid,
            "statement": ev.get("statement"),
            "rows_analyzed": ev.get("row_count_analyzed"),
            "sql": ev.get("sql_executed"),
            "verification": [
                {"primary": v.get("primary_tool"), "secondary": v.get("secondary_tool"),
                 "delta_pct": v.get("observed_delta_pct"), "tolerance": v.get("tolerance_threshold"),
                 "status": v.get("status")} for v in vs
            ],
        }
        if ok:
            verified.append(row)
        else:
            out.append(_item(
                f"EVID-FAIL:{eid}", "EVIDENCE", BLOCKER, FAILED if vs else NOT_RUN,
                "A result was not independently verified",
                "Two engines were supposed to compute this result independently; they did not agree or the check did not run.",
                "Recompute this number yourself and decide whether to trust the claims that rest on it.",
                "An unverified number can carry an arithmetic or query error into the conclusion.",
                evidence={**row, "validation_status": ev.get("validation_status")},
                reproduce={"dataset": dataset, "table_alias": "data_table", "sql": ev.get("sql_executed")},
            ))
    if verified:
        first_sql = next((r["sql"] for r in verified if r.get("sql")), None)
        out.append(_item(
            "EVID-VERIFIED", "EVIDENCE", SPOT_CHECK, PASSED,
            f"{len(verified)} result(s) independently recomputed and agreeing",
            "Each result was computed by DuckDB SQL and recomputed in Polars; the two agreed within tolerance. "
            "(This shows the arithmetic is right, not that the question or the metric is.)",
            f"Optional: run at least one query below in your own tool (table `data_table` is dataset '{dataset}') "
            "and confirm it matches the stated figure.",
            "The cheapest way to earn or lose trust in the system is to reproduce one number yourself.",
            evidence={"results": verified},
            reproduce={"dataset": dataset, "table_alias": "data_table", "sql": first_sql},
        ))
    return out


def _coverage_item(inv: Mapping[str, Any]) -> List[Dict[str, Any]]:
    failed = [s for s in (inv.get("failed_steps") or []) if s]
    if not failed:
        return []
    return [_item(
        "COVERAGE-FAILED-STEPS", "COVERAGE", REVIEW, FAILED,
        f"{len(failed)} planned step(s) did not complete",
        "These steps errored and were skipped; the investigation continued without them.",
        "Decide whether the answer is still sound with these checks missing.",
        "A check that silently did not run looks identical to a check that passed unless it is shown to you.",
        evidence={"failed_steps": failed},
    )]


def _challenge_items(inv: Mapping[str, Any]) -> List[Dict[str, Any]]:
    findings = [f for f in (inv.get("adversarial_findings") or []) if isinstance(f, Mapping)]
    attacks = [f for f in findings if f.get("attack_status")]
    skipped = [f for f in findings if not f.get("attack_status")]
    out: List[Dict[str, Any]] = []

    if not attacks:
        reason = (skipped[-1].get("reason") if skipped else None) or "No adversarial-challenge record exists for this investigation."
        return [_item(
            "CHALLENGE-NONE", "CHALLENGE", INFO, NOT_RUN,
            "The leading explanation was not adversarially challenged",
            "No attempt to break the leading hypothesis was recorded.",
            "If the answer is causal or will drive a decision, challenge it yourself (confounders, subgroups, outliers).",
            "Not being tested is not the same as passing a test.",
            evidence={"reason": reason},
        )]

    for i, a in enumerate(attacks, start=1):
        status = str(a.get("attack_status")).upper()
        tag = f"{a.get('leading_hypothesis') or 'H'}#{i}"
        ev = {
            "attack_status": status,
            "leading_hypothesis": a.get("leading_hypothesis"),
            "mechanism": a.get("attack_mechanism"),
            "what_was_found": a.get("epistemic_impact"),
        }
        repro = {"dataset": _primary_dataset(inv), "table_alias": "data_table",
                 "sql": a.get("discriminating_test_sql")} if a.get("discriminating_test_sql") else None
        rev = _g(a, "details", "simpsons_reversal")
        if status in _POSITIVE_ATTACK_FAILURES:
            if isinstance(rev, Mapping):
                sec = rev.get("secondary_dimension") or "the stratifying variable"
                dim = rev.get("primary_dimension") or "the compared dimension"
                ev["simpsons_reversal"] = dict(rev)
                confirm = (
                    f"The comparison of '{dim}' reverses within '{sec}'. Is '{sec}' a legitimate confounder (something that "
                    f"influences both which '{dim}' group a row is in and the outcome)? If '{sec}' is instead a *consequence* "
                    f"of '{dim}' (a mediator or collider), adjusting for it is wrong and this refutation should be rejected."
                )
                why = ("Only a person who knows how the data was generated can tell a true confounder from a mediator; "
                       "the machine detects the reversal but cannot know which way causation runs.")
            else:
                confirm = "Review the finding below and decide whether it invalidates the headline claim."
                why = "The system found evidence that the leading explanation is wrong."
            out.append(_item(
                f"CHALLENGE-{tag}", "CHALLENGE", BLOCKER, FAILED,
                "The leading explanation was refuted by the system's own challenge",
                "Actively tried to break the leading hypothesis and succeeded.",
                confirm, why, evidence=ev, reproduce=repro,
            ))
        elif status == "WEAKENED":
            out.append(_item(
                f"CHALLENGE-{tag}", "CHALLENGE", REVIEW, WARNING,
                "The leading explanation survived only in weakened form",
                "Actively tried to break the leading hypothesis; it was damaged but not disproved.",
                "Read what was found and decide whether the claim still holds well enough for your purpose.",
                "A weakened claim is a caveat the audience should hear about.",
                evidence=ev, reproduce=repro,
            ))
        elif status == "SURVIVED":
            out.append(_item(
                f"CHALLENGE-{tag}", "CHALLENGE", SPOT_CHECK, PASSED,
                "The leading explanation survived the system's own challenge",
                "Actively tried to break the leading hypothesis (subgroup reversal, outlier leverage, sample imbalance, "
                "selection bias, leakage); no attack succeeded.",
                "Optional: consider a confounder the system could not know about (it only tests columns present in the data).",
                "Survival is only as strong as the attacks that were possible with the available columns.",
                evidence=ev, reproduce=repro,
            ))
        elif status == "NOT_APPLICABLE":
            out.append(_item(
                f"CHALLENGE-{tag}", "CHALLENGE", REVIEW, NOT_RUN,
                "The system could not construct a challenge (this is not a pass)",
                "No attack could be built (e.g. no resolved dimension or metric).",
                "Challenge the claim yourself before relying on it.",
                "Treating 'could not test' as 'passed' is the classic false-confidence failure.",
                evidence=ev,
            ))
        else:
            out.append(_item(
                f"CHALLENGE-{tag}", "CHALLENGE", REVIEW, NEEDS_HUMAN,
                f"Challenge returned an unrecognised status: {status}",
                "The result could not be interpreted automatically.",
                "Inspect the raw finding and decide.",
                "Unknown outcomes must not be treated as passes.",
                evidence=ev,
            ))
    return out


def _robustness_item(inv: Mapping[str, Any]) -> Dict[str, Any]:
    mv = inv.get("multiverse")
    if not isinstance(mv, Mapping) or not mv.get("applicable"):
        reason = (mv or {}).get("reason") or (mv or {}).get("epistemic_summary") or "No robustness analysis was recorded."
        return _item(
            "ROBUSTNESS", "METHOD", INFO, NOT_RUN,
            "Robustness across analysis choices was not tested",
            "The specification-curve analysis did not run for this question.",
            "If the result is sensitive to outlier handling or aggregation, test that yourself.",
            "Researcher degrees of freedom can flip a result; not testing them is not the same as passing.",
            evidence={"reason": reason},
        )
    pct = mv.get("robustness_pct")
    curve = list(mv.get("specification_curve") or [])
    discordant = [c.get("specification_name") for c in curve if not c.get("is_concordant")]
    ev = {
        "robustness_pct": pct,
        "concordant": mv.get("concordant_specifications"),
        "total": mv.get("total_specifications"),
        "disagreeing_specifications": discordant,
        "compared_across": mv.get("dimension"),
        "metric": mv.get("metric"),
    }
    if pct is None:
        pri, ms = REVIEW, NEEDS_HUMAN
    elif pct >= ROBUSTNESS_OK_PCT:
        pri, ms = SPOT_CHECK, PASSED
    elif pct >= ROBUSTNESS_LOW_PCT:
        pri, ms = REVIEW, WARNING
    else:
        pri, ms = BLOCKER, FAILED
    return _item(
        "ROBUSTNESS", "METHOD", pri, ms,
        f"Direction holds in {mv.get('concordant_specifications')}/{mv.get('total_specifications')} analysis variants",
        "Re-ran the comparison under several defensible choices (outlier trimming x aggregation) and checked whether the "
        "direction of the effect stays the same.",
        "For any disagreeing variant, would you consider it a legitimate way to analyse this data? If so, the finding is "
        "less firm than the headline suggests.",
        "Conclusions that depend on one arbitrary analysis choice are fragile.",
        evidence=ev,
    )


def _missingness_item(inv: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    m = inv.get("missingness")
    if not isinstance(m, Mapping) or not m.get("claim_relevant", True):
        return None
    cls = str(m.get("epistemic_classification") or "").upper()
    if not cls:
        return None
    ev = {
        "classification": cls,
        "missingness_rate": m.get("missingness_rate"),
        "ranking_stable_under_worst_case": m.get("ranking_stable"),
        "rationale": m.get("rationale"),
    }
    if cls == "ROBUST":
        return _item(
            "MISSINGNESS", "DATA", INFO, PASSED, "Missing values do not change the conclusion",
            "Bounded the effect of missing values on the headline figure.", "No action needed.",
            "Missing data can bias a result silently.", evidence=ev,
        )
    return _item(
        "MISSINGNESS", "DATA", REVIEW, WARNING, f"Missing values may change the conclusion ({cls})",
        "Bounded the effect of missing values on the headline figure; the conclusion is not stable under the worst case.",
        "Decide whether the missing values are plausibly random; if not, treat the result as provisional.",
        "Structured missingness can create a confident but wrong answer.", evidence=ev,
    )


def _assumption_items(inv: Mapping[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, a in enumerate(_g(inv, "assumption_ledger", "items", default=[]) or []):
        stmt = str(a.get("statement", ""))
        if a.get("validated") or a.get("sensitivity_risk") != "high":
            continue
        if stmt.startswith("[INDEPENDENT_VERIFICATION]"):
            # Whether independent verification actually happened is already
            # reported per result via EVID-*; surfacing it again here would
            # duplicate that signal as a generic, less-actionable assumption
            # item. This is a dedup rule now, not a workaround: the ledger
            # entry itself carries the real post-execution outcome (see
            # AssumptionLedgerEngine.patch_validation in the controller), so
            # an unvalidated entry here is genuine, just already covered.
            continue
        out.append(_item(
            f"ASSUME-{i}", "ASSUMPTION", REVIEW, NEEDS_HUMAN,
            "A high-risk assumption is unvalidated",
            "Listed the assumption and could not validate it from the data.",
            f"Do you accept this assumption? -> {stmt}",
            "If a high-risk assumption is false, the conclusion may not hold.",
            evidence={"assumption": stmt, "sensitivity_risk": a.get("sensitivity_risk")},
        ))
    return out


def _claim_items(inv: Mapping[str, Any]) -> List[Dict[str, Any]]:
    cg = inv.get("claim_gate")
    out: List[Dict[str, Any]] = []
    if not isinstance(cg, Mapping):
        out.append(_item(
            "CLAIM-GATE", "CLAIM", REVIEW, NOT_RUN,
            "No claim-strength check was recorded",
            "The check that limits how strongly the result may be worded did not record an outcome.",
            "Word the result conservatively (association, not cause) unless you have your own basis for more.",
            "Overstated claims are the most common way a correct number misleads.",
        ))
        return out
    outcome = str(cg.get("outcome") or "").upper()
    ev = {
        "outcome": outcome,
        "requested_claim": cg.get("requested_claim"),
        "allowed_claim": cg.get("allowed_claim"),
        "blocked_claim": cg.get("blocked_claim"),
        "reason": cg.get("reason"),
        "recovery_actions": cg.get("recovery_actions"),
        "causal_gate": _g(inv, "causal_status", "gate", "status"),
    }
    if outcome == "REFUSE":
        pri, ms = BLOCKER, FAILED
    elif outcome == "QUALIFIED_ANSWER":
        pri, ms = REVIEW, WARNING
    elif outcome == "ANSWER":
        pri, ms = SPOT_CHECK, PASSED
    else:
        pri, ms = REVIEW, NEEDS_HUMAN
    causal_note = ""
    if str(ev["causal_gate"] or "").upper() == "OBSERVATIONAL_ONLY":
        causal_note = " The result is observational: do not report it as a cause."
    out.append(_item(
        "CLAIM-GATE", "CLAIM", pri, ms,
        f"How strongly the result may be stated: {outcome or 'UNKNOWN'}",
        "Compared the strength of claim requested by the question with what the evidence and study design can support.",
        f"Use no stronger wording than: \"{cg.get('allowed_claim')}\"." + causal_note,
        "The claim you publish is the one people act on.",
        evidence=ev,
    ))
    return out


def _stopping_item(inv: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    positive = str(inv.get("verdict_type") or "").upper() in {"DIAGNOSED", "CONFIRMED", "STATISTICALLY_SIGNIFICANT"}
    if positive and not inv.get("stopping_criteria_met"):
        return _item(
            "STOPPING", "METHOD", REVIEW, WARNING,
            "The investigation stopped without meeting its own convergence criteria",
            "Stopped because of a budget or limit, not because the explanations had converged.",
            "Decide whether more investigation is needed before relying on the leading explanation.",
            "An unfinished investigation can rank explanations wrongly.",
            evidence={"stopping_reason": inv.get("stopping_reason")},
        )
    return None


def _decision_items(inv: Mapping[str, Any]) -> List[Dict[str, Any]]:
    recs = list(inv.get("decision_recommendations") or [])
    if not recs:
        return []
    return [_item(
        "ACTIONS", "DECISION", REVIEW, NEEDS_HUMAN,
        f"{len(recs)} recommended action(s) depend on this result",
        "Derived candidate actions and their expected utility from the verified claim.",
        "Check the preconditions of each action; you own the decision, not the system.",
        "Expected-utility figures inherit every uncertainty above.",
        evidence={"actions": [
            {"title": r.get("action_title"), "preconditions": r.get("required_preconditions"),
             "policy_compliance_passed": r.get("policy_compliance_passed")} for r in recs
        ]},
    )]


# --- packet ----------------------------------------------------------------
def build_verification_packet(inv: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the verifier's worklist from a ``GET /investigations/{id}`` payload."""
    items: List[Dict[str, Any]] = []
    items += _interpretation_items(inv)
    items.append(_readiness_item(inv))
    items += _evidence_items(inv)
    items += _coverage_item(inv)
    items += _challenge_items(inv)
    items.append(_robustness_item(inv))
    m = _missingness_item(inv)
    if m:
        items.append(m)
    items += _assumption_items(inv)
    items += _claim_items(inv)
    s = _stopping_item(inv)
    if s:
        items.append(s)
    items += _decision_items(inv)

    # Stable, attention-first ordering: priority, then insertion order.
    items = [it for _, it in sorted(enumerate(items), key=lambda p: (_PRIORITY_ORDER[p[1]["priority"]], p[0]))]

    ids = [it["id"] for it in items]
    if len(ids) != len(set(ids)):  # defensive: ids are the key for decisions
        seen: Dict[str, int] = {}
        for it in items:
            seen[it["id"]] = seen.get(it["id"], 0) + 1
            if seen[it["id"]] > 1:
                it["id"] = f"{it['id']}~{seen[it['id']]}"

    counts = {p: sum(1 for it in items if it["priority"] == p) for p in _PRIORITY_ORDER}
    machine_checked = sum(1 for it in items if it["machine_status"] == PASSED)
    manifest = inv.get("reproducible_manifest_hash") or _g(inv, "provenance", "manifest_hash")
    packet_fp = _sha({"manifest": manifest, "items": [(it["id"], it["fingerprint"]) for it in items]})
    return {
        "schema_version": SCHEMA_VERSION,
        "investigation_id": inv.get("id"),
        "manifest_hash": manifest,
        "packet_fingerprint": packet_fp,
        "question": inv.get("question"),
        "machine_claim": {
            "verdict": inv.get("verdict_type"),
            "headline": inv.get("direct_answer"),
            "confidence": inv.get("confidence_score"),
        },
        "summary": {
            "total": len(items),
            "blockers": counts[BLOCKER],
            "needs_your_judgment": counts[REVIEW],
            "optional_spot_checks": counts[SPOT_CHECK],
            "context_only": counts[INFO],
            "machine_checked_and_passed": machine_checked,
            "decisions_required": counts[BLOCKER] + counts[REVIEW],
        },
        "items": items,
        "signoff_policy": (
            "VERIFIED requires an explicit decision on every BLOCKER and REVIEW item; a BLOCKER or any non-confirming "
            "decision needs a comment. REJECTED and NEEDS_REWORK can be recorded at any time. Decisions apply only to "
            "the exact result they were made on."
        ),
    }


# --- review ----------------------------------------------------------------
def _by_id(packet: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {it["id"]: it for it in packet.get("items", [])}


def validate_decision(packet: Mapping[str, Any], item_id: str, decision: str, comment: Optional[str]) -> Optional[str]:
    """Return an error string, or ``None`` if the decision may be recorded."""
    item = _by_id(packet).get(item_id)
    if item is None:
        return f"Unknown verification item '{item_id}'."
    if decision not in DECISIONS:
        return f"Decision must be one of {', '.join(DECISIONS)}."
    text = (comment or "").strip()
    if (item["priority"] == BLOCKER or decision != CONFIRMED) and len(text) < MIN_COMMENT_CHARS:
        why = "Confirming a blocker" if decision == CONFIRMED else f"Recording '{decision}'"
        return f"{why} needs a comment of at least {MIN_COMMENT_CHARS} characters explaining your reasoning."
    return None


def evaluate_review(packet: Mapping[str, Any], decisions: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Fold an append-only decision log into the current review state.

    ``decisions`` is oldest-first; the latest decision per item wins.  A decision
    whose ``item_fingerprint`` no longer matches the item is reported as stale
    and does not count.
    """
    items = _by_id(packet)
    latest: Dict[str, Mapping[str, Any]] = {}
    for d in decisions:
        if d.get("item_id") in items:
            latest[d["item_id"]] = d
    per_item: Dict[str, Dict[str, Any]] = {}
    stale: List[str] = []
    for iid, d in latest.items():
        is_stale = d.get("item_fingerprint") != items[iid]["fingerprint"]
        if is_stale:
            stale.append(iid)
        per_item[iid] = {
            "decision": d.get("decision"), "comment": d.get("comment"), "reviewer": d.get("reviewer"),
            "at": d.get("at"), "stale": is_stale,
        }
    valid = {iid: v for iid, v in per_item.items() if not v["stale"]}
    required = [it["id"] for it in packet.get("items", []) if it["requires_decision"]]
    pending = [i for i in required if i not in valid]
    rejected = [i for i, v in valid.items() if v["decision"] == REJECTED]
    rework = [i for i, v in valid.items() if v["decision"] == NEEDS_REWORK]
    confirmed = [i for i, v in valid.items() if v["decision"] == CONFIRMED]

    reasons: List[str] = []
    if pending:
        reasons.append(f"{len(pending)} item(s) still need your decision.")
    if rejected:
        reasons.append(f"You rejected {len(rejected)} item(s).")
    if rework:
        reasons.append(f"You marked {len(rework)} item(s) as needing rework.")
    can_verify = not reasons

    if rejected or rework:
        state = "OBJECTIONS_RAISED"
    elif can_verify:
        state = "READY_TO_SIGN"
    elif valid:
        state = "IN_PROGRESS"
    else:
        state = "NOT_STARTED"
    return {
        "state": state,
        "per_item": per_item,
        "required_item_ids": required,
        "pending": pending,
        "confirmed": confirmed,
        "rejected": rejected,
        "needs_rework": rework,
        "stale": stale,
        "can_verify": can_verify,
        "blocking_reasons": reasons,
    }


def validate_signoff(review: Mapping[str, Any], outcome: str, comment: Optional[str]) -> Optional[str]:
    """Return an error string, or ``None`` if this sign-off may be recorded."""
    if outcome not in SIGNOFF_OUTCOMES:
        return f"Outcome must be one of {', '.join(SIGNOFF_OUTCOMES)}."
    if outcome == SIGNOFF_VERIFIED:
        if not review.get("can_verify"):
            return "Cannot mark VERIFIED: " + " ".join(review.get("blocking_reasons") or ["review is incomplete."])
    elif len((comment or "").strip()) < MIN_COMMENT_CHARS:
        return f"Recording '{outcome}' needs a comment of at least {MIN_COMMENT_CHARS} characters."
    return None


def current_signoff(packet: Mapping[str, Any], signoffs: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Latest sign-off, flagged ``stale`` if the result changed since it was made."""
    if not signoffs:
        return None
    last = dict(signoffs[-1])
    last["stale"] = last.get("packet_fingerprint") != packet.get("packet_fingerprint")
    return last
