"""Post-execution reconciliation of what actually ran against the approved contract.

A positive result (``MATCHED``) requires positive evidence for every operation the contract
fixes. Missing execution evidence is ``NOT_VERIFIABLE``, never ``MATCHED``; contradicting
evidence is ``MISMATCH`` and takes precedence.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def reconcile_execution_with_contract(
    locked_contract: Any,
    executed_metric: Optional[str],
    executed_group: Optional[str],
) -> Dict[str, Any]:
    mismatches: List[str] = []
    unverifiable: List[str] = []
    confirmed: List[str] = []

    target = getattr(locked_contract, "target_column", None)
    if target:
        if executed_metric is None:
            unverifiable.append(f"No executed target recorded; contract requires '{target}'")
        elif executed_metric != target:
            mismatches.append(f"Target metric mismatch: contract '{target}', executed '{executed_metric}'")
        else:
            confirmed.append("target")

    groups = tuple(getattr(locked_contract, "grouping_columns", ()) or ())
    if groups:
        if executed_group is None:
            unverifiable.append(f"No executed grouping recorded; contract requires {list(groups)}")
        elif executed_group not in groups:
            mismatches.append(f"Grouping mismatch: contract {list(groups)}, executed '{executed_group}'")
        elif len(groups) > 1:
            # One reported grouping cannot prove every required grouping was executed.
            unverifiable.append(
                f"Contract requires groupings {list(groups)} but only '{executed_group}' was recorded"
            )
        else:
            confirmed.append("grouping")

    if mismatches:
        status = "MISMATCH"
    elif unverifiable or not confirmed:
        status = "NOT_VERIFIABLE"
    else:
        status = "MATCHED"
    return {
        "status": status,
        "discrepancies": mismatches,
        "unverifiable": unverifiable,
        "confirmed": confirmed,
    }
