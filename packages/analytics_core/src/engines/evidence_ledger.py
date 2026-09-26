"""EvidenceLedger: Canonical immutable empirical ledger for verified analytical claims and provenance."""
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

from packages.analytics_core.src.graph.evidence_identity import compute_evidence_identity

from packages.schemas.src.analysis import (
    EpistemicClaimType,
    CalculationTraceSchema,
    EvidenceLedgerSchema,
    EvidenceRecord,
)


class EvidenceLedger:
    """
    Canonical immutable ledger maintaining all empirical findings underpinning an investigation.
    Enforces the AA-OS Invariant: No claim can appear in a final report unless it is backed
    by a verified record in the EvidenceLedger.
    """

    def __init__(self, investigation_id: str):
        self.investigation_id = investigation_id
        self._records: List[EvidenceRecord] = []

    def record_claim(
        self,
        claim_statement: str,
        claim_type: EpistemicClaimType,
        source_experiment_id: str,
        computation_proof: Dict[str, Any],
        source_datasets: Optional[List[str]] = None,
        source_columns: Optional[List[str]] = None,
        row_count_evaluated: int = 0,
        assumptions: Optional[List[str]] = None,
        verification_status: str = "UNVERIFIED",
        statistical_p_value: Optional[float] = None,
        effect_size: Optional[float] = None,
        uncertainty_range: Optional[List[float]] = None,
        calculation_trace: Optional[Dict[str, Any]] = None,
        evidence_identity: Optional[str] = None,
        identity_parameters: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRecord:
        """Register an empirical claim, deduplicating identical computations.

        The ledger is investigation-scoped. A duplicate computation returns the
        original canonical record instead of creating a second evidentiary weight.
        """
        canonical_identity = evidence_identity
        if canonical_identity and canonical_identity in {
            getattr(r, "evidence_identity", None) for r in self._records
        }:
            return next(r for r in self._records if getattr(r, "evidence_identity", None) == canonical_identity)

        rec_id = f"EV-LEDGER-{len(self._records) + 1:03d}"
        
        # Cryptographic provenance hash of computation proof + statement
        payload = {
            "statement": claim_statement,
            "type": claim_type.value,
            "experiment": source_experiment_id,
            "computation": computation_proof,
            "datasets": source_datasets or [],
            "status": verification_status,
            "calculation_trace": calculation_trace or {},
        }
        prov_hash = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()

        rec = EvidenceRecord(
            evidence_id=rec_id,
            claim_statement=claim_statement,
            claim_type=claim_type,
            source_experiment_id=source_experiment_id,
            source_datasets=source_datasets or [],
            source_columns=source_columns or [],
            row_count_evaluated=row_count_evaluated,
            computation_proof=computation_proof,
            assumptions=assumptions or [],
            verification_status=verification_status,
            statistical_p_value=statistical_p_value,
            effect_size=effect_size,
            uncertainty_range=uncertainty_range,
            calculation_trace=(
                CalculationTraceSchema.model_validate(calculation_trace)
                if calculation_trace else None
            ),
            provenance_hash=prov_hash,
            timestamp=datetime.now(timezone.utc),
        )
        # EvidenceRecord is intentionally backward-compatible; attach the
        # canonical identity as an extra immutable field when the schema permits it.
        if canonical_identity:
            try:
                rec.evidence_identity = canonical_identity
            except Exception:
                pass
        self._records.append(rec)
        return rec

    def get_verified_claims(self) -> List[EvidenceRecord]:
        """Return all claims that passed deterministic secondary verification."""
        return [r for r in self._records if r.verification_status in ["VERIFIED", "PASSED"]]

    def get_all_claims(self) -> List[EvidenceRecord]:
        return list(self._records)

    def to_schema(self) -> EvidenceLedgerSchema:
        verified_cnt = sum(1 for r in self._records if r.verification_status in ["VERIFIED", "PASSED"])
        rejected_cnt = sum(1 for r in self._records if r.verification_status == "FAILED")
        return EvidenceLedgerSchema(
            investigation_id=self.investigation_id,
            records=self._records,
            total_claims=len(self._records),
            verified_claims_count=verified_cnt,
            rejected_claims_count=rejected_cnt,
        )

    def format_evidence_report(self) -> str:
        """Format an evidence-backed factual report strictly from ledger contents."""
        verified = self.get_verified_claims()
        if not verified:
            return "No verified empirical claims established by deterministic evidence."

        lines = [f"### Verified Evidence Ledger ({len(verified)} Empirical Claims)"]
        for r in verified:
            p_val_str = f", p={r.statistical_p_value:.4e}" if r.statistical_p_value is not None else ""
            effect_str = f", Effect Size={r.effect_size:.2f}" if r.effect_size is not None else ""
            lines.append(
                f"- **[{r.claim_type.value}]** {r.claim_statement} "
                f"(Source: `{r.source_experiment_id}`, Status: `{r.verification_status}`{p_val_str}{effect_str}, "
                f"Provenance: `{r.provenance_hash[:10]}...`)"
            )
        return "\n".join(lines)
