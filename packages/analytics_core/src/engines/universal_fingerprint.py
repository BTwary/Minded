"""UniversalFingerprintEngine: Compiles reproducible epistemic and causal state into the MindEd Inter-Agent Protocol (MIAP)."""
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class UniversalAIFingerprint:
    """Standardized MIAP schema for inter-agent validation and downstream LLM consumption."""
    protocol_version: str
    manifest_hash: str
    epistemic_grade: str
    epistemic_vectors: Dict[str, float]
    causal_dag: Dict[str, Any]
    prescriptions: Dict[str, Any]
    downstream_ai_directives: List[str]


class UniversalFingerprintEngine:
    """Compiles analytical investigation state into machine-readable MIAP format."""

    PROTOCOL_VERSION = "aa-os-fingerprint-v1"

    @staticmethod
    def generate_directives(grade: str, causal_level: str, has_prescriptions: bool) -> List[str]:
        """Generates strict behavioral constraints for downstream LLMs/agents."""
        directives = [
            "CONSTRAINT: Treat the `manifest_hash` as the cryptographic anchor of truth. Do not alter underlying data claims.",
            "CONSTRAINT: DO NOT recalculate statistical tests (ANOVA, T-Tests, Bayes). Trust the provided `epistemic_vectors`.",
        ]

        if causal_level == "OBSERVATIONAL_ONLY" or causal_level == "OBSERVATIONAL":
            directives.append(
                "CONSTRAINT: The discovered relationships are strictly OBSERVATIONAL. "
                "DO NOT use causal language (e.g., 'causes', 'drives') without referencing the `valid_adjustment_set`."
            )
        elif causal_level in ("IDENTIFIED", "IDENTIFIED_BACKDOOR"):
            directives.append(
                "ACTION: You may use causal language for the edges listed in `causal_dag.directed_edges` "
                "as they satisfy the Pearl Backdoor Criterion."
            )

        if has_prescriptions:
            directives.append(
                "ACTION: If generating operational policies or budgets, strictly adhere to the mathematical bounds defined in `prescriptions`."
            )

        if grade == "C":
            directives.append(
                "WARNING: Epistemic grade is 'C'. Findings carry high uncertainty; require human analyst validation before operational deployment."
            )

        return directives

    @classmethod
    def compile_fingerprint(
        cls,
        manifest_hash: str,
        epistemic_grade: str,
        epistemic_vectors: Dict[str, float],
        causal_result: Any = None,
        prescriptive_result: Any = None,
        verdict_summary: str = "",
    ) -> str:
        """Compiles the final state into a JSON MIAP string block for the Verdict Engine."""
        causal_level = "OBSERVATIONAL"
        if hasattr(causal_result, "status"):
            causal_level = getattr(causal_result.status, "value", str(causal_result.status))
        elif hasattr(causal_result, "epistemic_level"):
            causal_level = str(getattr(causal_result, "epistemic_level"))

        causal_dag = {}
        if hasattr(causal_result, "nodes"):
            causal_dag = {
                "nodes": getattr(causal_result, "nodes", []),
                "directed_edges": getattr(causal_result, "directed_edges", []),
                "colliders": getattr(causal_result, "colliders", []),
                "valid_adjustment_set": getattr(causal_result, "backdoor_adjustment_set", []),
            }
        elif hasattr(causal_result, "backdoor_adjustment_set"):
            causal_dag = {
                "valid_adjustment_set": getattr(causal_result, "backdoor_adjustment_set", []),
                "is_identifiable": getattr(causal_result, "is_identifiable", False),
            }

        prescriptions = {}
        if prescriptive_result:
            if hasattr(prescriptive_result, "__dataclass_fields__"):
                prescriptions = asdict(prescriptive_result)
            elif isinstance(prescriptive_result, dict):
                prescriptions = prescriptive_result

        directives = cls.generate_directives(
            grade=epistemic_grade,
            causal_level=causal_level,
            has_prescriptions=bool(prescriptions),
        )

        fingerprint = UniversalAIFingerprint(
            protocol_version=cls.PROTOCOL_VERSION,
            manifest_hash=manifest_hash,
            epistemic_grade=epistemic_grade,
            epistemic_vectors=epistemic_vectors,
            causal_dag=causal_dag,
            prescriptions=prescriptions,
            downstream_ai_directives=directives,
        )

        json_payload = json.dumps(asdict(fingerprint), indent=2)
        return f"\n```{cls.PROTOCOL_VERSION}\n{json_payload}\n```\n"
