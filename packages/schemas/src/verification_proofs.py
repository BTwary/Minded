"""Compatibility shim for canonical verification proof schemas.

The authoritative definitions live in ``packages.schemas.src.analysis``.
Do not add a second proof schema here.  This module exists only so legacy
imports continue to resolve while all new code uses the canonical classes.
"""
from .analysis import (
    GrainVerificationProof,
    JoinVerificationProof,
    CausalIdentifiabilityProof,
    VerificationResult,
)

# Legacy name retained as an alias. The canonical verification result makes
# the secondary value and delta optional because absence of a comparison is
# not evidence of zero difference.
DualEngineVerificationProof = VerificationResult

__all__ = [
    "GrainVerificationProof",
    "JoinVerificationProof",
    "CausalIdentifiabilityProof",
    "DualEngineVerificationProof",
]
