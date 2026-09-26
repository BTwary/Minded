"""Pure analytical-contract equivalence checks used by independent verification."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class VerificationContract:
    """Identity of the quantity that an independent verifier is permitted to compare."""
    estimand: str
    population: str = ""
    grain: str = ""
    parameterization: str = ""

    def canonical(self) -> tuple[str, str, str, str]:
        return (
            str(self.estimand or "").strip(),
            str(self.population or "").strip(),
            str(self.grain or "").strip(),
            str(self.parameterization or "").strip(),
        )


def compare_verification_contracts(
    primary: VerificationContract,
    secondary: VerificationContract,
) -> tuple[bool, list[str]]:
    """Return whether two numerical results represent the same estimand."""
    fields = ("estimand", "population", "grain", "parameterization")
    mismatches = [
        name
        for name, left, right in zip(fields, primary.canonical(), secondary.canonical())
        if left != right
    ]
    return (not mismatches, mismatches)
