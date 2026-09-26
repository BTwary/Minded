"""v20-C2.1 regression suite: deep immutability of the canonical semantic
snapshot.

The external audit of v20-C2 (2026-09-17) correctly identified that
``frozen=True`` on ``CanonicalSemanticResolution`` and its component
dataclasses only blocked reassigning a top-level attribute
(``canonical.outcome = ...``); the mutable list/dict objects those
attributes pointed to (``candidates``, ``columns``,
``available_categorical_candidates``, ``provenance``) could still be
mutated in place (``canonical.outcome.candidates.append("fake_column")``),
silently invalidating the RESOLVED/AMBIGUOUS/UNRESOLVED shape invariants
``ResolvedSemanticField.__post_init__`` had only checked once, at
construction time.

This suite proves each of those mutation paths is now actually blocked,
and that everything still round-trips through JSON (a MappingProxyType or
tuple leaking into a persisted contract's JSON columns would break
serialization silently at write time).
"""
import json
import unittest
from types import MappingProxyType

from packages.schemas.src.semantic_binding import (
    SemanticBinding,
    SemanticBindingSet,
)
from packages.schemas.src.semantic_resolution_contract import (
    CanonicalSemanticResolution,
    ResolvedSemanticField,
    SemanticCandidateSet,
)
from packages.schemas.src.semantic_role import ResolutionStatus, SemanticRole


def _make_field(candidates, value=None, status=ResolutionStatus.AMBIGUOUS):
    return ResolvedSemanticField(role=SemanticRole.GROUPING_DIMENSION, value=value, status=status, candidates=list(candidates))


def _make_unresolved(role=SemanticRole.OUTCOME):
    return ResolvedSemanticField(role=role, value=None, status=ResolutionStatus.UNRESOLVED, candidates=[])


class TestResolvedSemanticFieldDeepImmutability(unittest.TestCase):
    def test_top_level_attribute_reassignment_blocked(self):
        f = _make_field(["a", "b"])
        with self.assertRaises(Exception):
            f.value = "hacked"

    def test_candidates_list_mutation_blocked(self):
        """The exact exploit the audit demonstrated: frozen=True does not
        stop in-place mutation of a mutable attribute's contents."""
        f = ResolvedSemanticField(role=SemanticRole.OUTCOME, value="revenue",
                                   status=ResolutionStatus.RESOLVED, candidates=["revenue"])
        with self.assertRaises(AttributeError):
            f.candidates.append("fake_injected_column")
        # The RESOLVED invariant (candidates == [value]) must still hold.
        self.assertEqual(("revenue",), f.candidates)

    def test_provenance_dict_mutation_blocked(self):
        f = _make_field(["a", "b"])
        with self.assertRaises(TypeError):
            f.provenance["source"] = "spoofed"

    def test_candidates_is_tuple_not_list(self):
        f = _make_field(["a", "b"])
        self.assertIsInstance(f.candidates, tuple)

    def test_provenance_is_read_only_mapping(self):
        f = _make_field(["a", "b"])
        self.assertIsInstance(f.provenance, MappingProxyType)

    def test_mutating_caller_supplied_source_list_after_construction_does_not_affect_field(self):
        """A defensive-copy check: the field must not merely wrap the
        caller's own list (which the caller could still mutate) -- it must
        hold an independent, frozen snapshot taken at construction time."""
        source = ["a", "b"]
        f = _make_field(source)
        source.append("c")
        self.assertEqual(("a", "b"), f.candidates)

    def test_to_dict_is_json_serializable_and_plain(self):
        f = ResolvedSemanticField(role=SemanticRole.OUTCOME, value="revenue",
                                   status=ResolutionStatus.RESOLVED, candidates=["revenue"],
                                   provenance={"source": "test"})
        d = f.to_dict()
        self.assertIsInstance(d["candidates"], list)
        self.assertIsInstance(d["provenance"], dict)
        json.dumps(d)  # must not raise


class TestSemanticCandidateSetDeepImmutability(unittest.TestCase):
    def test_columns_mutation_blocked(self):
        cs = SemanticCandidateSet(role=SemanticRole.EXPLANATORY_VARIABLE, purpose="CONFOUNDER",
                                   columns=["cohort", "tenure_band"])
        with self.assertRaises(AttributeError):
            cs.columns.append("fake_confounder")
        self.assertEqual(("cohort", "tenure_band"), cs.columns)

    def test_provenance_mutation_blocked(self):
        cs = SemanticCandidateSet(role=SemanticRole.EXPLANATORY_VARIABLE, purpose="CONFOUNDER", columns=["a"])
        with self.assertRaises(TypeError):
            cs.provenance["injected"] = "value"

    def test_to_dict_is_json_serializable(self):
        cs = SemanticCandidateSet(role=SemanticRole.EXPLANATORY_VARIABLE, purpose="CONFOUNDER",
                                   columns=["a", "b"], provenance={"source": "test"})
        json.dumps(cs.to_dict())


class TestSemanticBindingSetDeepImmutability(unittest.TestCase):
    def _binding(self, col):
        return SemanticBinding(column=col, table="t", role=SemanticRole.EXPLANATORY_VARIABLE,
                                resolution_status=ResolutionStatus.RESOLVED)

    def test_bindings_container_itself_is_frozen(self):
        bs = SemanticBindingSet(bindings=[self._binding("a")])
        with self.assertRaises(Exception):
            bs.bindings = ()

    def test_bindings_tuple_mutation_blocked(self):
        bs = SemanticBindingSet(bindings=[self._binding("a")])
        with self.assertRaises(AttributeError):
            bs.bindings.append(self._binding("fake_injected"))
        self.assertEqual(1, len(bs.bindings))

    def test_single_binding_provenance_mutation_blocked(self):
        b = SemanticBinding(column="a", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.RESOLVED,
                             provenance={"source": "real"})
        with self.assertRaises(TypeError):
            b.provenance["source"] = "spoofed"

    def test_binding_to_dict_is_json_serializable(self):
        b = SemanticBinding(column="a", table="t", role=SemanticRole.OUTCOME,
                             resolution_status=ResolutionStatus.RESOLVED,
                             provenance={"source": "real"})
        json.dumps(b.to_dict())

    def test_binding_set_still_functions_after_freezing(self):
        """The freeze must not break existing consumer behavior --
        get_by_role/executable_bindings/validate all still work exactly as
        before against a tuple-backed set."""
        bs = SemanticBindingSet(bindings=[self._binding("a"), self._binding("b")])
        self.assertEqual(2, len(bs.get_by_role(SemanticRole.EXPLANATORY_VARIABLE)))
        self.assertEqual(2, len(bs.executable_bindings()))
        self.assertEqual([], bs.validate())


class TestCanonicalSemanticResolutionDeepImmutability(unittest.TestCase):
    def _make_canonical(self):
        return CanonicalSemanticResolution(
            bindings=SemanticBindingSet(bindings=[]),
            outcome=ResolvedSemanticField(role=SemanticRole.OUTCOME, value="revenue",
                                           status=ResolutionStatus.RESOLVED, candidates=["revenue"]),
            dimension=_make_unresolved(SemanticRole.GROUPING_DIMENSION),
            time=_make_unresolved(SemanticRole.TIME_VARIABLE),
            secondary_metric=_make_unresolved(SemanticRole.EXPLANATORY_VARIABLE),
            exposure=_make_unresolved(SemanticRole.EXPLANATORY_VARIABLE),
            censored=_make_unresolved(SemanticRole.EXPLANATORY_VARIABLE),
            confounders=SemanticCandidateSet(role=SemanticRole.EXPLANATORY_VARIABLE, purpose="CONFOUNDER", columns=[]),
            available_categorical_candidates=["region", "plan"],
        )

    def test_available_categorical_candidates_mutation_blocked(self):
        canon = self._make_canonical()
        with self.assertRaises(AttributeError):
            canon.available_categorical_candidates.append("fake_column")
        self.assertEqual(("region", "plan"), canon.available_categorical_candidates)

    def test_nested_outcome_field_still_frozen_through_container(self):
        """Freezing must be genuinely deep -- reaching a nested field
        through the container and mutating IT must also fail, not just
        direct construction-site mutation."""
        canon = self._make_canonical()
        with self.assertRaises(AttributeError):
            canon.outcome.candidates.append("fake_column")
        with self.assertRaises(TypeError):
            canon.confounders.provenance["x"] = "y"

    def test_top_level_reassignment_still_blocked(self):
        canon = self._make_canonical()
        with self.assertRaises(Exception):
            canon.outcome = _make_unresolved()


if __name__ == "__main__":
    unittest.main()
