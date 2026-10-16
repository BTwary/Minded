# Stateful Intent / Clarification Patch — Implementation Audit

**Date:** 2026-10-14  
**Base source:** `MindEd_AAOS_M1_approval_hash_fix.zip`  
**Status:** Focused implementation and tests added; **not release-certified**.

## Purpose

Make the existing canonical question contract the authoritative representation across intent interpretation, guided preview, clarification, approval and execution. Clarification choices should update typed contract fields instead of being treated only as transcript text. Preserve the user's material requirements or refuse to execute when this version cannot represent them safely.

## Changes in this patch

- Added `packages/analytics_core/src/intelligence/intent_state.py` for shared guidance normalization, typed intent patch validation, slot-aware freeform interpretation, intent completeness ledger, relative time-scope representation and typed filter evaluation.
- Extended `CanonicalQuestionContract` with slot provenance/confidence, an intent ledger, time-range representation, and numerator/denominator slots. Hash serialization now rejects unsupported values and non-finite floats rather than using `default=str`.
- Updated `guided_investigation.py` to apply checkbox/clarification choices as contract patches, bind the proposal to the canonical contract, expose patch-shaped clarification options, avoid automatic fallback plans when required intent is unresolved, and refuse compound requests until they can be represented as an approved multi-operation plan.
- Updated the API preview, creation and revision paths to normalize guidance consistently and accept intent patches and pending-slot context. Assumption acknowledgements are not treated as analytical instructions.
- Updated the UI/API types to submit explicit patch objects for clarification options and to avoid sending untouched UI defaults as user intent.
- Updated runtime/controller and answer filtering to apply the approved contract's supported filters to the actual analysis population and reuse the approved time range instead of independently reinterpreting the original time phrase.
- Added `tests/independent_release/test_intent_state_contract.py` and added a dedicated required check, `intent_contract_patch_lifecycle`, to `tests/independent_release/manifest.toml`.

## Verification performed in this environment

- `47 passed, 2 deselected` across focused intent-state, reconciliation, plan-consistency, complex-objective and controlled-association suites.
- Canonical serialization audit: `PASS` (189 files scanned).
- Python compilation: `python -m compileall -q packages apps scripts tests` passed against the final Python source tree.
- Manifest TOML parsed successfully; 32 unique required checks are present, including `intent_contract_patch_lifecycle`.
- HTTP preview-to-approval tests: 3 test cases could not initialize because this environment lacks `python-jose` (`jose`) and `duckdb`. This is an environment blocker, not proof those tests pass or a confirmed application defect. They need to run in the pinned project environment.
- Frontend full build/typecheck not run: `apps/web/node_modules` is absent. Syntax transpilation passed for the three changed TypeScript/TSX files; full project typechecking and the production Next.js build still require the project dependencies.
- The full release manifest was **not** run. The source ZIP does not contain the Windows bundle or installer, so packaged-product certification was not possible here.

## Known limitations / non-goals of this patch

1. This is not a universal freeform-language model. The deterministic slot-aware answer parser deliberately supports a bounded set of high-confidence forms and rejects uncertain bindings. A future optional local/cloud language model may propose structured patches, but the schema and validators must remain authoritative.
2. Compound/multi-intent questions are explicitly refused by the guided contract path until the approved contract can persist and validate an operation DAG. The patch does not implement that DAG.
3. Derived ratios requiring a denominator and operations that are not compiled deterministically remain blocked rather than being approximated.
4. The intent ledger currently focuses on material clauses and slots covered by the existing contract/compiler. It does not prove correct interpretation of every conceivable analytical question or dataset.
5. API end-to-end checks, full locked-environment regression, installer build, clean-Windows installation, offline operation and signature verification remain outstanding.

## Release position

This patch improves intent-state consistency and has passing focused unit/regression evidence, but **it is not evidence that MindEd AA-OS 1.0.0 is release-ready**. Before release, run the full required manifest in the pinned environment, run the API end-to-end tests with declared dependencies installed, verify the new contract behavior against the accepted benchmark, regenerate the source-bound attestation from the new gate report, and certify the actual Windows artifact on a clean supported machine.
