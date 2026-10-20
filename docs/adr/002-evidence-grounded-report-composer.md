# ADR 002: Evidence-grounded hybrid report composition

- **Status:** Implemented in source; full application acceptance remains pending.
- **Date:** 2026-10-19

## Context

MindEd needs reports that adapt to different analytical questions without maintaining a separate hand-written document for every question, and without allowing free-form language generation to invent conclusions or numbers. Browser answers and exported documents must explain the same calculation and use the same saved result contract.

## Decision

Use a hybrid report composer:

1. A small, extensible set of analysis-family blueprints controls relevant sections: descriptive, comparison, trend, trend comparison, relationship, distribution, forecast, diagnostic, and composition.
2. Narrative claims are composed from saved `analyst_result`, execution records, and verified evidence. The composer does not re-run analysis or inspect raw data.
3. Each claim stores `claim_id`, `family`, `fact_ids`, `source_paths`, `evidence_refs`, numeric literals, validation status, and validation issues.
4. Validation checks that linked facts and source paths exist, evidence references are verified, and every printed number is supported by facts linked to that claim. Claims that fail are withheld from the reader but remain in the audit object.
5. Browser answers and PDF/DOCX exports consume the same report contract. Its narrative and presentation-relevant fields are covered by the report integrity hash.
6. Chart specifications must be built from persisted output values. A chart is omitted when the result does not contain the real observations, bins, quartiles, intervals, or contribution values required to draw it faithfully.

## LLM boundary

No unconstrained LLM rewrite is active in the current implementation. The composer deterministically generates constrained natural-language claims from calculated values. An LLM paraphraser may be introduced later only as a surface-language layer over already validated claims. It must preserve claim IDs and fact/evidence references, may not add new factual content or numeric values, and must pass the same validator before display or export.

## Consequences

- Report structure adapts by analysis family while prose and conclusions vary with actual findings.
- A simple result need not receive every possible section; complex or combined questions can include multiple families.
- Unsupported or insufficiently verified analyses do not receive an asserted factual narrative.
- Narrative generation is reproducible and can be audited independently of a language model.
- Quality depends on upstream methods persisting the source values needed for insights and charts. Missing result detail is reported as a limitation rather than reconstructed from prose.

## Acceptance requirements

- Every displayed claim has at least one linked fact and verified evidence reference.
- No numeric literal appears in a claim unless it is supported by one of that claim's linked facts or documented arithmetic derivation.
- The browser and exports use the same narrative and report contract.
- Tests include fabricated-number rejection, unverified-result withholding, combined analysis families, and rendered/exported evidence references.
- Full release acceptance also requires real API execution, frontend build, and browser-to-export testing. Unit and harness tests alone are not product-level acceptance.
