"""Item 1 requirement: 'restart -> same identity and evidence recovered'.

This deliberately runs the WRITE side and the READ side in two separate OS
processes talking only through a shared sqlite file on disk -- the writer
process persists hypotheses/evidence and then exits (no shared Python
objects, no in-memory shortcut), and a second, freshly-started process opens
the same file and reconstructs. That is a real process boundary, not an
in-memory simulation of one.

Run: python3 scripts/test_item1_state_reconstruction.py
"""
import os
import subprocess
import sys
import tempfile
import textwrap

DB_PATH = os.path.join(tempfile.gettempdir(), "item1_restart_test.sqlite3")
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

WRITER = textwrap.dedent(f"""
    import sys
    sys.path.insert(0, {REPO_ROOT!r})
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from apps.api.src.models.entities import Base, Investigation, Hypothesis, Evidence
    from packages.analytics_core.src.intelligence.hypothesis_identity import compute_semantic_identity
    from packages.analytics_core.src.intelligence.predictive_hypothesis import PredictiveHypothesis
    from packages.analytics_core.src.runtime.state import InvestigationStateManager

    engine = create_engine("sqlite:///{DB_PATH}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    investigation_id = "INV-RESTART-1"
    session.add(Investigation(id=investigation_id, project_id="proj-1",
                               question="Why did revenue fall in March 2026?", status="RUNNING"))
    session.commit()

    state_mgr = InvestigationStateManager(investigation_id=investigation_id,
                                           question="Why did revenue fall in March 2026?")

    # Two same-batch duplicate emergent hypotheses about Region B, which
    # must consolidate to ONE canonical hypothesis before we ever persist.
    def build(code, claim, mechanism):
        h = PredictiveHypothesis(
            id=f"{{investigation_id}}_{{code}}", hypothesis_code=code, claim=claim, mechanism=mechanism,
            predicted_observables_if_true=["x"], predicted_observables_if_false=["y"],
            falsification_criteria="z", required_assumptions=[],
            prior_probability=0.4, posterior_probability=0.4,
            target_metric="revenue", target_dimension="region", target_value="Region B",
            source_evidence=[f"EXP-{{code}}"],
        )
        h.direction = "decrease"
        h.canonical_identity = compute_semantic_identity(h)
        h.consolidation_log.append({{"event": "created", "hypothesis_code": code, "claim": claim}})
        return h

    from packages.analytics_core.src.intelligence.hypothesis_consolidation import HypothesisConsolidator
    batch = [
        build("HYP-1", "Region B caused the March revenue decline.", "Unequal shift concentrated in Region B partitions."),
        build("HYP-2", "Revenue decline in March was driven by Region B.", "Categorical partitioning across region reveals heterogeneity."),
    ]
    consolidated = HypothesisConsolidator.consolidate_batch(batch)
    assert len(consolidated) == 1, "pre-persist batch consolidation failed"
    survivor = consolidated[0]

    action, canonical_code, resolved = state_mgr.create_hypothesis(survivor)
    assert action == "created"

    entity = Hypothesis(
        id=f"{{investigation_id}}_{{canonical_code}}", investigation_id=investigation_id,
        hypothesis_code=canonical_code, canonical_identity=resolved.canonical_identity,
        statement=resolved.claim, rationale=resolved.mechanism,
        prior_probability=resolved.prior_probability, posterior_probability=resolved.posterior_probability,
        belief_state="ACTIVE", is_counter_hypothesis=False, status="ACTIVE",
        target_metric=resolved.target_metric, target_dimension=resolved.target_dimension,
        target_value=str(resolved.target_value) if resolved.target_value is not None else None,
        mechanism_detail=resolved.mechanism_detail or None,
        source_evidence_json=list(resolved.source_evidence),
        parent_hypotheses_json=list(resolved.parent_hypotheses),
    )
    session.merge(entity)

    ev1 = Evidence(id="EV-1", investigation_id=investigation_id, hypothesis_id=entity.id,
                    statement="Region B accounts for 62% of the March revenue decline.",
                    validation_status="verified", effect_size=0.62)
    ev2 = Evidence(id="EV-2", investigation_id=investigation_id, hypothesis_id=entity.id,
                    statement="Region B segment dispersion is statistically significant.",
                    validation_status="verified", effect_size=0.41)
    session.add(ev1)
    session.add(ev2)
    session.commit()
    session.close()

    print("WRITER_OK", canonical_code, resolved.canonical_identity)
""")

READER = textwrap.dedent(f"""
    import sys
    sys.path.insert(0, {REPO_ROOT!r})
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from apps.api.src.models.entities import Base
    from packages.analytics_core.src.runtime.state_reconstruction import reconstruct_investigation_state

    engine = create_engine("sqlite:///{DB_PATH}")
    Session = sessionmaker(bind=engine)
    session = Session()

    state_mgr = reconstruct_investigation_state("INV-RESTART-1", session,
                                                 question="Why did revenue fall in March 2026?")
    active = state_mgr.get_active_hypotheses()
    print("READER_HYPOTHESIS_COUNT", len(active))
    for h in active:
        print("READER_HYP", h.hypothesis_code, h.canonical_identity,
              sorted(h.supporting_evidence_ids), h.claim)
""")

writer_script = os.path.join(tempfile.gettempdir(), "item1_restart_writer.py")
reader_script = os.path.join(tempfile.gettempdir(), "item1_restart_reader.py")
with open(writer_script, "w") as f:
    f.write(WRITER)
with open(reader_script, "w") as f:
    f.write(READER)

FAILURES = []


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        FAILURES.append(name)


print("--- writer process (persists pre-restart state) ---")
writer_result = subprocess.run([sys.executable, writer_script], capture_output=True, text=True)
print(writer_result.stdout)
if writer_result.returncode != 0:
    print(writer_result.stderr, file=sys.stderr)
check("writer process exited cleanly", writer_result.returncode == 0)
writer_line = next((l for l in writer_result.stdout.splitlines() if l.startswith("WRITER_OK")), "")
check("writer persisted a single canonical hypothesis", writer_line.startswith("WRITER_OK"))
pre_restart_code = writer_line.split()[1] if writer_line else None
pre_restart_identity = writer_line.split()[2] if writer_line else None

print("\n--- reader process (fresh process, restart simulation) ---")
reader_result = subprocess.run([sys.executable, reader_script], capture_output=True, text=True)
print(reader_result.stdout)
if reader_result.returncode != 0:
    print(reader_result.stderr, file=sys.stderr)
check("reader process (separate PID) exited cleanly", reader_result.returncode == 0)

count_line = next((l for l in reader_result.stdout.splitlines() if l.startswith("READER_HYPOTHESIS_COUNT")), "")
check("reconstruction yields exactly 1 hypothesis (no re-duplication)", count_line.strip() == "READER_HYPOTHESIS_COUNT 1")

hyp_line = next((l for l in reader_result.stdout.splitlines() if l.split(None, 1)[:1] == ["READER_HYP"]), "")
parts = hyp_line.split(None, 3)
if len(parts) >= 4:
    _, reconstructed_code, reconstructed_identity, rest = parts
    check("reconstructed hypothesis_code matches pre-restart code", reconstructed_code == pre_restart_code)
    check("reconstructed canonical_identity matches pre-restart identity (recomputed, not stored)",
          reconstructed_identity == pre_restart_identity)
    check("reconstructed hypothesis recovers both evidence contributions",
          "'EV-1'" in rest and "'EV-2'" in rest)
else:
    check("reader printed a reconstructed hypothesis line", False)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL RESTART/RECONSTRUCTION TESTS PASSED")
