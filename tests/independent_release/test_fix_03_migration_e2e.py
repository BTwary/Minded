"""End-to-end proof that the actual alembic migration file (not just logic
copied into a unit test) correctly upgrades a real pre-Fix-3 database.
"""
import os
import sys
import sqlite3
import tempfile
import uuid

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from alembic.config import Config
from alembic import command


def test_fix3_migration_upgrades_a_real_pre_fix3_database(monkeypatch):
    """Runs the real alembic files end-to-end.

    Historically this file executed at import time. Under the repo's conftest.py the ambient
    DATABASE_URL takes precedence over the URL passed in code (migrations/env.py), so the
    migration ran against a different database than the one inspected and collection failed
    with "no such table: investigations". The test now owns its database explicitly.
    """
    monkeypatch.chdir(REPO_ROOT)
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "e2e_migration_test.sqlite3")

    cfg = Config(os.path.join(REPO_ROOT, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(REPO_ROOT, "migrations"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    print("== upgrading to pre-Fix-3 head (b7e6f1a4c2d9) ==")
    command.upgrade(cfg, "b7e6f1a4c2d9")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cols = [r[1] for r in cur.execute("PRAGMA table_info(hypotheses)").fetchall()]
    assert "canonical_identity" not in cols, f"canonical_identity should not exist yet: {cols}"
    print("confirmed: canonical_identity column does not exist pre-migration")

    inv_id = "INV-E2E-1"
    now = "2026-01-01 00:00:00"
    cur.execute(
        "INSERT INTO investigations (id, project_id, question, status, created_at, updated_at) VALUES (?,?,?,?,?,?)",
        (inv_id, "proj-e2e", "Why did Enterprise unit cost rise?", "RUNNING", now, now),
    )

    survivor_id = f"{inv_id}_HYP-01"
    dup_id = f"{inv_id}_HYP-07"
    cur.execute(
        """INSERT INTO hypotheses (id, investigation_id, hypothesis_code, statement, rationale,
           prior_probability, posterior_probability, belief_state, status,
           is_counter_hypothesis, source_evidence_json, parent_hypotheses_json,
           created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (survivor_id, inv_id, "HYP-01", "Enterprise accounts have higher unit cost.",
         "Enterprise support tickets require more engineering time.",
         0.5, 0.7, "active", "ACTIVE", 0, "[\"EXP-01\"]", "[]", now, now),
    )
    cur.execute(
        """INSERT INTO hypotheses (id, investigation_id, hypothesis_code, statement, rationale,
           prior_probability, posterior_probability, belief_state, status,
           is_counter_hypothesis, source_evidence_json, parent_hypotheses_json,
           created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (dup_id, inv_id, "HYP-07", "Unit cost is elevated for Enterprise customers.",
         "Enterprise support tickets require more engineering time.",
         0.5, 0.65, "active", "ACTIVE", 0, "[\"EXP-07\"]", "[\"HYP-03\"]",
         "2026-01-02 00:00:00", "2026-01-02 00:00:00"),
    )

    pred_survivor = str(uuid.uuid4())
    pred_dup = str(uuid.uuid4())
    cur.execute("INSERT INTO predictions (id, investigation_id, hypothesis_id, statement, created_at) VALUES (?,?,?,?,?)",
                (pred_survivor, inv_id, survivor_id, "Enterprise unit cost > SMB", now))
    cur.execute("INSERT INTO predictions (id, investigation_id, hypothesis_id, statement, created_at) VALUES (?,?,?,?,?)",
                (pred_dup, inv_id, dup_id, "Enterprise unit cost > SMB (round 3)", now))

    exp_survivor = str(uuid.uuid4())
    exp_dup = str(uuid.uuid4())
    cur.execute("INSERT INTO experiments (id, investigation_id, hypothesis_id, tool_name, created_at) VALUES (?,?,?,?,?)",
                (exp_survivor, inv_id, survivor_id, "group_by_mean", now))
    cur.execute("INSERT INTO experiments (id, investigation_id, hypothesis_id, tool_name, created_at) VALUES (?,?,?,?,?)",
                (exp_dup, inv_id, dup_id, "group_by_mean", now))

    ev_survivor = str(uuid.uuid4())
    ev_dup = str(uuid.uuid4())
    cur.execute("INSERT INTO evidence (id, investigation_id, hypothesis_id, statement, created_at) VALUES (?,?,?,?,?)",
                (ev_survivor, inv_id, survivor_id, "Enterprise unit cost measured at 2.1x SMB.", now))
    cur.execute("INSERT INTO evidence (id, investigation_id, hypothesis_id, statement, created_at) VALUES (?,?,?,?,?)",
                (ev_dup, inv_id, dup_id, "Enterprise unit cost measured at 2.0x SMB (round 3).", now))

    bu_survivor = str(uuid.uuid4())
    bu_dup = str(uuid.uuid4())
    cur.execute("""INSERT INTO belief_updates (id, investigation_id, hypothesis_id, prior_probability,
                 likelihood_p, posterior_probability, created_at) VALUES (?,?,?,?,?,?,?)""",
                (bu_survivor, inv_id, survivor_id, 0.5, 0.8, 0.7, now))
    cur.execute("""INSERT INTO belief_updates (id, investigation_id, hypothesis_id, prior_probability,
                 likelihood_p, posterior_probability, created_at) VALUES (?,?,?,?,?,?,?)""",
                (bu_dup, inv_id, dup_id, 0.5, 0.75, 0.65, now))

    other_id = f"{inv_id}_HYP-99"
    cur.execute(
        """INSERT INTO hypotheses (id, investigation_id, hypothesis_code, statement,
           prior_probability, posterior_probability, belief_state, status,
           is_counter_hypothesis, created_at, updated_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (other_id, inv_id, "HYP-99", "Churn is unrelated to unit cost.", 0.3, 0.3, "active", "ACTIVE", 0, now, now),
    )
    counter_id = str(uuid.uuid4())
    cur.execute("INSERT INTO counter_hypotheses (id, primary_hypothesis_id, counter_hypothesis_id, created_at) VALUES (?,?,?,?)",
                (counter_id, dup_id, other_id, now))

    dec_id = str(uuid.uuid4())
    cur.execute("""INSERT INTO decision_recommendations (id, investigation_id, recommendation_id,
                 action_title, action_description, grounded_hypothesis_id, target_metric, created_at)
                 VALUES (?,?,?,?,?,?,?,?)""",
                (dec_id, inv_id, "REC-01", "Rebalance Enterprise pricing",
                 "Adjust Enterprise pricing tier to reflect true unit cost.",
                 dup_id, "unit_cost", now))

    conn.commit()
    conn.close()
    print("seeded pre-Fix-3 duplicate hypotheses + dependent rows")

    print("== running upgrade to head (applying Fix 3 migration) ==")
    command.upgrade(cfg, "head")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cols = [r[1] for r in cur.execute("PRAGMA table_info(hypotheses)").fetchall()]
    assert "canonical_identity" in cols, "canonical_identity column must exist after migration"
    print("confirmed: canonical_identity column now exists")

    remaining = cur.execute("SELECT id, hypothesis_code, canonical_identity, source_evidence_json, parent_hypotheses_json FROM hypotheses WHERE investigation_id=?", (inv_id,)).fetchall()
    print("remaining hypotheses:", remaining)
    remaining_ids = {r[0] for r in remaining}
    assert remaining_ids == {survivor_id, other_id}, f"expected survivor+other only, got {remaining_ids}"
    print("confirmed: duplicate HYP-07 row removed, survivor HYP-01 and unrelated HYP-99 remain")

    survivor_row = [r for r in remaining if r[0] == survivor_id][0]
    assert survivor_row[2], "survivor must have a non-null canonical_identity"
    print("confirmed: survivor has canonical_identity =", survivor_row[2])
    assert "EXP-01" in survivor_row[3] and "EXP-07" in survivor_row[3], f"expected both evidence refs unioned: {survivor_row[3]}"
    print("confirmed: source_evidence_json union preserved on survivor")

    pred_count = cur.execute("SELECT COUNT(*) FROM predictions WHERE hypothesis_id=?", (survivor_id,)).fetchone()[0]
    exp_count = cur.execute("SELECT COUNT(*) FROM experiments WHERE hypothesis_id=?", (survivor_id,)).fetchone()[0]
    ev_count = cur.execute("SELECT COUNT(*) FROM evidence WHERE hypothesis_id=?", (survivor_id,)).fetchone()[0]
    bu_count = cur.execute("SELECT COUNT(*) FROM belief_updates WHERE hypothesis_id=?", (survivor_id,)).fetchone()[0]
    print(f"reparented onto survivor -> predictions={pred_count} experiments={exp_count} evidence={ev_count} belief_updates={bu_count}")
    assert pred_count == 2 and exp_count == 2 and ev_count == 2 and bu_count == 2, "all dependent rows must be reparented, none dropped"

    counter_row = cur.execute("SELECT primary_hypothesis_id, counter_hypothesis_id FROM counter_hypotheses WHERE id=?", (counter_id,)).fetchone()
    assert counter_row[0] == survivor_id, f"counter_hypotheses.primary_hypothesis_id must be repointed: {counter_row}"
    print("confirmed: counter_hypotheses reparented onto survivor")

    dec_row = cur.execute("SELECT grounded_hypothesis_id FROM decision_recommendations WHERE id=?", (dec_id,)).fetchone()
    assert dec_row[0] == survivor_id, f"decision_recommendations.grounded_hypothesis_id must be repointed: {dec_row}"
    print("confirmed: decision_recommendations reparented onto survivor's id")

    # Now prove the constraint is actually live: inserting a second row with the
    # survivor's canonical_identity for the same investigation must fail.
    try:
        cur.execute(
            """INSERT INTO hypotheses (id, investigation_id, hypothesis_code, statement,
               canonical_identity, prior_probability, posterior_probability, belief_state,
               status, is_counter_hypothesis, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f"{inv_id}_HYP-NEW", inv_id, "HYP-NEW", "Yet another rewording of the same claim.",
             survivor_row[2], 0.5, 0.5, "active", "ACTIVE", 0, now, now),
        )
        conn.commit()
        pytest.fail("expected sqlite3.IntegrityError for duplicate canonical_identity, but insert succeeded")
    except sqlite3.IntegrityError as e:
        print("confirmed: DB rejects a second row for the same (investigation_id, canonical_identity):", e)

    conn.close()
    print("\nALL END-TO-END MIGRATION ASSERTIONS PASSED")
