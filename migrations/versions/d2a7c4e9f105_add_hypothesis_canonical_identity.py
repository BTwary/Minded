"""Add durable canonical hypothesis identity and consolidate historical duplicates."""
from typing import Sequence, Union
import json
from alembic import op
import sqlalchemy as sa

revision: str = "d2a7c4e9f105"
down_revision: Union[str, None] = "b7e6f1a4c2d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _compute_identity(row):
    from packages.analytics_core.src.intelligence.hypothesis_identity import compute_semantic_identity
    obj = type("PersistedHypothesis", (), {})()
    for k in ("target_metric", "target_dimension", "target_value", "rationale", "statement", "mechanism_detail", "generated_reason"):
        setattr(obj, k, row.get(k) or "")
    obj.is_counter_hypothesis = bool(row.get("is_counter_hypothesis"))
    obj.direction = ""
    obj.temporal_scope = ""
    return compute_semantic_identity(obj)


def _coerce_json_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            return [value]
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _json_union(a, b):
    out = _coerce_json_list(a)
    for item in _coerce_json_list(b):
        if item not in out:
            out.append(item)
    return out


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {c["name"] for c in insp.get_columns("hypotheses")}
    if "canonical_identity" not in cols:
        op.add_column("hypotheses", sa.Column("canonical_identity", sa.String(length=64), nullable=True))

    rows = [dict(r._mapping) for r in bind.execute(sa.text("SELECT * FROM hypotheses ORDER BY created_at ASC, id ASC"))]
    identities = {}
    for row in rows:
        ident = _compute_identity(row)
        bind.execute(sa.text("UPDATE hypotheses SET canonical_identity=:i WHERE id=:id"), {"i": ident, "id": row["id"]})
        identities.setdefault((row["investigation_id"], ident), []).append(row)

    hypothesis_cols = {c["name"] for c in sa.inspect(bind).get_columns("hypotheses")}
    for (_inv_id, _ident), group in identities.items():
        if len(group) <= 1:
            continue
        survivor = group[0]
        for dup in group[1:]:
            # Preserve provenance and counts on the survivor before deleting the duplicate.
            updates = {}
            for field in ("source_evidence_json", "parent_hypotheses_json"):
                if field in hypothesis_cols:
                    merged = _json_union(survivor.get(field), dup.get(field))
                    updates[field] = json.dumps(merged, separators=(",", ":"))
            for field in ("supporting_prediction_count", "refuted_prediction_count", "unresolved_prediction_count"):
                if field in hypothesis_cols:
                    updates[field] = int(survivor.get(field) or 0) + int(dup.get(field) or 0)
            if updates:
                sets = ", ".join(f"{k}=:{k}" for k in updates)
                updates["id"] = survivor["id"]
                # SQLite/Postgres both accept JSON text through bound values here.
                bind.execute(sa.text(f"UPDATE hypotheses SET {sets} WHERE id=:id"), updates)

            # Reparent every known scientific dependency.
            for table, col in (("predictions", "hypothesis_id"), ("experiments", "hypothesis_id"), ("evidence", "hypothesis_id"), ("belief_updates", "hypothesis_id")):
                if table in insp.get_table_names() and col in {c["name"] for c in sa.inspect(bind).get_columns(table)}:
                    bind.execute(sa.text(f"UPDATE {table} SET {col}=:surv WHERE {col}=:dup"), {"surv": survivor["id"], "dup": dup["id"]})

            if "counter_hypotheses" in insp.get_table_names():
                cc = {c["name"] for c in sa.inspect(bind).get_columns("counter_hypotheses")}
                if "primary_hypothesis_id" in cc:
                    bind.execute(sa.text("UPDATE counter_hypotheses SET primary_hypothesis_id=:surv WHERE primary_hypothesis_id=:dup"), {"surv": survivor["id"], "dup": dup["id"]})
                if "counter_hypothesis_id" in cc:
                    bind.execute(sa.text("UPDATE counter_hypotheses SET counter_hypothesis_id=:surv WHERE counter_hypothesis_id=:dup"), {"surv": survivor["id"], "dup": dup["id"]})
                # Remove any self-links created by reparenting, then duplicate links.
                bind.execute(sa.text("DELETE FROM counter_hypotheses WHERE primary_hypothesis_id=counter_hypothesis_id"))

            if "decision_recommendations" in insp.get_table_names():
                dc = {c["name"] for c in sa.inspect(bind).get_columns("decision_recommendations")}
                if "grounded_hypothesis_id" in dc:
                    bind.execute(sa.text("UPDATE decision_recommendations SET grounded_hypothesis_id=:surv WHERE grounded_hypothesis_id=:dup"), {"surv": survivor["id"], "dup": dup["id"]})

            if "investigation_graph_edges" in insp.get_table_names():
                gc = {c["name"] for c in sa.inspect(bind).get_columns("investigation_graph_edges")}
                if "target_node_id" in gc:
                    bind.execute(sa.text("UPDATE investigation_graph_edges SET target_node_id=:surv WHERE target_node_type='HYPOTHESIS' AND target_node_id=:dup"), {"surv": survivor["id"], "dup": dup["id"]})
                if "source_node_id" in gc:
                    bind.execute(sa.text("UPDATE investigation_graph_edges SET source_node_id=:surv WHERE source_node_type='HYPOTHESIS' AND source_node_id=:dup"), {"surv": survivor["id"], "dup": dup["id"]})

            bind.execute(sa.text("DELETE FROM hypotheses WHERE id=:id"), {"id": dup["id"]})

    # SQLite requires batch ALTER for adding a unique constraint; use Alembic's
    # batch mode for both SQLite and PostgreSQL.
    op.create_index("ix_hypotheses_canonical_identity", "hypotheses", ["canonical_identity"], unique=False)
    with op.batch_alter_table("hypotheses", recreate="always") as batch_op:
        batch_op.alter_column("canonical_identity", existing_type=sa.String(length=64), nullable=False)
        batch_op.create_unique_constraint("uq_hypotheses_investigation_identity", ["investigation_id", "canonical_identity"])


def downgrade() -> None:
    with op.batch_alter_table("hypotheses", recreate="always") as batch_op:
        batch_op.drop_constraint("uq_hypotheses_investigation_identity", type_="unique")
    op.drop_index("ix_hypotheses_canonical_identity", table_name="hypotheses")
    op.drop_column("hypotheses", "canonical_identity")
