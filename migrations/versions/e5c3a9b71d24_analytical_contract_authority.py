"""v20-C4.2.3 analytical contract authority: final contract + deterministic identities

Adds the durable FINAL analytical contract to investigation_contracts, splits the
compiler proposal (proposed_method_json) from the selected method, and adds
deterministic analytical-identity / role columns to hypotheses, experiments and
evidence.  Existing rows keep NULL identities (they predate finalization); the
data migration below moves the legacy compiler proposal out of selected_method_json
so it can no longer masquerade as the selected method.
"""
from __future__ import annotations
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "e5c3a9b71d24"
down_revision: Union[str, None] = "c7f5a1d2e8b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("investigation_contracts") as b:
        b.add_column(sa.Column("proposed_method_json", sa.JSON(), nullable=True))
        b.add_column(sa.Column("final_contract_json", sa.JSON(), nullable=True))
        b.add_column(sa.Column("analytical_identity", sa.String(length=64), nullable=True))
        b.add_column(sa.Column("finalized_at", sa.DateTime(), nullable=True))
        b.create_index("ix_investigation_contracts_analytical_identity", ["analytical_identity"])
    # Legacy rows: selected_method_json held the compiler proposal.  Preserve it
    # as the proposal and clear the "selected" slot (no final contract exists).
    op.execute("UPDATE investigation_contracts SET proposed_method_json = selected_method_json, "
               "selected_method_json = NULL WHERE final_contract_json IS NULL")

    with op.batch_alter_table("hypotheses") as b:
        b.add_column(sa.Column("analytical_identity", sa.String(length=64), nullable=True))
        b.create_index("ix_hypotheses_analytical_identity", ["analytical_identity"])

    with op.batch_alter_table("experiments") as b:
        b.add_column(sa.Column("experiment_role", sa.String(length=20), nullable=True))
        b.add_column(sa.Column("analytical_identity", sa.String(length=64), nullable=True))
        b.add_column(sa.Column("hypothesis_canonical_identity", sa.String(length=64), nullable=True))
        b.add_column(sa.Column("contract_id", sa.String(length=36), nullable=True))
        b.create_foreign_key("fk_experiments_contract_id", "investigation_contracts", ["contract_id"], ["id"])
        b.create_index("ix_experiments_experiment_role", ["experiment_role"])
        b.create_index("ix_experiments_analytical_identity", ["analytical_identity"])
        b.create_index("ix_experiments_hypothesis_canonical_identity", ["hypothesis_canonical_identity"])
        b.create_index("ix_experiments_contract_id", ["contract_id"])

    with op.batch_alter_table("evidence") as b:
        b.add_column(sa.Column("analytical_identity", sa.String(length=64), nullable=True))
        b.create_index("ix_evidence_analytical_identity", ["analytical_identity"])


def downgrade() -> None:
    with op.batch_alter_table("evidence") as b:
        b.drop_index("ix_evidence_analytical_identity")
        b.drop_column("analytical_identity")
    with op.batch_alter_table("experiments") as b:
        for ix in ("ix_experiments_contract_id", "ix_experiments_hypothesis_canonical_identity",
                   "ix_experiments_analytical_identity", "ix_experiments_experiment_role"):
            b.drop_index(ix)
        b.drop_constraint("fk_experiments_contract_id", type_="foreignkey")
        for c in ("contract_id", "hypothesis_canonical_identity", "analytical_identity", "experiment_role"):
            b.drop_column(c)
    with op.batch_alter_table("hypotheses") as b:
        b.drop_index("ix_hypotheses_analytical_identity")
        b.drop_column("analytical_identity")
    # Restore the legacy meaning of selected_method_json before dropping the split.
    op.execute("UPDATE investigation_contracts SET selected_method_json = proposed_method_json "
               "WHERE final_contract_json IS NULL AND selected_method_json IS NULL")
    with op.batch_alter_table("investigation_contracts") as b:
        b.drop_index("ix_investigation_contracts_analytical_identity")
        for c in ("finalized_at", "analytical_identity", "final_contract_json", "proposed_method_json"):
            b.drop_column(c)
