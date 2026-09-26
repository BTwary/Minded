"""Scope/contract hardening for compound investigations.

Adds the parent/sub-investigation metadata needed to execute genuinely compound
analytical questions as independent child investigations while keeping the user
facing investigation as the orchestration record. Also adds the semantic binding
JSON column that already exists in the ORM but was missing from the migration chain.
"""
from __future__ import annotations
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "f4b7c2d9e1a0"
down_revision: Union[str, None] = "e5c3a9b71d24"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("investigation_contracts") as b:
        b.add_column(sa.Column("semantic_bindings_json", sa.JSON(), nullable=True))

    with op.batch_alter_table("investigations") as b:
        b.add_column(sa.Column("parent_investigation_id", sa.String(length=64), nullable=True))
        b.add_column(sa.Column("is_subinvestigation", sa.Boolean(), nullable=False, server_default=sa.false()))
        b.create_foreign_key(
            "fk_investigations_parent_investigation",
            "investigations",
            ["parent_investigation_id"],
            ["id"],
        )
        b.create_index("ix_investigations_parent_investigation_id", ["parent_investigation_id"])
        b.create_index("ix_investigations_is_subinvestigation", ["is_subinvestigation"])


def downgrade() -> None:
    with op.batch_alter_table("investigations") as b:
        b.drop_index("ix_investigations_is_subinvestigation")
        b.drop_index("ix_investigations_parent_investigation_id")
        b.drop_constraint("fk_investigations_parent_investigation", type_="foreignkey")
        b.drop_column("is_subinvestigation")
        b.drop_column("parent_investigation_id")
    with op.batch_alter_table("investigation_contracts") as b:
        b.drop_column("semantic_bindings_json")
