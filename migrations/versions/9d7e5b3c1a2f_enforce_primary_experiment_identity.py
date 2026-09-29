"""Enforce one PRIMARY experiment per analytical identity.

Revision ID: 9d7e5b3c1a2f
Revises: f4b7c2d9e1a0
"""
from alembic import op
import sqlalchemy as sa

revision = "9d7e5b3c1a2f"
down_revision = "f4b7c2d9e1a0"
branch_labels = None
depends_on = None

_INDEX = "uq_experiments_primary_identity"


def upgrade() -> None:
    bind = op.get_bind()
    duplicate_sql = sa.text(
        """
        SELECT investigation_id, analytical_identity, COUNT(*) AS n
        FROM experiments
        WHERE experiment_role = 'PRIMARY'
          AND analytical_identity IS NOT NULL
          AND TRIM(analytical_identity) <> ''
        GROUP BY investigation_id, analytical_identity
        HAVING COUNT(*) > 1
        """
    )
    duplicates = bind.execute(duplicate_sql).fetchall()
    if duplicates:
        raise RuntimeError(
            "Cannot create uq_experiments_primary_identity: existing duplicate PRIMARY "
            f"experiments detected for {[(r[0], r[1], r[2]) for r in duplicates]!r}. "
            "Resolve the duplicates explicitly before migrating."
        )

    op.create_index(
        _INDEX,
        "experiments",
        ["investigation_id", "analytical_identity"],
        unique=True,
        sqlite_where=sa.text("experiment_role = 'PRIMARY'"),
        postgresql_where=sa.text("experiment_role = 'PRIMARY'"),
    )


def downgrade() -> None:
    op.drop_index(_INDEX, table_name="experiments")
