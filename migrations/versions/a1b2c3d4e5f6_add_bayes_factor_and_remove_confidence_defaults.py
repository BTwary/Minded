"""Add explicit Bayes-factor storage and remove epistemic confidence defaults.

Revision ID: a1b2c3d4e5f6
Revises: 6f8c4a1cb67a
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "6f8c4a1cb67a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite requires batch mode for the NOT NULL -> nullable alteration.
    with op.batch_alter_table("belief_updates", schema=None) as batch_op:
        batch_op.alter_column("likelihood_p", existing_type=sa.Float(), nullable=True)
        batch_op.add_column(sa.Column("bayes_factor", sa.Float(), nullable=True))

    # AnalysisRun.confidence is already nullable in the canonical migration;
    # this batch operation deliberately only removes an ORM/server default if
    # a deployment has one. SQLite's table recreation is safe here.
    with op.batch_alter_table("analysis_runs", schema=None) as batch_op:
        batch_op.alter_column("confidence", existing_type=sa.String(length=50), nullable=True, server_default=None)


def downgrade() -> None:
    # A downgrade cannot safely reconstruct historical BF values into the
    # probability-valued legacy field. Refuse rather than fabricate a mapping.
    raise RuntimeError(
        "Downgrade of a1b2c3d4e5f6 is intentionally unsupported: bayes_factor "
        "cannot be losslessly converted into likelihood_p. Restore a database "
        "backup from before this migration if rollback is required."
    )
