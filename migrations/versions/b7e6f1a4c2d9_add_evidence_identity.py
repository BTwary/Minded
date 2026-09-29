"""add canonical evidence identity"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "b7e6f1a4c2d9"
down_revision: Union[str, None] = "4f2e9d1a7c11"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("evidence", sa.Column("evidence_identity_hash", sa.String(length=64), nullable=True))
    op.create_index(
        "ix_evidence_evidence_identity_hash",
        "evidence",
        ["evidence_identity_hash"],
        unique=False,
    )
    op.create_index(
        "uq_evidence_investigation_identity",
        "evidence",
        ["investigation_id", "evidence_identity_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_evidence_investigation_identity", table_name="evidence")
    op.drop_index("ix_evidence_evidence_identity_hash", table_name="evidence")
    op.drop_column("evidence", "evidence_identity_hash")
