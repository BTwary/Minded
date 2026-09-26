"""add persisted deterministic Claim Gate decisions"""
from __future__ import annotations
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "c7f5a1d2e8b4"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "claim_gate_decisions",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("investigation_id", sa.String(length=64), sa.ForeignKey("investigations.id"), nullable=False),
        sa.Column("question_id", sa.String(length=64), nullable=True),
        sa.Column("requested_claim", sa.String(length=50), nullable=False),
        sa.Column("requested_level", sa.Integer(), nullable=False),
        sa.Column("evidence_level", sa.Integer(), nullable=False),
        sa.Column("max_supported_level", sa.Integer(), nullable=False),
        sa.Column("outcome", sa.String(length=30), nullable=False),
        sa.Column("design_status", sa.String(length=30), nullable=False),
        sa.Column("identification_strategy", sa.String(length=255), nullable=True),
        sa.Column("assumptions_json", sa.JSON(), nullable=True),
        sa.Column("blocking_conditions_json", sa.JSON(), nullable=True),
        sa.Column("missing_evidence_json", sa.JSON(), nullable=True),
        sa.Column("recovery_actions_json", sa.JSON(), nullable=True),
        sa.Column("allowed_claim", sa.Text(), nullable=False),
        sa.Column("blocked_claim", sa.Text(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("computed_evidence_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_claim_gate_decisions_investigation_id", "claim_gate_decisions", ["investigation_id"])
    op.create_index("ix_claim_gate_decisions_question_id", "claim_gate_decisions", ["question_id"])
    op.create_index("ix_claim_gate_decisions_outcome", "claim_gate_decisions", ["outcome"])
    op.create_index("ix_claim_gate_decisions_created_at", "claim_gate_decisions", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_claim_gate_decisions_created_at", table_name="claim_gate_decisions")
    op.drop_index("ix_claim_gate_decisions_outcome", table_name="claim_gate_decisions")
    op.drop_index("ix_claim_gate_decisions_question_id", table_name="claim_gate_decisions")
    op.drop_index("ix_claim_gate_decisions_investigation_id", table_name="claim_gate_decisions")
    op.drop_table("claim_gate_decisions")
