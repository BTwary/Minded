"""Add persistent multi-question analysis sessions."""
from __future__ import annotations
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "ab42d0e9c771"
down_revision: Union[str, None] = "bd7a3f2e91c4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "analysis_sessions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("project_id", sa.String(length=36), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("user_id", sa.String(length=36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="OPEN"),
        sa.Column("dataset_ids_json", sa.JSON(), nullable=False),
        sa.Column("scope_filter", sa.Text(), nullable=True),
        sa.Column("question_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("accepted_investigation_ids_json", sa.JSON(), nullable=False),
        sa.Column("report_revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("closed_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_analysis_sessions_project_id", "analysis_sessions", ["project_id"])
    op.create_index("ix_analysis_sessions_user_id", "analysis_sessions", ["user_id"])
    op.create_index("ix_analysis_sessions_status", "analysis_sessions", ["status"])
    with op.batch_alter_table("investigations") as b:
        b.add_column(sa.Column("analysis_session_id", sa.String(length=36), nullable=True))
        b.create_foreign_key("fk_investigations_analysis_session_id", "analysis_sessions", ["analysis_session_id"], ["id"])
        b.create_index("ix_investigations_analysis_session_id", ["analysis_session_id"])


def downgrade() -> None:
    with op.batch_alter_table("investigations") as b:
        b.drop_index("ix_investigations_analysis_session_id")
        b.drop_constraint("fk_investigations_analysis_session_id", type_="foreignkey")
        b.drop_column("analysis_session_id")
    op.drop_index("ix_analysis_sessions_status", table_name="analysis_sessions")
    op.drop_index("ix_analysis_sessions_user_id", table_name="analysis_sessions")
    op.drop_index("ix_analysis_sessions_project_id", table_name="analysis_sessions")
    op.drop_table("analysis_sessions")
