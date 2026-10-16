"""Persist alert evaluation provenance, lifecycle, and versioned profile evidence.

Revision ID: bd7a3f2e91c4
Revises: a2b3c4d5e6f7
"""
from alembic import op
import sqlalchemy as sa

revision = "bd7a3f2e91c4"
down_revision = "a2b3c4d5e6f7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("dataset_versions") as batch_op:
        batch_op.add_column(sa.Column("profile_json", sa.JSON(), nullable=True))

    with op.batch_alter_table("alert_rules") as batch_op:
        batch_op.add_column(sa.Column("last_evaluation_status", sa.String(length=50), nullable=True))
        batch_op.add_column(sa.Column("last_evaluation_trigger", sa.String(length=50), nullable=True))

    with op.batch_alter_table("alert_events") as batch_op:
        batch_op.add_column(sa.Column("event_identity", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("dataset_id", sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column("dataset_version", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("dataset_version_id", sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column("dataset_fingerprint", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("metric_name", sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column("baseline_period", sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column("comparison_period", sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column("lifecycle_status", sa.String(length=50), nullable=True))
        batch_op.add_column(sa.Column("acknowledged_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("snoozed_until", sa.DateTime(), nullable=True))
        batch_op.create_index("ix_alert_events_dataset_id", ["dataset_id"], unique=False)
        batch_op.create_index("ix_alert_events_event_identity", ["event_identity"], unique=True)


def downgrade() -> None:
    with op.batch_alter_table("alert_events") as batch_op:
        batch_op.drop_index("ix_alert_events_event_identity")
        batch_op.drop_index("ix_alert_events_dataset_id")
        batch_op.drop_column("snoozed_until")
        batch_op.drop_column("acknowledged_at")
        batch_op.drop_column("lifecycle_status")
        batch_op.drop_column("comparison_period")
        batch_op.drop_column("baseline_period")
        batch_op.drop_column("metric_name")
        batch_op.drop_column("dataset_fingerprint")
        batch_op.drop_column("dataset_version_id")
        batch_op.drop_column("dataset_version")
        batch_op.drop_column("dataset_id")
        batch_op.drop_column("event_identity")

    with op.batch_alter_table("alert_rules") as batch_op:
        batch_op.drop_column("last_evaluation_trigger")
        batch_op.drop_column("last_evaluation_status")

    with op.batch_alter_table("dataset_versions") as batch_op:
        batch_op.drop_column("profile_json")
