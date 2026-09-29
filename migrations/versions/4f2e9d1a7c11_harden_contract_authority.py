"""harden contract authority and durable analytical phase state"""
from alembic import op
import sqlalchemy as sa

revision = "4f2e9d1a7c11"
down_revision = "3e8b6e5b4d1a"
branch_labels = None
depends_on = None

def upgrade():
    op.add_column("investigations", sa.Column("active_contract_id", sa.String(length=36), nullable=True))
    op.add_column("investigations", sa.Column("contract_revision", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("investigations", sa.Column("current_phase", sa.String(length=64), nullable=True))
    # batch recreation is required for SQLite; direct CREATE FOREIGN KEY is not supported.
    with op.batch_alter_table("investigations", recreate="always") as batch_op:
        batch_op.create_foreign_key(
            "fk_investigations_active_contract_id",
            "investigation_contracts",
            ["active_contract_id"],
            ["id"],
        )
    op.create_index("ix_investigations_active_contract_id", "investigations", ["active_contract_id"])
    op.add_column("investigation_contracts", sa.Column("current_phase", sa.String(length=64), nullable=True))
    op.add_column("investigation_contracts", sa.Column("execution_state_json", sa.JSON(), nullable=True))
    op.add_column("investigation_contracts", sa.Column("last_replan_reason", sa.Text(), nullable=True))
    op.add_column("investigation_contracts", sa.Column("last_replanned_at", sa.DateTime(), nullable=True))
    op.add_column("investigation_contracts", sa.Column("superseded_at", sa.DateTime(), nullable=True))
    # DEFECT-009: decision recommendation persistence is part of the canonical
    # investigation graph and must exist before Fix-3 can safely reparent it.
    op.create_table(
        "decision_recommendations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("investigation_id", sa.String(length=64), nullable=False),
        sa.Column("recommendation_id", sa.String(length=64), nullable=False),
        sa.Column("action_title", sa.Text(), nullable=False),
        sa.Column("action_description", sa.Text(), nullable=False),
        sa.Column("grounded_hypothesis_id", sa.String(length=128), nullable=False),
        sa.Column("target_metric", sa.String(length=255), nullable=False),
        sa.Column("expected_gain_metric", sa.Float(), nullable=True),
        sa.Column("downside_risk_metric", sa.Float(), nullable=True),
        sa.Column("probability_of_success", sa.Float(), nullable=True),
        sa.Column("net_expected_utility", sa.Float(), nullable=True),
        sa.Column("utility_function_description", sa.Text(), nullable=True),
        sa.Column("policy_compliance_passed", sa.Boolean(), nullable=True),
        sa.Column("required_preconditions_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["investigation_id"], ["investigations.id"]),
    )
    op.create_index("ix_decision_recommendations_investigation_id", "decision_recommendations", ["investigation_id"])

def downgrade():
    op.drop_index("ix_decision_recommendations_investigation_id", table_name="decision_recommendations")
    op.drop_table("decision_recommendations")
    op.drop_column("investigation_contracts", "superseded_at")
    op.drop_column("investigation_contracts", "last_replanned_at")
    op.drop_column("investigation_contracts", "last_replan_reason")
    op.drop_column("investigation_contracts", "execution_state_json")
    op.drop_column("investigation_contracts", "current_phase")
    op.drop_column("investigations", "current_phase")
    op.drop_column("investigations", "contract_revision")
    op.drop_index("ix_investigations_active_contract_id", table_name="investigations")
    with op.batch_alter_table("investigations", recreate="always") as batch_op:
        batch_op.drop_constraint("fk_investigations_active_contract_id", type_="foreignkey")
        batch_op.drop_column("active_contract_id")
