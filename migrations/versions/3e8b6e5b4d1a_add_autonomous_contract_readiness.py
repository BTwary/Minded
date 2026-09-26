"""add durable autonomous analysis contract and data readiness"""
from alembic import op
import sqlalchemy as sa

revision = "3e8b6e5b4d1a"
down_revision = "9c1c76a818a4"
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "investigation_contracts",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("investigation_id", sa.String(length=64), sa.ForeignKey("investigations.id"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("parent_contract_id", sa.String(length=36), sa.ForeignKey("investigation_contracts.id"), nullable=True),
        sa.Column("status", sa.String(length=40), nullable=False, server_default="ACTIVE"),
        sa.Column("original_question", sa.Text(), nullable=False),
        sa.Column("normalized_question", sa.Text()),
        sa.Column("problem_class", sa.String(length=50), nullable=False),
        sa.Column("claim_type", sa.String(length=50), nullable=False),
        sa.Column("target_json", sa.JSON()), sa.Column("explanatory_variables_json", sa.JSON()),
        sa.Column("population_json", sa.JSON()), sa.Column("grain_json", sa.JSON()), sa.Column("scope_json", sa.JSON()),
        sa.Column("time_window_json", sa.JSON()), sa.Column("estimand_json", sa.JSON()), sa.Column("assumptions_json", sa.JSON()),
        sa.Column("data_requirements_json", sa.JSON()), sa.Column("candidate_methods_json", sa.JSON()),
        sa.Column("selected_method_json", sa.JSON()), sa.Column("evidence_requirements_json", sa.JSON()),
        sa.Column("stopping_criteria_json", sa.JSON()), sa.Column("ambiguity_state_json", sa.JSON()),
        sa.Column("semantic_interpretations_json", sa.JSON()), sa.Column("unresolved_questions_json", sa.JSON()),
        sa.Column("confidence", sa.Float()), sa.Column("compiled_at", sa.DateTime()),
    )
    op.create_index("ix_investigation_contracts_investigation_id", "investigation_contracts", ["investigation_id"])
    op.create_index("ix_investigation_contracts_parent_contract_id", "investigation_contracts", ["parent_contract_id"])
    op.create_table(
        "investigation_data_readiness",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("investigation_id", sa.String(length=64), sa.ForeignKey("investigations.id"), nullable=False),
        sa.Column("contract_id", sa.String(length=36), sa.ForeignKey("investigation_contracts.id"), nullable=True),
        sa.Column("dataset_name", sa.String(length=255), nullable=False), sa.Column("dataset_version_ids_json", sa.JSON()),
        sa.Column("source_fingerprints_json", sa.JSON()), sa.Column("row_count", sa.Integer(), server_default="0"),
        sa.Column("column_count", sa.Integer(), server_default="0"), sa.Column("overall_quality_score", sa.Float()),
        sa.Column("fitness_verdict", sa.String(length=30), nullable=False), sa.Column("can_proceed", sa.Boolean(), server_default=sa.false()),
        sa.Column("checks_json", sa.JSON()), sa.Column("critical_issues_json", sa.JSON()), sa.Column("warnings_json", sa.JSON()),
        sa.Column("recommendations_json", sa.JSON()), sa.Column("assessed_at", sa.DateTime()),
    )
    op.create_index("ix_investigation_data_readiness_investigation_id", "investigation_data_readiness", ["investigation_id"])
    op.create_index("ix_investigation_data_readiness_contract_id", "investigation_data_readiness", ["contract_id"])

def downgrade():
    op.drop_index("ix_investigation_data_readiness_contract_id", table_name="investigation_data_readiness")
    op.drop_index("ix_investigation_data_readiness_investigation_id", table_name="investigation_data_readiness")
    op.drop_table("investigation_data_readiness")
    op.drop_index("ix_investigation_contracts_parent_contract_id", table_name="investigation_contracts")
    op.drop_index("ix_investigation_contracts_investigation_id", table_name="investigation_contracts")
    op.drop_table("investigation_contracts")
