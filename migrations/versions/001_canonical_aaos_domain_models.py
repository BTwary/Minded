"""Canonical AA-OS Semantic and Investigation Domain Models

Revision ID: 001_canonical_aaos
Revises: 
Create Date: 2026-08-22 14:25:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001_canonical_aaos'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Semantic Domain Tables
    op.create_table(
        'semantic_models',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('project_id', sa.String(length=36), sa.ForeignKey('projects.id'), nullable=False),
        sa.Column('version_number', sa.Integer(), nullable=False, default=1),
        sa.Column('status', sa.String(length=50), nullable=False, default='active'),
        sa.Column('summary_description', sa.Text(), nullable=True),
        sa.Column('table_grains_json', sa.JSON(), nullable=True),
        sa.Column('dataset_version_ids_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_semantic_models_project_id', 'semantic_models', ['project_id'])

    op.create_table(
        'semantic_entities',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('semantic_model_id', sa.String(length=36), sa.ForeignKey('semantic_models.id'), nullable=False),
        sa.Column('entity_name', sa.String(length=255), nullable=False),
        sa.Column('primary_key', sa.String(length=255), nullable=False),
        sa.Column('table_name', sa.String(length=255), nullable=False),
        sa.Column('natural_keys_json', sa.JSON(), nullable=True),
        sa.Column('attributes_json', sa.JSON(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_semantic_entities_semantic_model_id', 'semantic_entities', ['semantic_model_id'])

    op.create_table(
        'semantic_metrics',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('semantic_model_id', sa.String(length=36), sa.ForeignKey('semantic_models.id'), nullable=False),
        sa.Column('metric_name', sa.String(length=255), nullable=False),
        sa.Column('display_name', sa.String(length=255), nullable=False),
        sa.Column('column_name', sa.String(length=255), nullable=True),
        sa.Column('table_name', sa.String(length=255), nullable=False),
        sa.Column('additivity', sa.String(length=50), nullable=True),
        sa.Column('unit', sa.String(length=50), nullable=True),
        sa.Column('target_direction', sa.String(length=50), nullable=True),
        sa.Column('sql_formula', sa.Text(), nullable=True),
        sa.Column('derived_expression', sa.Text(), nullable=True),
        sa.Column('dependent_metrics_json', sa.JSON(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_semantic_metrics_semantic_model_id', 'semantic_metrics', ['semantic_model_id'])

    op.create_table(
        'semantic_dimensions',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('semantic_model_id', sa.String(length=36), sa.ForeignKey('semantic_models.id'), nullable=False),
        sa.Column('dimension_name', sa.String(length=255), nullable=False),
        sa.Column('column_name', sa.String(length=255), nullable=False),
        sa.Column('table_name', sa.String(length=255), nullable=False),
        sa.Column('cardinality', sa.Integer(), nullable=True),
        sa.Column('hierarchy_level', sa.Integer(), nullable=True),
        sa.Column('is_hierarchical', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_semantic_dimensions_semantic_model_id', 'semantic_dimensions', ['semantic_model_id'])

    op.create_table(
        'semantic_relationships',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('semantic_model_id', sa.String(length=36), sa.ForeignKey('semantic_models.id'), nullable=False),
        sa.Column('source_table', sa.String(length=255), nullable=False),
        sa.Column('source_column', sa.String(length=255), nullable=False),
        sa.Column('target_table', sa.String(length=255), nullable=False),
        sa.Column('target_column', sa.String(length=255), nullable=False),
        sa.Column('cardinality', sa.String(length=50), nullable=True),
        sa.Column('join_safety_score', sa.Float(), nullable=True),
        sa.Column('fanout_risk', sa.Boolean(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_semantic_relationships_semantic_model_id', 'semantic_relationships', ['semantic_model_id'])

    op.create_table(
        'semantic_time_definitions',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('semantic_model_id', sa.String(length=36), sa.ForeignKey('semantic_models.id'), nullable=False),
        sa.Column('table_name', sa.String(length=255), nullable=False),
        sa.Column('column_name', sa.String(length=255), nullable=False),
        sa.Column('grain', sa.String(length=50), nullable=True),
        sa.Column('min_date', sa.String(length=50), nullable=True),
        sa.Column('max_date', sa.String(length=50), nullable=True),
        sa.Column('is_continuous', sa.Boolean(), nullable=True),
        sa.Column('gap_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_semantic_time_definitions_semantic_model_id', 'semantic_time_definitions', ['semantic_model_id'])

    op.create_table(
        'semantic_business_rules',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('semantic_model_id', sa.String(length=36), sa.ForeignKey('semantic_models.id'), nullable=False),
        sa.Column('table_name', sa.String(length=255), nullable=False),
        sa.Column('column_name', sa.String(length=255), nullable=False),
        sa.Column('rule_type', sa.String(length=100), nullable=False),
        sa.Column('expression', sa.Text(), nullable=False),
        sa.Column('severity', sa.String(length=50), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_semantic_business_rules_semantic_model_id', 'semantic_business_rules', ['semantic_model_id'])

    # 2. Investigation & Epistemic Domain Tables
    op.create_table(
        'investigations',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('project_id', sa.String(length=36), sa.ForeignKey('projects.id'), nullable=False),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('semantic_model_id', sa.String(length=36), sa.ForeignKey('semantic_models.id'), nullable=True),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, default='PLANNED'),
        sa.Column('verdict_type', sa.String(length=50), nullable=False, default='INCONCLUSIVE'),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('direct_answer', sa.Text(), nullable=True),
        sa.Column('main_finding', sa.Text(), nullable=True),
        sa.Column('entropy_initial', sa.Float(), nullable=True),
        sa.Column('entropy_current', sa.Float(), nullable=True),
        sa.Column('stopping_criteria_met', sa.Boolean(), nullable=True),
        sa.Column('stopping_rationale', sa.Text(), nullable=True),
        sa.Column('reproducible_manifest_hash', sa.String(length=128), nullable=True),
        sa.Column('dataset_version_ids_json', sa.JSON(), nullable=True),
        sa.Column('execution_time_seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_investigations_project_id', 'investigations', ['project_id'])
    op.create_index('ix_investigations_user_id', 'investigations', ['user_id'])
    op.create_index('ix_investigations_semantic_model_id', 'investigations', ['semantic_model_id'])

    op.create_table(
        'investigation_objectives',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('investigation_id', sa.String(length=64), sa.ForeignKey('investigations.id'), nullable=False),
        sa.Column('statement', sa.Text(), nullable=False),
        sa.Column('target_variable', sa.String(length=255), nullable=True),
        sa.Column('priority_rank', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_investigation_objectives_investigation_id', 'investigation_objectives', ['investigation_id'])

    op.create_table(
        'investigation_unknowns',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('investigation_id', sa.String(length=64), sa.ForeignKey('investigations.id'), nullable=False),
        sa.Column('variable_name', sa.String(length=255), nullable=False),
        sa.Column('target_metric', sa.String(length=255), nullable=True),
        sa.Column('dimension_scope', sa.String(length=255), nullable=True),
        sa.Column('prior_estimate', sa.String(length=255), nullable=True),
        sa.Column('posterior_estimate', sa.String(length=255), nullable=True),
        sa.Column('uncertainty_range', sa.String(length=255), nullable=True),
        sa.Column('is_resolved', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_investigation_unknowns_investigation_id', 'investigation_unknowns', ['investigation_id'])

    op.create_table(
        'hypotheses',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('investigation_id', sa.String(length=64), sa.ForeignKey('investigations.id'), nullable=False),
        sa.Column('hypothesis_code', sa.String(length=50), nullable=True),
        sa.Column('statement', sa.Text(), nullable=False),
        sa.Column('rationale', sa.Text(), nullable=True),
        sa.Column('prior_probability', sa.Float(), nullable=True),
        sa.Column('posterior_probability', sa.Float(), nullable=True),
        sa.Column('belief_state', sa.String(length=50), nullable=True),
        sa.Column('is_counter_hypothesis', sa.Boolean(), nullable=True),
        sa.Column('last_evidence_likelihood', sa.Float(), nullable=True),
        sa.Column('reason_for_rejection', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_hypotheses_investigation_id', 'hypotheses', ['investigation_id'])

    op.create_table(
        'predictions',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('hypothesis_id', sa.String(length=64), sa.ForeignKey('hypotheses.id'), nullable=False),
        sa.Column('statement', sa.Text(), nullable=False),
        sa.Column('expected_direction', sa.String(length=50), nullable=True),
        sa.Column('expected_magnitude', sa.String(length=100), nullable=True),
        sa.Column('falsification_condition', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_predictions_hypothesis_id', 'predictions', ['hypothesis_id'])

    op.create_table(
        'experiments',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('investigation_id', sa.String(length=64), sa.ForeignKey('investigations.id'), nullable=False),
        sa.Column('hypothesis_id', sa.String(length=64), sa.ForeignKey('hypotheses.id'), nullable=True),
        sa.Column('test_code', sa.String(length=50), nullable=True),
        sa.Column('tool_name', sa.String(length=100), nullable=False),
        sa.Column('arguments_json', sa.JSON(), nullable=True),
        sa.Column('rationale', sa.Text(), nullable=True),
        sa.Column('expected_information_gain', sa.Float(), nullable=True),
        sa.Column('test_cost', sa.Float(), nullable=True),
        sa.Column('test_reliability', sa.Float(), nullable=True),
        sa.Column('utility_score', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('executed_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_experiments_investigation_id', 'experiments', ['investigation_id'])
    op.create_index('ix_experiments_hypothesis_id', 'experiments', ['hypothesis_id'])

    op.create_table(
        'observations',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('experiment_id', sa.String(length=64), sa.ForeignKey('experiments.id'), nullable=False),
        sa.Column('result_json', sa.JSON(), nullable=True),
        sa.Column('row_count_analyzed', sa.Integer(), nullable=True),
        sa.Column('execution_time_ms', sa.Float(), nullable=True),
        sa.Column('sql_executed', sa.Text(), nullable=True),
        sa.Column('raw_data_hash', sa.String(length=128), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_observations_experiment_id', 'observations', ['experiment_id'])

    op.create_table(
        'evidence',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('investigation_id', sa.String(length=64), sa.ForeignKey('investigations.id'), nullable=False),
        sa.Column('experiment_id', sa.String(length=64), sa.ForeignKey('experiments.id'), nullable=True),
        sa.Column('hypothesis_id', sa.String(length=64), sa.ForeignKey('hypotheses.id'), nullable=True),
        sa.Column('statement', sa.Text(), nullable=False),
        sa.Column('calculation_summary', sa.Text(), nullable=True),
        sa.Column('statistical_test_name', sa.String(length=100), nullable=True),
        sa.Column('p_value', sa.Float(), nullable=True),
        sa.Column('effect_size', sa.Float(), nullable=True),
        sa.Column('validation_status', sa.String(length=50), nullable=True),
        sa.Column('relative_tolerance_observed', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_evidence_investigation_id', 'evidence', ['investigation_id'])
    op.create_index('ix_evidence_experiment_id', 'evidence', ['experiment_id'])
    op.create_index('ix_evidence_hypothesis_id', 'evidence', ['hypothesis_id'])

    op.create_table(
        'evidence_verifications',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('evidence_id', sa.String(length=64), sa.ForeignKey('evidence.id'), nullable=False),
        sa.Column('primary_tool', sa.String(length=100), nullable=False),
        sa.Column('secondary_tool', sa.String(length=100), nullable=False),
        sa.Column('tolerance_threshold', sa.Float(), nullable=True),
        sa.Column('observed_delta_pct', sa.Float(), nullable=True),
        sa.Column('is_deterministic', sa.Boolean(), nullable=True),
        sa.Column('validator_fingerprint', sa.String(length=128), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_evidence_verifications_evidence_id', 'evidence_verifications', ['evidence_id'])

    op.create_table(
        'belief_updates',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('investigation_id', sa.String(length=64), sa.ForeignKey('investigations.id'), nullable=False),
        sa.Column('hypothesis_id', sa.String(length=64), sa.ForeignKey('hypotheses.id'), nullable=False),
        sa.Column('evidence_id', sa.String(length=64), sa.ForeignKey('evidence.id'), nullable=True),
        sa.Column('prior_probability', sa.Float(), nullable=False),
        sa.Column('likelihood_p', sa.Float(), nullable=False),
        sa.Column('posterior_probability', sa.Float(), nullable=False),
        sa.Column('entropy_delta', sa.Float(), nullable=True),
        sa.Column('update_step_index', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_belief_updates_investigation_id', 'belief_updates', ['investigation_id'])
    op.create_index('ix_belief_updates_hypothesis_id', 'belief_updates', ['hypothesis_id'])

    op.create_table(
        'counter_hypotheses',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('primary_hypothesis_id', sa.String(length=64), sa.ForeignKey('hypotheses.id'), nullable=False),
        sa.Column('counter_hypothesis_id', sa.String(length=64), sa.ForeignKey('hypotheses.id'), nullable=False),
        sa.Column('relation_type', sa.String(length=50), nullable=True),
        sa.Column('discrimination_test_id', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_counter_hypotheses_primary_hypothesis_id', 'counter_hypotheses', ['primary_hypothesis_id'])
    op.create_index('ix_counter_hypotheses_counter_hypothesis_id', 'counter_hypotheses', ['counter_hypothesis_id'])

    op.create_table(
        'assumptions',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('investigation_id', sa.String(length=64), sa.ForeignKey('investigations.id'), nullable=False),
        sa.Column('statement', sa.Text(), nullable=False),
        sa.Column('is_validated', sa.Boolean(), nullable=True),
        sa.Column('sensitivity_risk', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_assumptions_investigation_id', 'assumptions', ['investigation_id'])

    op.create_table(
        'investigation_verdicts',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('investigation_id', sa.String(length=64), sa.ForeignKey('investigations.id'), unique=True, nullable=False),
        sa.Column('verdict_type', sa.String(length=50), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('justification', sa.Text(), nullable=False),
        sa.Column('net_variance_explained_pct', sa.Float(), nullable=True),
        sa.Column('counter_hypothesis_refuted', sa.Boolean(), nullable=True),
        sa.Column('epistemic_grade', sa.String(length=10), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_investigation_verdicts_investigation_id', 'investigation_verdicts', ['investigation_id'])

    op.create_table(
        'investigation_graph_edges',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('investigation_id', sa.String(length=64), sa.ForeignKey('investigations.id'), nullable=False),
        sa.Column('source_node_type', sa.String(length=50), nullable=False),
        sa.Column('source_node_id', sa.String(length=64), nullable=False),
        sa.Column('target_node_type', sa.String(length=50), nullable=False),
        sa.Column('target_node_id', sa.String(length=64), nullable=False),
        sa.Column('relationship_type', sa.String(length=50), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_investigation_graph_edges_investigation_id', 'investigation_graph_edges', ['investigation_id'])
    op.create_index('ix_investigation_graph_edges_source_node_id', 'investigation_graph_edges', ['source_node_id'])
    op.create_index('ix_investigation_graph_edges_target_node_id', 'investigation_graph_edges', ['target_node_id'])


def downgrade() -> None:
    op.drop_table('investigation_graph_edges')
    op.drop_table('investigation_verdicts')
    op.drop_table('assumptions')
    op.drop_table('counter_hypotheses')
    op.drop_table('belief_updates')
    op.drop_table('evidence_verifications')
    op.drop_table('evidence')
    op.drop_table('observations')
    op.drop_table('experiments')
    op.drop_table('predictions')
    op.drop_table('hypotheses')
    op.drop_table('investigation_unknowns')
    op.drop_table('investigation_objectives')
    op.drop_table('investigations')
    op.drop_table('semantic_business_rules')
    op.drop_table('semantic_time_definitions')
    op.drop_table('semantic_relationships')
    op.drop_table('semantic_dimensions')
    op.drop_table('semantic_metrics')
    op.drop_table('semantic_entities')
    op.drop_table('semantic_models')
