"""sync experiment scope authority and dataset scoping columns

Revision ID: 6f8c4a1cb67a
Revises: 8c1d2e4f7a90
Create Date: 2026-09-09 10:25:24.963847

Hand-adjusted after autogenerate (see AAOS_FORENSIC_AUDIT_2026-09-09.md):
`alembic check` found these columns/constraint already declared on the ORM
models (apps.api.src.models.entities) but never migrated, so every
"harden the sampling/scope authority" code path that writes
analysis_row_count/full_scope/sampling_policy/sampling_reason/
sampling_method on Experiment, or requested_dataset_ids_json on
Investigation, fails at INSERT time against a properly-migrated (non-stale)
database. Two changes vs the raw autogenerate output:

1. Wrapped in `batch_alter_table` -- SQLite can't ALTER a table in place for
   constraint changes, and naive `op.add_column`/`op.drop_index` outside
   batch mode either fails outright or silently no-ops depending on driver
   version. Batch mode rebuilds the table safely on SQLite and is a no-op
   wrapper on Postgres.
2. Added `server_default` to the three NOT NULL columns being added to an
   already-populated `experiments` table (analysis_row_count=0,
   full_scope=true, sampling_policy='NONE') -- without a server-side
   default, adding a NOT NULL column to a table with existing rows fails
   immediately on any backend. The Python-side ORM default was already
   defined on the model, but that only applies to rows inserted through
   SQLAlchemy after this migration runs, not to the ALTER TABLE itself.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6f8c4a1cb67a'
down_revision: Union[str, None] = '8c1d2e4f7a90'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('evidence', schema=None) as batch_op:
        batch_op.drop_index('uq_evidence_investigation_identity')
        batch_op.create_unique_constraint(
            'uq_evidence_investigation_identity',
            ['investigation_id', 'evidence_identity_hash'],
        )

    with op.batch_alter_table('experiments', schema=None) as batch_op:
        batch_op.add_column(sa.Column(
            'analysis_row_count', sa.Integer(), nullable=False, server_default='0',
        ))
        batch_op.add_column(sa.Column(
            'full_scope', sa.Boolean(), nullable=False, server_default=sa.true(),
        ))
        batch_op.add_column(sa.Column(
            'sampling_policy', sa.String(length=50), nullable=False, server_default='NONE',
        ))
        batch_op.add_column(sa.Column('sampling_reason', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('sampling_method', sa.String(length=100), nullable=True))

    with op.batch_alter_table('investigations', schema=None) as batch_op:
        batch_op.add_column(sa.Column('requested_dataset_ids_json', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('investigations', schema=None) as batch_op:
        batch_op.drop_column('requested_dataset_ids_json')

    with op.batch_alter_table('experiments', schema=None) as batch_op:
        batch_op.drop_column('sampling_method')
        batch_op.drop_column('sampling_reason')
        batch_op.drop_column('sampling_policy')
        batch_op.drop_column('full_scope')
        batch_op.drop_column('analysis_row_count')

    with op.batch_alter_table('evidence', schema=None) as batch_op:
        batch_op.drop_constraint('uq_evidence_investigation_identity', type_='unique')
        batch_op.create_index(
            'uq_evidence_investigation_identity',
            ['investigation_id', 'evidence_identity_hash'],
            unique=True,
        )
