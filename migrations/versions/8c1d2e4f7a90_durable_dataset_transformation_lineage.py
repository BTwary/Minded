"""add durable dataset transformation lineage"""
from __future__ import annotations
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "8c1d2e4f7a90"
down_revision: Union[str, None] = "f2a91e8b5c70"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dataset_transformations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("dataset_id", sa.String(length=36), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("input_version_id", sa.String(length=36), sa.ForeignKey("dataset_versions.id"), nullable=True),
        sa.Column("output_version_id", sa.String(length=36), sa.ForeignKey("dataset_versions.id"), nullable=False),
        sa.Column("operation", sa.String(length=100), nullable=False),
        sa.Column("input_content_hash", sa.String(length=64), nullable=True),
        sa.Column("output_content_hash", sa.String(length=64), nullable=True),
        sa.Column("parameters_json", sa.JSON(), nullable=True),
        sa.Column("affected_rows", sa.Integer(), nullable=True),
        sa.Column("affected_columns_json", sa.JSON(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("source_record_json", sa.JSON(), nullable=True),
        sa.Column("record_hash", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("input_version_id", "output_version_id", "record_hash", name="uq_dataset_transformation_record"),
    )
    for name, cols in [
        ("ix_dataset_transformations_dataset_id", ["dataset_id"]),
        ("ix_dataset_transformations_input_version_id", ["input_version_id"]),
        ("ix_dataset_transformations_output_version_id", ["output_version_id"]),
        ("ix_dataset_transformations_input_content_hash", ["input_content_hash"]),
        ("ix_dataset_transformations_output_content_hash", ["output_content_hash"]),
        ("ix_dataset_transformations_record_hash", ["record_hash"]),
    ]:
        op.create_index(name, "dataset_transformations", cols, unique=False)


def downgrade() -> None:
    for name in [
        "ix_dataset_transformations_record_hash",
        "ix_dataset_transformations_output_content_hash",
        "ix_dataset_transformations_input_content_hash",
        "ix_dataset_transformations_output_version_id",
        "ix_dataset_transformations_input_version_id",
        "ix_dataset_transformations_dataset_id",
    ]:
        op.drop_index(name, table_name="dataset_transformations")
    op.drop_table("dataset_transformations")
