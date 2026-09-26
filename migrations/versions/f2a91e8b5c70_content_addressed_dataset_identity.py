"""add content-addressed dataset identity and lineage metadata"""
from __future__ import annotations
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "f2a91e8b5c70"
down_revision: Union[str, None] = "d2a7c4e9f105"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("dataset_versions", sa.Column("content_hash", sa.String(length=64), nullable=True))
    op.create_index("ix_dataset_versions_content_hash", "dataset_versions", ["content_hash"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_dataset_versions_content_hash", table_name="dataset_versions")
    op.drop_column("dataset_versions", "content_hash")
