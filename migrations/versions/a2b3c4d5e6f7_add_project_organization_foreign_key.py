"""Add foreign key constraint on projects.org_id to organizations.id.

Revision ID: a2b3c4d5e6f7
Revises: 9d7e5b3c1a2f
"""
from alembic import op
import sqlalchemy as sa

revision = "a2b3c4d5e6f7"
down_revision = "9d7e5b3c1a2f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("projects") as batch_op:
        batch_op.create_foreign_key(
            "fk_projects_org_id_organizations",
            "organizations",
            ["org_id"],
            ["id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("projects") as batch_op:
        batch_op.drop_constraint("fk_projects_org_id_organizations", type_="foreignkey")
