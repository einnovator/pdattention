"""Add model-list desired state to deployments.

Revision ID: 0003_multi_model_desired
Revises: 0002_managed_instances
Create Date: 2026-09-02
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0003_multi_model_desired"
down_revision: Union[str, None] = "0002_managed_instances"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("registry_deployments") as batch:
        batch.add_column(sa.Column(
            "desired_models", sa.JSON(), nullable=False, server_default=sa.text("'[]'"),
        ))
        batch.add_column(sa.Column(
            "allow_extra_models", sa.Boolean(), nullable=False, server_default=sa.true(),
        ))


def downgrade() -> None:
    with op.batch_alter_table("registry_deployments") as batch:
        batch.drop_column("allow_extra_models")
        batch.drop_column("desired_models")
