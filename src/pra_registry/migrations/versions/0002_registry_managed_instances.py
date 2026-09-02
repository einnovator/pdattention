"""Add first-class managed runtime instances.

Revision ID: 0002_managed_instances
Revises: 0001_registry
Create Date: 2026-09-02
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0002_managed_instances"
down_revision: Union[str, None] = "0001_registry"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "registry_managed_instances",
        sa.Column("instance_id", sa.String(length=255), nullable=False),
        sa.Column("instance_type", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("environment", sa.String(length=128), nullable=False),
        sa.Column("region", sa.String(length=128), nullable=False),
        sa.Column("cluster", sa.String(length=255), nullable=False),
        sa.Column("namespace", sa.String(length=255), nullable=False),
        sa.Column("host", sa.String(length=255), nullable=False),
        sa.Column("management_url", sa.String(length=1024), nullable=False),
        sa.Column("inference_url", sa.String(length=1024), nullable=True),
        sa.Column("pra_version", sa.String(length=128), nullable=False),
        sa.Column("component_version", sa.String(length=128), nullable=False),
        sa.Column("engine_kind", sa.String(length=128), nullable=True),
        sa.Column("engine_version", sa.String(length=128), nullable=True),
        sa.Column("health", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.Float(), nullable=False),
        sa.Column("last_heartbeat", sa.DateTime(timezone=True), nullable=False),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("capabilities", sa.JSON(), nullable=False),
        sa.Column("models", sa.JSON(), nullable=False),
        sa.Column("runtime_summary", sa.JSON(), nullable=False),
        sa.Column("observability", sa.JSON(), nullable=False),
        sa.Column("desired_revision", sa.Integer(), nullable=True),
        sa.Column("observed_revision", sa.Integer(), nullable=False),
        sa.Column("in_sync", sa.Boolean(), nullable=False),
        sa.Column("drift_fields", sa.JSON(), nullable=False),
        sa.Column("labels", sa.JSON(), nullable=False),
        sa.Column("metadata_payload", sa.JSON(), nullable=False),
        sa.Column("registration_source", sa.String(length=32), nullable=False),
        sa.Column("credential_identity", sa.String(length=255), nullable=True),
        sa.Column("deregistered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("instance_id"),
    )
    for column in (
        "instance_type", "name", "environment", "region", "cluster", "namespace",
        "host", "engine_kind", "status", "last_heartbeat",
    ):
        op.create_index(
            op.f(f"ix_registry_managed_instances_{column}"),
            "registry_managed_instances", [column], unique=False,
        )


def downgrade() -> None:
    for column in reversed((
        "instance_type", "name", "environment", "region", "cluster", "namespace",
        "host", "engine_kind", "status", "last_heartbeat",
    )):
        op.drop_index(
            op.f(f"ix_registry_managed_instances_{column}"),
            table_name="registry_managed_instances",
        )
    op.drop_table("registry_managed_instances")
