"""Add transport-neutral router desired and observed state.

Revision ID: 0004_router_control_state
Revises: 0003_multi_model_desired
Create Date: 2026-09-03
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0004_router_control_state"
down_revision: Union[str, None] = "0003_multi_model_desired"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "registry_router_instances",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("kind", sa.String(64), nullable=False, index=True),
        sa.Column("version", sa.String(128)),
        sa.Column("management_url", sa.String(1024), nullable=False),
        sa.Column("inference_url", sa.String(1024)),
        sa.Column("credential_reference", sa.String(255)),
        sa.Column("region", sa.String(128), nullable=False, index=True),
        sa.Column("cluster", sa.String(255), nullable=False, index=True),
        sa.Column("health", sa.String(64), nullable=False, index=True),
        sa.Column("desired_revision", sa.Integer(), nullable=False),
        sa.Column("observed_revision", sa.Integer(), nullable=False),
        sa.Column("supported_features", sa.JSON(), nullable=False),
        sa.Column("labels", sa.JSON(), nullable=False),
        sa.Column("metadata_payload", sa.JSON(), nullable=False),
        sa.Column("last_sync", sa.DateTime(timezone=True)),
        sa.Column("last_error", sa.Text()),
        *_timestamps(),
    )
    op.create_table(
        "registry_routes",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("public_model", sa.String(512), nullable=False, index=True),
        sa.Column("route_kind", sa.String(32), nullable=False, index=True),
        sa.Column("policy_id", sa.String(255), nullable=False, index=True),
        sa.Column("pool_ids", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, index=True),
        sa.Column("fallback_pool_ids", sa.JSON(), nullable=False),
        sa.Column("tenant_ids", sa.JSON(), nullable=False),
        sa.Column("metadata_payload", sa.JSON(), nullable=False),
        sa.Column("desired_revision", sa.Integer(), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "registry_model_pools",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("model_id", sa.String(512), nullable=False, index=True),
        sa.Column("model_revision", sa.String(255)),
        sa.Column("selectors", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, index=True),
        sa.Column("metadata_payload", sa.JSON(), nullable=False),
        sa.Column("desired_revision", sa.Integer(), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "registry_backend_endpoints",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("pool_ids", sa.JSON(), nullable=False),
        sa.Column("engine_instance_id", sa.String(255), index=True),
        sa.Column("runtime_model_id", sa.String(255), nullable=False),
        sa.Column("inference_url", sa.String(1024), nullable=False),
        sa.Column("engine", sa.String(128), nullable=False, index=True),
        sa.Column("engine_version", sa.String(128)),
        sa.Column("model_id", sa.String(512), nullable=False, index=True),
        sa.Column("model_revision", sa.String(255)),
        sa.Column("model_fingerprint", sa.String(255)),
        sa.Column("bundle_id", sa.String(512), index=True),
        sa.Column("bundle_revision", sa.String(255)),
        sa.Column("profile", sa.String(128), index=True),
        sa.Column("modes", sa.JSON(), nullable=False),
        sa.Column("qualification_tier", sa.String(64), nullable=False, index=True),
        sa.Column("approval_state", sa.String(32), nullable=False, index=True),
        sa.Column("region", sa.String(128), nullable=False, index=True),
        sa.Column("cluster", sa.String(255), nullable=False, index=True),
        sa.Column("health", sa.String(64), nullable=False, index=True),
        sa.Column("maintenance", sa.Boolean(), nullable=False, index=True),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("cost", sa.Float()),
        sa.Column("labels", sa.JSON(), nullable=False),
        sa.Column("metadata_payload", sa.JSON(), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "registry_routing_policies",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("strategy", sa.String(64), nullable=False, index=True),
        sa.Column("constraints", sa.JSON(), nullable=False),
        sa.Column("preferences", sa.JSON(), nullable=False),
        sa.Column("fallback", sa.JSON(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, index=True),
        sa.Column("metadata_payload", sa.JSON(), nullable=False),
        sa.Column("desired_revision", sa.Integer(), nullable=False),
        *_timestamps(),
    )
    op.create_table(
        "registry_route_bindings",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("route_id", sa.String(255), nullable=False, index=True),
        sa.Column("router_id", sa.String(255), nullable=False, index=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, index=True),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("metadata_payload", sa.JSON(), nullable=False),
        sa.Column("desired_revision", sa.Integer(), nullable=False),
        *_timestamps(),
    )


def downgrade() -> None:
    for table in (
        "registry_route_bindings",
        "registry_routing_policies",
        "registry_backend_endpoints",
        "registry_model_pools",
        "registry_routes",
        "registry_router_instances",
    ):
        op.drop_table(table)
