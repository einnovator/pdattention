"""Add explicit staged evidence conditions to qualifications.

Revision ID: 0005_evidence_condition
Revises: 0004_router_control_state
Create Date: 2026-09-03
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0005_evidence_condition"
down_revision: Union[str, None] = "0004_router_control_state"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "registry_qualifications",
        sa.Column(
            "condition",
            sa.String(64),
            nullable=False,
            server_default="AMBIGUOUS_LEGACY_CONDITION",
        ),
    )
    op.create_index(
        "ix_registry_qualifications_condition",
        "registry_qualifications",
        ["condition"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_registry_qualifications_condition",
        table_name="registry_qualifications",
    )
    op.drop_column("registry_qualifications", "condition")
