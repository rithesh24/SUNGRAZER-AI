"""add candidate review fields

Revision ID: 34c9687c7261
Revises: 0cdd0a392a0c
Create Date: 2026-09-22 19:04:53.662569

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '34c9687c7261'
down_revision: Union[str, Sequence[str], None] = '0cdd0a392a0c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


REVIEW_VERDICTS = ("approved", "rejected", "artifact", "known_object",
                   "uncertain")


def upgrade() -> None:
    """Add human-review columns to candidates (S7 review workflow)."""
    op.add_column("candidates", sa.Column("review", sa.String(20)))
    op.add_column("candidates", sa.Column("reviewer_notes", sa.Text()))
    op.add_column("candidates",
                  sa.Column("reviewed_at", sa.DateTime(timezone=True)))
    op.create_check_constraint(
        "ck_candidate_review", "candidates",
        f"review IS NULL OR review IN {REVIEW_VERDICTS}")


def downgrade() -> None:
    """Drop the review columns."""
    op.drop_constraint("ck_candidate_review", "candidates")
    op.drop_column("candidates", "reviewed_at")
    op.drop_column("candidates", "reviewer_notes")
    op.drop_column("candidates", "review")
