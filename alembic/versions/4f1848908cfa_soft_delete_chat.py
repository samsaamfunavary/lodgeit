"""soft delete chat

Revision ID: 4f1848908cfa
Revises: 05df84d18899
Create Date: 2025-09-29 11:32:06.702344

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4f1848908cfa'
down_revision: Union[str, Sequence[str], None] = '05df84d18899'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
