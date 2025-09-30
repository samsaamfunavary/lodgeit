"""soft delete chatcontent

Revision ID: 4b0ac1ed0797
Revises: 4f1848908cfa
Create Date: 2025-09-29 11:33:57.320526

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4b0ac1ed0797'
down_revision: Union[str, Sequence[str], None] = '4f1848908cfa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
