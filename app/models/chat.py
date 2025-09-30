from sqlalchemy import (
    Boolean,
    Column,
    Integer,
    String,
    DateTime,
    Text,
    ForeignKey,
    Float,
    func,
    Index,
    text
)
from sqlalchemy.orm import relationship
from app.core.database import Base


# --- Chat Table ---
class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    is_deleted = Column(Boolean, nullable=False, server_default=text('false'), index=True)

    messages = relationship("ChatMessage", back_populates="chat", cascade="all, delete-orphan")
    user = relationship("User")

# --- ChatMessage Table ---
class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id"), nullable=False, index=True)
    role = Column(String(50), nullable=False, index=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    chat = relationship("Chat", back_populates="messages")
    processed_query = Column(Text, nullable=True)

    sources = relationship("MessageSource", back_populates="chat_message", cascade="all, delete-orphan")

# --- MessageSource Table (The "RAG Memory") ---
class MessageSource(Base):
    __tablename__ = "message_sources"
    id = Column(Integer, primary_key=True)
    chat_message_id = Column(Integer, ForeignKey("chat_messages.id"), nullable=False, index=True)
    source_title = Column(String(512), nullable=True)
    
    # The column can still store up to 2048 characters.
    # We remove index=True from here and define it in __table_args__ below.
    source_url = Column(String(2048), nullable=True)
    
    source_hierarchy = Column(String(1024), nullable=True)
    retrieval_score = Column(Float, nullable=True)
    chat_message = relationship("ChatMessage", back_populates="sources")

    # This is the standard way to define complex or dialect-specific indexes.
    # It creates an index on the 'source_url' column, but only on the first 255 characters.
    __table_args__ = (
        Index('ix_message_sources_source_url', 'source_url', mysql_length=255),
    )

