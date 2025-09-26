from sqlalchemy.orm import Session, joinedload
from fastapi import HTTPException, status
from typing import List, Dict, Optional

# Import your database models
from app.models.user import User
from app.models.chat import Chat, ChatMessage, MessageSource

# --- DATABASE LOGIC HELPER FUNCTIONS ---

def create_chat_session(db: Session, user: User) -> Chat:
    """Creates a new chat session in the database with a default title."""
    new_chat = Chat(user_id=user.id, title="New Chat")
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return new_chat

def get_chat_session_for_user(db: Session, chat_id: int, user: User) -> Chat:
    """Retrieves an existing chat session, ensuring the user owns it."""
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        # --- THIS IS THE FIX ---
        # Changed status.HTTP_4_NOT_FOUND to status.HTTP_404_NOT_FOUND
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Chat session not found or access denied."
        )
    return chat

def get_chat_history_for_session(db: Session, chat_id: int) -> List[Dict[str, str]]:
    """Retrieves the full message history for a given chat session."""
    history = (
        db.query(ChatMessage)
        .filter(ChatMessage.chat_id == chat_id)
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
    return [{"role": msg.role, "content": msg.content} for msg in history]


def add_message_to_db(db: Session, chat_id: int, role: str, content: str) -> ChatMessage:
    """Adds a new message to a chat session."""
    new_message = ChatMessage(chat_id=chat_id, role=role, content=content)
    db.add(new_message)
    db.commit()
    db.refresh(new_message)
    return new_message

def add_sources_to_message_in_db(db: Session, message_id: int, sources: List[dict]):
    """Adds RAG source documents to an assistant's message."""
    if not sources:
        return
    for source_data in sources:
        new_source = MessageSource(
            chat_message_id=message_id,
            source_title=source_data.get("title"),
            source_url=source_data.get("url"),
            source_hierarchy=source_data.get("hierarchy"),
            retrieval_score=source_data.get("score")
        )
        db.add(new_source)
    db.commit()


def get_chat_sessions_for_user(db: Session, user_id: int) -> List[Chat]:
    """Retrieves all chat sessions for a specific user, most recent first."""
    return db.query(Chat).filter(Chat.user_id == user_id).order_by(Chat.updated_at.desc()).all()

def get_messages_for_chat_session(db: Session, chat_id: int, user_id: int) -> List[ChatMessage]:
    """
    Retrieves all messages for a given chat session if the user owns it.
    Uses joinedload to efficiently fetch sources for assistant messages.
    """
    # First, verify the user owns the chat they are trying to load
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user_id).first()
    if not chat:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found or access denied.")
    
    # If ownership is verified, fetch the messages with their sources
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.chat_id == chat_id)
        .options(joinedload(ChatMessage.sources)) # Eagerly load sources to prevent extra queries
        .order_by(ChatMessage.created_at.asc())
        .all()
    )
