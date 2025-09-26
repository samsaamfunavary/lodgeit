from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import List, Optional

# --- Chat Schemas ---
class ChatRequest(BaseModel):
    chat_id: int # This is now a required field
    message: str
    chat_id: Optional[int] = None
    hierarchy_filters: Optional[List[str]] = []
    index_name: Optional[str] = None
    limit: int = 3
    stream: bool = True


class NewChatRequest(BaseModel):
    initial_message: str

class NewChatResponse(BaseModel):
    chat_id: int
    title: str

class ChatSessionInfo(BaseModel):
    """Minimal info for listing chat sessions in the sidebar."""
    id: int
    title: str
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MessageSourceResponse(BaseModel):
    """Represents a single RAG source document."""
    source_title: Optional[str] = None
    source_url: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)

class ChatMessageResponse(BaseModel):
    """Represents a single message in the chat history, including its sources."""
    id: int
    role: str
    content: str
    created_at: datetime
    sources: List[MessageSourceResponse] = []

    
    model_config = ConfigDict(from_attributes=True)
