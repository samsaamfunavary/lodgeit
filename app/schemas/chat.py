
from calendar import c
from datetime import datetime
from pydantic import BaseModel, ConfigDict
from typing import List, Optional , Dict

# --- Chat Schemas ---
class ChatRequest(BaseModel):
    chat_id: int # This is now a required field
    message: str
    hierarchy_filters: Optional[List[str]] = []
    index_name: Optional[str] = None
    limit: int = 4
    stream: bool = True


class ChatRequestWidget(BaseModel):
    message: str
    chat_history: List[Dict[str, str]]
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

class ChatSessionPage(BaseModel):
    """Represents a paginated list of chat sessions."""
    items: List[ChatSessionInfo]
    total: int
    page: int
    size: int