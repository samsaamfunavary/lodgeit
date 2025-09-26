from typing import List, Dict, Any

# WARNING: This is a simple in-memory store for demonstration.
# In a production environment, you MUST use a persistent, scalable storage
# solution like Redis, a PostgreSQL table, or a NoSQL database.
CONVERSATION_MEMORY = {}

def get_history(conversation_id: str) -> List[Dict[str, str]]:
    """Retrieves the chat history for a given conversation ID."""
    return CONVERSATION_MEMORY.get(conversation_id, [])

def update_history(conversation_id: str, user_message: str, assistant_message: str):
    """Updates the chat history for a given conversation ID."""
    if conversation_id not in CONVERSATION_MEMORY:
        CONVERSATION_MEMORY[conversation_id] = []
    
    # Append the new turn to the history
    CONVERSATION_MEMORY[conversation_id].append({"role": "user", "content": user_message})
    CONVERSATION_MEMORY[conversation_id].append({"role": "assistant", "content": assistant_message})

    # Optional: Implement a strategy to limit history size, e.g., keep last 10 messages
    # CONVERSATION_MEMORY[conversation_id] = CONVERSATION_MEMORY[conversation_id][-10:]