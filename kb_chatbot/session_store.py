from kb_chatbot.memory import get_memory

# Simple in-memory session store
# (Can later be replaced with Redis)
SESSION_MEMORY = {}

def get_session_memory(session_id: str):
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = get_memory(session_id)
    return SESSION_MEMORY[session_id]
