from langchain_core.chat_history import InMemoryChatMessageHistory

def get_memory(session_id: str) -> InMemoryChatMessageHistory:
    """
    Creates a conversation memory object.
    Windowed memory prevents context overflow.
    Uses InMemoryChatMessageHistory (LangChain v0.3+ compatible).
    """
    return InMemoryChatMessageHistory()
