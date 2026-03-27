
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from kb_chatbot.prompt import RAG_PROMPT


def build_rag_chain(retriever, get_session_history):
    """
    Builds a conversational RAG chain compatible with LangChain v0.3+.

    Args:
        retriever: A LangChain retriever (e.g., from Pinecone vectorstore).
        get_session_history: A callable(session_id) -> BaseChatMessageHistory.

    Returns:
        A RunnableWithMessageHistory chain.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_PROMPT.template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["question"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
