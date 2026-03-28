from functools import lru_cache

from kb_config import load_settings
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from kb_chatbot.retriever import get_retriever
from kb_chatbot.rag_chain import build_rag_chain
from kb_chatbot.session_store import get_session_memory

load_settings()

app = FastAPI(title="Knowledge Base Chatbot")


CHAT_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Project RAG Chatbot</title>
    <style>
        :root {
            --bg: #efe7da;
            --panel: rgba(255, 250, 242, 0.94);
            --surface: #fffdf8;
            --text: #1b1b18;
            --muted: #5f5b55;
            --accent: #0d6e6e;
            --accent-strong: #084c4c;
            --user: #d9ebff;
            --assistant: #edf7f1;
            --border: rgba(83, 69, 51, 0.18);
            --shadow: 0 24px 60px rgba(54, 41, 22, 0.14);
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: Georgia, "Times New Roman", serif;
            color: var(--text);
            background:
                radial-gradient(circle at top left, rgba(255, 219, 173, 0.85), transparent 32%),
                radial-gradient(circle at bottom right, rgba(163, 230, 220, 0.9), transparent 28%),
                linear-gradient(180deg, #f6f0e8 0%, #eadfce 100%);
            padding: 28px;
        }

        .shell {
            max-width: 1120px;
            margin: 0 auto;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 28px;
            overflow: hidden;
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
        }

        .hero {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 20px;
            padding: 28px;
            background: linear-gradient(135deg, rgba(255, 248, 235, 0.95), rgba(234, 251, 247, 0.9));
            border-bottom: 1px solid var(--border);
        }

        .hero h1 {
            margin: 0;
            font-size: clamp(2.2rem, 4vw, 3.6rem);
            line-height: 0.96;
            letter-spacing: -0.04em;
            max-width: 10ch;
        }

        .hero p {
            margin: 14px 0 0;
            max-width: 56ch;
            color: var(--muted);
            font-size: 1rem;
            line-height: 1.6;
        }

        .hero-card {
            align-self: end;
            padding: 18px;
            border: 1px solid var(--border);
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.65);
        }

        .hero-card strong {
            display: block;
            margin-bottom: 10px;
            font-size: 0.95rem;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: var(--accent-strong);
        }

        .hero-card p {
            margin: 0;
            font-size: 0.95rem;
        }

        .layout {
            display: grid;
            grid-template-columns: 280px 1fr;
            min-height: 68vh;
        }

        .sidebar {
            padding: 24px;
            border-right: 1px solid var(--border);
            background: rgba(255, 252, 246, 0.85);
        }

        .sidebar h2 {
            margin: 0 0 14px;
            font-size: 1rem;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: var(--accent-strong);
        }

        .sidebar label {
            display: block;
            margin: 14px 0 8px;
            font-size: 0.9rem;
            color: var(--muted);
        }

        .sidebar input,
        .sidebar textarea {
            width: 100%;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 12px 14px;
            font: inherit;
            color: var(--text);
            background: var(--surface);
        }

        .sidebar textarea {
            min-height: 140px;
            resize: vertical;
        }

        .sidebar button {
            width: 100%;
            margin-top: 16px;
            border: 0;
            border-radius: 16px;
            padding: 14px 16px;
            font: inherit;
            font-weight: 700;
            color: #fff;
            background: linear-gradient(135deg, var(--accent), #16867c);
            cursor: pointer;
            transition: transform 120ms ease, box-shadow 120ms ease;
            box-shadow: 0 12px 24px rgba(13, 110, 110, 0.24);
        }

        .sidebar button:hover {
            transform: translateY(-1px);
        }

        .sidebar button:disabled {
            cursor: wait;
            opacity: 0.7;
            transform: none;
        }

        .sidebar .hint {
            margin-top: 14px;
            font-size: 0.88rem;
            line-height: 1.5;
            color: var(--muted);
        }

        .chat-area {
            display: grid;
            grid-template-rows: 1fr auto;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.56), rgba(255, 255, 255, 0.84)),
                repeating-linear-gradient(180deg, transparent, transparent 26px, rgba(0, 0, 0, 0.02) 27px);
        }

        .messages {
            padding: 24px;
            overflow-y: auto;
            display: grid;
            gap: 16px;
        }

        .message {
            max-width: min(86%, 760px);
            padding: 16px 18px;
            border-radius: 22px;
            line-height: 1.6;
            white-space: pre-wrap;
            animation: slideIn 180ms ease-out;
            border: 1px solid transparent;
        }

        .message.user {
            margin-left: auto;
            background: var(--user);
            border-bottom-right-radius: 8px;
            border-color: rgba(70, 122, 188, 0.18);
        }

        .message.assistant {
            background: var(--assistant);
            border-bottom-left-radius: 8px;
            border-color: rgba(55, 112, 82, 0.18);
        }

        .message-meta {
            display: block;
            margin-bottom: 8px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: var(--muted);
        }

        .images {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-top: 14px;
        }

        .images a {
            display: block;
            text-decoration: none;
            color: inherit;
        }

        .images img {
            width: 100%;
            aspect-ratio: 4 / 3;
            object-fit: cover;
            border-radius: 16px;
            border: 1px solid var(--border);
            background: #fff;
        }

        .footer-bar {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding: 14px 24px 22px;
            font-size: 0.9rem;
            color: var(--muted);
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(6px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @media (max-width: 860px) {
            body {
                padding: 16px;
            }

            .hero,
            .layout {
                grid-template-columns: 1fr;
            }

            .sidebar {
                border-right: 0;
                border-bottom: 1px solid var(--border);
            }

            .message {
                max-width: 100%;
            }
        }
    </style>
</head>
<body>
    <main class="shell">
        <section class="hero">
            <div>
                <h1>Knowledge Base Chatbot</h1>
                <p>Ask questions about your ingested documents, get a grounded answer from the RAG pipeline, and review the screenshots returned from the matching chunks.</p>
            </div>
            <aside class="hero-card">
                <strong>Live Service</strong>
                <p>This UI sends requests directly to the existing <code>/chat</code> endpoint, so the same retrieval and screenshot response flow powers both the API and this frontend.</p>
            </aside>
        </section>

        <section class="layout">
            <form class="sidebar" id="chat-form">
                <h2>Ask Anything</h2>

                <label for="session-id">Session ID</label>
                <input id="session-id" name="session-id" type="text" value="render-session" />

                <label for="question">Question</label>
                <textarea id="question" name="question" placeholder="Example: What is the pivot process?"></textarea>

                <button id="send-button" type="submit">Send Question</button>

                <p class="hint">Use the same session ID for follow-up questions so the chatbot can keep conversation history.</p>
            </form>

            <section class="chat-area">
                <div class="messages" id="messages">
                    <article class="message assistant">
                        <span class="message-meta">Assistant</span>
                        Ask a question to start the conversation.
                    </article>
                </div>

                <div class="footer-bar">
                    <span>Health check: <a href="/health">/health</a></span>
                    <span>API docs: <a href="/docs">/docs</a></span>
                </div>
            </section>
        </section>
    </main>

    <script>
        const form = document.getElementById("chat-form");
        const messages = document.getElementById("messages");
        const questionInput = document.getElementById("question");
        const sessionInput = document.getElementById("session-id");
        const sendButton = document.getElementById("send-button");

        function appendMessage(role, text, images = []) {
            const article = document.createElement("article");
            article.className = `message ${role}`;

            const meta = document.createElement("span");
            meta.className = "message-meta";
            meta.textContent = role === "user" ? "You" : "Assistant";
            article.appendChild(meta);

            const content = document.createElement("div");
            content.textContent = text;
            article.appendChild(content);

            if (images.length > 0) {
                const imageGrid = document.createElement("div");
                imageGrid.className = "images";

                for (const imageUrl of images) {
                    const link = document.createElement("a");
                    link.href = imageUrl;
                    link.target = "_blank";
                    link.rel = "noreferrer";

                    const image = document.createElement("img");
                    image.src = imageUrl;
                    image.alt = "Retrieved screenshot";
                    image.loading = "lazy";

                    link.appendChild(image);
                    imageGrid.appendChild(link);
                }

                article.appendChild(imageGrid);
            }

            messages.appendChild(article);
            messages.scrollTop = messages.scrollHeight;
        }

        form.addEventListener("submit", async (event) => {
            event.preventDefault();

            const question = questionInput.value.trim();
            const sessionId = sessionInput.value.trim() || "default";
            if (!question) {
                return;
            }

            appendMessage("user", question);
            questionInput.value = "";
            sendButton.disabled = true;
            sendButton.textContent = "Thinking...";

            try {
                const response = await fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: sessionId, question })
                });

                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(errorText || "Request failed.");
                }

                const payload = await response.json();
                appendMessage("assistant", payload.answer || "No answer returned.", payload.images || []);
            } catch (error) {
                appendMessage("assistant", `Request failed: ${error.message}`);
            } finally {
                sendButton.disabled = false;
                sendButton.textContent = "Send Question";
            }
        });
    </script>
</body>
</html>
"""


@lru_cache(maxsize=1)
def get_runtime():
    retriever = get_retriever()
    rag_chain = build_rag_chain(retriever, get_session_memory)
    return retriever, rag_chain

class Query(BaseModel):
    session_id: str = "default"
    question: str


@app.get("/", response_class=HTMLResponse)
def root():
    return CHAT_UI_HTML


@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/chat")
def chat(query: Query):
    _, rag_chain = get_runtime()
    result = rag_chain.invoke(
        {"question": query.question},
        config={"configurable": {"session_id": query.session_id}},
    )

    image_urls = []
    for doc in result["source_docs"]:
        image_urls.extend(doc.metadata.get("image_urls", []))

    return {
        "answer": result["answer"],
        "images": list(set(image_urls))
    }

