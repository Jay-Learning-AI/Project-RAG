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
            let errorMessage = "Request failed.";

            try {
                const errorPayload = await response.json();
                errorMessage = errorPayload.detail || JSON.stringify(errorPayload);
            } catch (parseError) {
                const errorText = await response.text();
                errorMessage = errorText || errorMessage;
            }

            throw new Error(errorMessage);
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