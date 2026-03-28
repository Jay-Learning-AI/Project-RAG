const form = document.getElementById("chat-form");
const messages = document.getElementById("messages");
const questionInput = document.getElementById("question");
const sessionInput = document.getElementById("session-id");
const sendButton = document.getElementById("send-button");

function extractSteps(text) {
    const matches = [...text.matchAll(/(?:^|\n)(\d+)\.\s([\s\S]*?)(?=(?:\n\d+\.\s)|$)/g)];
    return matches.map((match) => match[2].trim()).filter(Boolean);
}

function appendImageGrid(container, images = [], title = "") {
    if (!images.length) {
        return;
    }

    if (title) {
        const heading = document.createElement("div");
        heading.className = "image-section-title";
        heading.textContent = title;
        container.appendChild(heading);
    }

    const imageGrid = document.createElement("div");
    imageGrid.className = "images";

    for (const imageUrl of images) {
        const card = document.createElement("div");
        card.className = "image-card";

        const link = document.createElement("a");
        link.href = imageUrl;
        link.target = "_blank";
        link.rel = "noreferrer";
        link.className = "image-link";

        const image = document.createElement("img");
        image.src = imageUrl;
        image.alt = "Retrieved screenshot";
        image.loading = "lazy";
        image.addEventListener("error", () => {
            card.classList.add("image-card-error");
            image.remove();
            const fallback = document.createElement("span");
            fallback.className = "image-fallback";
            fallback.textContent = "Screenshot unavailable";
            if (!link.querySelector(".image-fallback")) {
                link.appendChild(fallback);
            }
        }, { once: true });

        link.appendChild(image);
        card.appendChild(link);
        imageGrid.appendChild(card);
    }

    container.appendChild(imageGrid);
}

function appendGuidedResponse(article, text, imageSections = []) {
    const steps = extractSteps(text);
    if (!steps.length) {
        const content = document.createElement("div");
        content.textContent = text;
        article.appendChild(content);

        if (imageSections.length) {
            const fallbackImages = imageSections.flatMap((section) => section.images || []);
            appendImageGrid(article, fallbackImages, "Related Screenshots");
        }
        return;
    }

    const guide = document.createElement("div");
    guide.className = "guide";

    steps.forEach((stepText, index) => {
        const step = document.createElement("section");
        step.className = "guide-step";

        const stepLabel = document.createElement("div");
        stepLabel.className = "guide-step-label";
        stepLabel.textContent = `Step ${index + 1}`;
        step.appendChild(stepLabel);

        const stepBody = document.createElement("div");
        stepBody.className = "guide-step-body";
        stepBody.textContent = stepText;
        step.appendChild(stepBody);

        const section = imageSections[index];
        if (section && Array.isArray(section.images) && section.images.length) {
            appendImageGrid(step, section.images, section.title || `Step ${index + 1} Screenshot`);
        }

        guide.appendChild(step);
    });

    article.appendChild(guide);
}

function appendMessage(role, text, images = [], imageSections = []) {
    const article = document.createElement("article");
    article.className = `message ${role}`;

    const meta = document.createElement("span");
    meta.className = "message-meta";
    meta.textContent = role === "user" ? "You" : "Assistant";
    article.appendChild(meta);

    if (role === "assistant") {
        appendGuidedResponse(article, text, imageSections);
    } else {
        const content = document.createElement("div");
        content.textContent = text;
        article.appendChild(content);

        if (images.length > 0) {
            appendImageGrid(article, images, "Related Screenshots");
        }
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
        appendMessage(
            "assistant",
            payload.answer || "No answer returned.",
            payload.images || [],
            payload.image_sections || [],
        );
    } catch (error) {
        appendMessage("assistant", `Request failed: ${error.message}`);
    } finally {
        sendButton.disabled = false;
        sendButton.textContent = "Send Question";
    }
});