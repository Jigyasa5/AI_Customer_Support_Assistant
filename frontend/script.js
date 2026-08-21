const API_URL = "http://127.0.0.1:8000";

async function askQuestion() {

    const input = document.getElementById("question");

    const question = input.value.trim();

    if (!question) {
        return;
    }

    addMessage(question, "user");

    input.value = "";

    const token = localStorage.getItem("access_token");

    if (!token) {

        addMessage(
            "Please login first.",
            "bot"
        );

        return;
    }

    try {

        const response = await fetch(
            `${API_URL}/ask`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${token}`
                },

                body: JSON.stringify({
                    question: question
                })
            }
        );

        const data = await response.json();

        if (!response.ok) {

            addMessage(
                data.detail || "Something went wrong.",
                "bot"
            );

            return;
        }

        addMessage(
            data.answer,
            "bot"
        );

    } catch (error) {

        console.error(error);

        addMessage(
            "Unable to connect to the backend.",
            "bot"
        );
    }
}


function addMessage(text, type) {

    const chatBox =
        document.getElementById("chatBox");

    const message =
        document.createElement("div");

    message.className =
        `message ${type}`;

    message.textContent = text;

    chatBox.appendChild(message);

    chatBox.scrollTop =
        chatBox.scrollHeight;
}


document
    .getElementById("question")
    .addEventListener(
        "keypress",
        function(event) {

            if (event.key === "Enter") {

                askQuestion();

            }

        }
    );