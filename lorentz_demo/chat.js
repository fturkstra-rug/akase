function sendServerMessage(message, speaker) {
    if (message) {
        // Find the last punctuation mark (., !, ?)
        // let lastPunctuationIndex = Math.max(
        //     message.lastIndexOf('.'),
        //     message.lastIndexOf('!'),
        //     message.lastIndexOf('?')
        // );

        // // If a punctuation mark is found and it's not the last character, truncate
        // if (lastPunctuationIndex !== -1 && lastPunctuationIndex !== message.length - 1) {
        //     message = message.substring(0, lastPunctuationIndex + 1);
        // }

        appendMessage(message, speaker);
    }
}

let messageList = [];
let messagesAreHidden = false;

function sendUserMessage() {
    let inputField = document.getElementById("chat-message");
    let send_btn = document.getElementById("send-button");
    let mic_btn = document.getElementById("mic-button");
    let user_turn = document.getElementById("user-turn-status");
    let message = inputField.value.trim();
    let speaker = {"name": "You", "role": "deliberator"};

    if (message && !send_btn.disabled) {
        appendMessage(message, speaker);
        
        // Sends message to dynamoDB
        socket.send(JSON.stringify({
            action: "storeClientMessage",
            message: message
        }));
        console.log("Send message to server.", message);
    }
    
    // Disable send functionality
    send_btn.disabled = true;
    mic_btn.disabled = true;
    send_btn.title = "Wait for the moderator to request user input";
    mic_btn.title = "Wait for the moderator to request user input";
    user_turn.style.display = "none";
    inputField.value = "";
}

function saveChatMessages() {
    const topic = document.querySelector('h1')?.innerText || "Untitled";
    const timestamp = new Date().toISOString();

    // Extract text and speaker info
    const chat = Array.from(messageList).map(message => {
        const text = message.querySelector('.message-content p')?.innerText.trim() || "No message";
        const speakerElement = message.querySelector('.hidden-speaker-info')
        const speakerInfo = speakerElement ? JSON.parse(speakerElement.innerText) : { name: "Unknown", role: "Unknown" };

        return { text, speaker: speakerInfo };
    });

    // Store the information in the required format
    const chatData = {
        topic,
        timestamp,
        chat
    };

     // Convert the JSON object to a string
     const jsonString = JSON.stringify(chatData, null, 4);

     // Create a Blob and a link to download it
     const blob = new Blob([jsonString], { type: 'application/json' });
     const link = document.createElement('a');
     link.href = URL.createObjectURL(blob);
     link.download = `chat_${timestamp}.json`; // Filename with timestamp
     document.body.appendChild(link);
     link.click();
     document.body.removeChild(link); // Clean up

     console.log("Successfully exported chat to json file.")
}

document.addEventListener("keydown", function(event) {
    let inputField = document.getElementById("chat-message");

    if (event.key === "Enter") {
        if (document.activeElement !== inputField) {
            // Focus the input field if it's not active
            event.preventDefault()
            inputField.focus();
        } else {
            // Send the message if the input is already focused unless shift is pressed, then add a newline
            if (event.shiftKey) {
                inputField.value += '\n';
            } else {
                event.preventDefault()
                sendUserMessage();
            }
        }
    }
});

let chatContainer = document.querySelector(".chat-container");
scrollButton = document.getElementById("scroll-to-bottom-button");

function scrollToBottom() {
    chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: "smooth" });
    scrollButton.style.display = "none"; // Hide after scrolling down
    hideAllButLastMessage();
}

function appendMessage(msg, speaker) {
    const allowedSpeakers = ["walton", "eemeren", "hypatia", "socrates", "cicero", "you"]; 
    let profile_picture = allowedSpeakers.includes(speaker["name"].toLowerCase()) 
        ? `${speaker["name"].toLowerCase()}.png` 
        : "default.png";
    let isScrolledToBottom = chatContainer.scrollHeight - chatContainer.clientHeight <= chatContainer.scrollTop + 1;

    let messageDiv = document.createElement("div");
    messageDiv.textContent = msg;
    chatContainer.appendChild(messageDiv);

    messageDiv.innerHTML = `
        <div class="message">
            <div class="profile-pic-column">
                <img src="images/${profile_picture}" alt="Profile Picture" class="profile-pic">
            </div>
            <div class="message-content">
                <div class="speaker-info">${speaker["name"]} (${speaker["role"]})</div>
                <p>${msg}</p>
            </div>
            <div class="hidden-speaker-info">
                ${JSON.stringify(speaker)}
            </div>
        </div>
    `;

    messageList.push(messageDiv);

    hideAllButLastMessage();

    // If user is at bottom, scroll automatically
    if (isScrolledToBottom) {
        chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: "instant" });
    } else {
        scrollButton.style.display = "block"; // Show button if user is scrolled up
    }
}

// Shows/hides the scroll to bottom button depending on whether we are at the bottom or not
chatContainer.addEventListener("scroll", () => {
    let isScrolledToBottom = chatContainer.scrollHeight - chatContainer.clientHeight <= chatContainer.scrollTop + 1;
    scrollButton.style.display = isScrolledToBottom ? "none" : "block";
});

function hideAllButLastMessage() {
    messageList.forEach((msg, index) => {
        if (index !== messageList.length - 1) {
            msg.style.display = "none";
        }
    });
    messagesAreHidden = true;
}

// Detect the wheel scroll event to trigger showing all messages when scrolling up
chatContainer.addEventListener("wheel", function (event) {
    // If the user scrolls up, show all messages unless they are already visible
    if (event.deltaY < 0 && messagesAreHidden) {
        messageList.forEach(msg => msg.style.display = "block");
        messagesAreHidden = false;
        // After all messages are made visible, the screen automatically scrolls up to the first message.
        // Therefore, we automatically scroll down to the bottom/most recent message again.
        // This does not trigger the scroll to bottom button so should we trigger it here?
        chatContainer.scrollTo({ top: chatContainer.scrollHeight, behavior: "instant" });
    }

    // If the user scrolls down to the bottom, hide all messages except the latest
    if (event.deltaY > 0 && chatContainer.scrollTop + chatContainer.clientHeight >= chatContainer.scrollHeight) {
        hideAllButLastMessage();
    }
});
