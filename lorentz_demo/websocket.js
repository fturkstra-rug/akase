const websocketUrl = "wss://vi7tdp1gw6.execute-api.us-west-2.amazonaws.com/production/";
const socket = new WebSocket(websocketUrl);
const autoPlay = true;
let isAudioPlaying = false;

socket.onopen = function(event) {
    statusDiv = document.getElementById("websocket-status")
    statusDiv.textContent = "Connected";
    statusDiv.className = "connected";
    console.log("WebSocket connection established");
};

socket.onmessage = function(event) {
    try {
        const data = JSON.parse(event.data);

        if ('audio_url' in data && 'text' in data && 'speaker' in data) {
            handleAudioMessage(data.audio_url, data.text, data.speaker);
        } else if ('userInputRequest' in data) {
            handleUserInputRequest(data.userInputRequest);
        } else if ('transcriptUrl' in data) {
            console.log('Received transcript:', data.transcriptUrl);
        } else {
            console.error("Received unknown message format:", data);
        }        
    } catch (error) {
        console.error("Error playing audio:", error);
    }
};

socket.onerror = function(event) {
    console.log("Error occurred: " + event.message);
};

socket.onclose = function(event) {
    console.log(event);
    statusDiv = document.getElementById("websocket-status")
    statusDiv.textContent = "Disconnected";
    statusDiv.className = "disconnected";
    console.log("WebSocket connection closed");
};

function handleAudioMessage(audio_url, text, speaker) {
    console.log("Received audio:", audio_url);
    console.log("Received text:", text);
    console.log("Received speaker:", speaker);
    
    const audioPlayer = document.getElementById("audioPlayer");
    audioPlayer.src = audio_url;
    if (autoPlay) {
        audioPlayer.play();
    }

    sendServerMessage(text, speaker)
}

function handleUserInputRequest(requestMessage) {
    console.log("User input requested:", requestMessage);
    
    // Enable send functionality
    let send_btn = document.getElementById("send-button");
    let mic_btn = document.getElementById("mic-button");
    let user_turn = document.getElementById("user-turn-status");
    send_btn.disabled = false;
    mic_btn.disabled = false;
    send_btn.title = "Send Message";
    mic_btn.title = "Start Listening";
    user_turn.style.display = "block";
}

// Detect when the audio is finished
audioPlayer.addEventListener("ended", function () {
    // Sets flag in dynamoDB
    socket.send(JSON.stringify({
        action: "storeClientMessage",
        message: "audio_finished_playing"
    }));
    console.log("Audio playback finished!");
});
