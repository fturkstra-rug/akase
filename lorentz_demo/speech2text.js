let recognition;
let isListening = false;

// Check if the browser supports SpeechRecognition
if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    let finalTranscript = '';

    recognition.onresult = (event) => {
        let interimTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript + ' ';
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }
        document.getElementById('chat-message').value = finalTranscript + interimTranscript;
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
    };

    document.getElementById('mic-button').addEventListener('click', () => {
        if (!isListening) {
            finalTranscript = '';
            recognition.start();
            isListening = true;
            document.getElementById('mic-button').classList.replace("off", "on");
        } else {
            recognition.stop();
            isListening = false;
            document.getElementById('mic-button').classList.replace("on", "off");
            sendUserMessage();
        }
    });
} else {
    console.warn('Speech recognition is not supported in this browser.');
}
