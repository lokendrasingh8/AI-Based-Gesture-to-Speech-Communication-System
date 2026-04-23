// DOM elements
const video = document.getElementById('webcam');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const predictionElement = document.getElementById('prediction');
const predictionHint = document.getElementById('predictionHint');
const confidenceElement = document.getElementById('confidence');
const stabilityElement = document.getElementById('stability');
const confidenceBar = document.getElementById('confidenceBar');
const stabilityBar = document.getElementById('stabilityBar');
const statusIndicator = document.getElementById('statusIndicator');
const statusText = document.getElementById('statusText');
const lastSpoken = document.getElementById('lastSpoken');
// Create canvas overlay for landmarks
const canvasOverlay = document.createElement('canvas');
canvasOverlay.style.position = 'absolute';
canvasOverlay.style.top = '0';
canvasOverlay.style.left = '0';
canvasOverlay.style.pointerEvents = 'none';
document.querySelector('.video-container').appendChild(canvasOverlay);

// Global variables
let stream = null;
let isRunning = false;
let processingFrame = false;
let processingInterval = null;
let speechEnabled = 'speechSynthesis' in window;
let lastSpokenLabel = null;

// Start the webcam
startBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480,
                facingMode: 'user'
            }
        });
        
        video.srcObject = stream;
        video.style.transform = "scale(-1, 1)";
        await video.play();
        await new Promise(resolve => {
            if (video.videoWidth > 0 && video.videoHeight > 0) {
                resolve();
                return;
            }
            video.onloadedmetadata = () => resolve();
        });
        isRunning = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        
        statusIndicator.classList.remove('status-inactive');
        statusIndicator.classList.add('status-active');
        statusText.textContent = 'Camera active, detecting gestures...';
        
        // Start processing frames
        processFrame();
        processingInterval = setInterval(processFrame, 250);
    } catch (err) {
        console.error('Error accessing webcam:', err);
        alert('Could not access webcam. Please check permissions.');
    }
});

// Stop the webcam
stopBtn.addEventListener('click', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        isRunning = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        lastSpokenLabel = null;
        
        statusIndicator.classList.remove('status-active');
        statusIndicator.classList.add('status-inactive');
        statusText.textContent = 'Camera stopped';
        predictionHint.textContent = 'Start the camera to begin detection.';
        
        clearInterval(processingInterval);
    }
});

// Process a frame from the webcam
async function processFrame() {
    if (!isRunning || processingFrame) return;
    
    processingFrame = true;
    
    try {
        // Create a canvas to capture the current frame
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        if (!canvas.width || !canvas.height) {
            return;
        }
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to blob
        const blob = await new Promise(resolve => {
            canvas.toBlob(resolve, 'image/jpeg');
        });
        
        // Create form data for the API request
        const formData = new FormData();
        formData.append('image', blob, 'frame.jpg');
        
        // Send the frame to the server for processing
        const response = await fetch('/process_frame', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update the UI with the prediction results
        updateUI(data);
        
        // Play speech if needed
        if (data.speak && data.prediction !== lastSpokenLabel) {
            lastSpokenLabel = data.prediction;
            speakPrediction(data.prediction);
        }
    } catch (err) {
        console.error('Error processing frame:', err);
        statusText.textContent = 'Processing error. Check the Flask console.';
    } finally {
        processingFrame = false;
    }
}

// Draw landmarks on canvas overlay
function drawLandmarks(landmarks) {
    const ctx = canvasOverlay.getContext('2d');
    const width = video.videoWidth;
    const height = video.videoHeight;
    
    // Set canvas dimensions to match video
    canvasOverlay.width = width;
    canvasOverlay.height = height;
    
    // Clear previous drawings
    ctx.clearRect(0, 0, width, height);
    
    if (!landmarks || landmarks.length === 0) return;
    
    // Define connections between landmarks (MediaPipe hand connections)
    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],           // thumb
        [0, 5], [5, 6], [6, 7], [7, 8],           // index finger
        [0, 9], [9, 10], [10, 11], [11, 12],      // middle finger
        [0, 13], [13, 14], [14, 15], [15, 16],    // ring finger
        [0, 17], [17, 18], [18, 19], [19, 20],    // pinky
        [5, 9], [9, 13], [13, 17],                // palm
        [0, 5], [0, 17]                           // wrist connections
    ];
    
    // Draw connections
    ctx.strokeStyle = 'rgb(255, 0, 0)';
    ctx.lineWidth = 2;
    
    for (const [i, j] of connections) {
        if (i >= landmarks.length || j >= landmarks.length) continue;
        
        const start = landmarks[i];
        const end = landmarks[j];
        
        ctx.beginPath();
        ctx.moveTo(start.x * width, start.y * height);
        ctx.lineTo(end.x * width, end.y * height);
        ctx.stroke();
    }
    
    // Draw landmarks
    ctx.fillStyle = 'rgb(0, 255, 0)';
    
    landmarks.forEach(landmark => {
        ctx.beginPath();
        ctx.arc(landmark.x * width, landmark.y * height, 5, 0, 2 * Math.PI);
        ctx.fill();
    });
}

// Update the UI with prediction results
function updateUI(data) {
    if (data.hand_detected) {
        statusText.textContent = data.message || 'Hand detected';
        if (data.landmarks) {
            drawLandmarks(data.landmarks);
        }

        const confidencePercent = Math.round((data.confidence || 0) * 100);
        const stabilityPercent = Math.round((data.stability || 0) * 100);
        confidenceElement.textContent = `${confidencePercent}%`;
        stabilityElement.textContent = `${stabilityPercent}%`;
        confidenceBar.style.width = `${confidencePercent}%`;
        stabilityBar.style.width = `${stabilityPercent}%`;

        if (data.state === 'stable' && data.prediction) {
            predictionElement.textContent = data.prediction;
            predictionHint.textContent = data.message || 'Stable gesture detected';

            if (data.speak) {
                lastSpoken.textContent = data.prediction;
            }
        } else if (data.state === 'warming_up') {
            predictionElement.textContent = 'Hold steady...';
            predictionHint.textContent = data.raw_prediction
                ? `${data.message} Best guess: ${data.raw_prediction}`
                : (data.message || 'Reading hand...');
        } else {
            predictionElement.textContent = 'Unclear gesture';
            predictionHint.textContent = data.raw_prediction
                ? `${data.message} Best guess: ${data.raw_prediction}`
                : (data.message || 'Gesture is not clear yet');
        }
    } else {
        const ctx = canvasOverlay.getContext('2d');
        ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);
        predictionElement.textContent = 'No hand detected';
        predictionHint.textContent = data.message || 'Place one hand inside the frame';
        confidenceElement.textContent = '0%';
        stabilityElement.textContent = '0%';
        confidenceBar.style.width = '0%';
        stabilityBar.style.width = '0%';
        statusText.textContent = data.message || 'Place one hand inside the frame';
    }
}

function speakPrediction(text) {
    if (!speechEnabled || !text) {
        return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text.replaceAll('_', ' '));
    utterance.lang = 'en-US';
    window.speechSynthesis.speak(utterance);
}


// Initialize the application
function init() {
    // Check if the browser supports getUserMedia
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert('Your browser does not support webcam access. Please try a different browser.');
        startBtn.disabled = true;
        return;
    }

    const testAudioBtn = document.getElementById('testAudioBtn');
    if (testAudioBtn) {
        testAudioBtn.addEventListener('click', () => {
            if (!speechEnabled) {
                alert('Speech synthesis is not supported in this browser.');
                return;
            }
            speakPrediction('System initialized');
        });
    }
}

// Start the application when the page loads
window.addEventListener('load', init);
