from flask import Flask, render_template, jsonify, request
import cv2
import mediapipe as mp
import torch
import numpy as np
import os
import time
from collections import Counter, deque
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from inference import GestureCNN, reshape_landmarks_for_cnn

app = Flask(__name__)
BASE_DIR = os.path.dirname(__file__)

# Initialize MediaPipe
hand_landmarker_path = os.path.join(BASE_DIR, "model", "hand_landmarker.task")
hand_landmarker_options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=hand_landmarker_path),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)
hands = mp_vision.HandLandmarker.create_from_options(hand_landmarker_options)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(BASE_DIR, "model", "gesture_recognition_model.pth")
checkpoint = torch.load(model_path, map_location=device)
class_mapping = checkpoint['class_mapping']
idx_to_class = {v: k for k, v in class_mapping.items()}
num_classes = len(class_mapping)

# Initialize model
model = GestureCNN(num_classes=num_classes, grid_size=7)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# For prediction stabilization
smoothing_window = 8
min_history_for_prediction = 4
frame_confidence_threshold = 0.60
display_confidence_threshold = 0.62
margin_threshold = 0.12
stability_threshold = 0.60
prediction_history = deque(maxlen=smoothing_window)
probability_history = deque(maxlen=smoothing_window)
last_spoken_text = ""
last_prediction_time = time.time()
cooldown_period = 1.0
last_frame_timestamp_ms = 0


def format_class_name(class_name):
    if class_name.startswith("gesture_"):
        class_name = class_name[8:]
    return class_name.replace("_", " ")


def get_next_timestamp_ms():
    global last_frame_timestamp_ms

    timestamp_ms = time.monotonic_ns() // 1_000_000
    if timestamp_ms <= last_frame_timestamp_ms:
        timestamp_ms = last_frame_timestamp_ms + 1

    last_frame_timestamp_ms = timestamp_ms
    return timestamp_ms


def get_top_candidates(probabilities, top_k=3):
    ranked_indices = np.argsort(probabilities)[::-1][:top_k]
    return [
        {
            "label": format_class_name(idx_to_class[idx]),
            "confidence": float(probabilities[idx]),
        }
        for idx in ranked_indices
    ]


def clear_prediction_state():
    prediction_history.clear()
    probability_history.clear()


def distance_2d(a, b):
    return float(np.linalg.norm(np.array([a.x - b.x, a.y - b.y], dtype=np.float32)))


def is_finger_extended(hand_landmarks, mcp_idx, pip_idx, tip_idx):
    wrist = hand_landmarks[0]
    mcp = hand_landmarks[mcp_idx]
    pip = hand_landmarks[pip_idx]
    tip = hand_landmarks[tip_idx]
    return (
        distance_2d(wrist, tip) > distance_2d(wrist, pip)
        and distance_2d(wrist, pip) > distance_2d(wrist, mcp)
    )


def apply_landmark_heuristics(probabilities, hand_landmarks):
    fist_idx = next((idx for idx, label in idx_to_class.items() if format_class_name(label) == "Fist"), None)
    perfect_idx = next((idx for idx, label in idx_to_class.items() if format_class_name(label) == "Perfect"), None)
    if fist_idx is None or perfect_idx is None:
        return probabilities

    wrist = hand_landmarks[0]
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]
    palm_width = distance_2d(hand_landmarks[5], hand_landmarks[17])
    palm_height = distance_2d(wrist, hand_landmarks[9])
    hand_scale = max(palm_width, palm_height, 1e-6)

    thumb_index_distance = distance_2d(thumb_tip, index_tip) / hand_scale
    middle_extended = is_finger_extended(hand_landmarks, 9, 10, 12)
    ring_extended = is_finger_extended(hand_landmarks, 13, 14, 16)
    pinky_extended = is_finger_extended(hand_landmarks, 17, 18, 20)
    index_curled = not is_finger_extended(hand_landmarks, 5, 6, 8)

    perfect_pose = (
        thumb_index_distance < 0.32
        and index_curled
        and sum([middle_extended, ring_extended, pinky_extended]) >= 2
        and distance_2d(index_tip, middle_tip) / hand_scale > 0.32
    )

    if not perfect_pose:
        return probabilities

    adjusted = probabilities.copy()
    perfect_confidence = float(adjusted[perfect_idx])
    fist_confidence = float(adjusted[fist_idx])
    boost = max(0.18, min(0.42, fist_confidence - perfect_confidence + 0.12))
    adjusted[perfect_idx] += boost
    adjusted[fist_idx] *= 0.55
    adjusted /= adjusted.sum()
    return adjusted

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global last_spoken_text, last_prediction_time
    
    # Get image data from request
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    
    # Convert to OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Invalid image data'}), 400

    # Mirror the frame before detection to match the notebook inference path
    # and what the user sees in the browser preview.
    image = cv2.flip(image, 1)
    
    # Process with MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = hands.detect_for_video(mp_image, get_next_timestamp_ms())
    
    response = {
        'prediction': None,
        'confidence': 0,
        'stability': 0,
        'speak': False,
        'hand_detected': False,
        'landmarks': [],
        'state': 'waiting',
        'message': 'Place one hand inside the frame',
        'raw_prediction': None,
        'top_candidates': []
    }
    
    # Process hand landmarks if detected
    if results.hand_landmarks:
        response['hand_detected'] = True
        
        # Extract landmarks
        hand_landmarks = results.hand_landmarks[0]
        landmarks = []
        
        # Add landmark coordinates to response
        for lm in hand_landmarks:
            response['landmarks'].append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z
            })
            landmarks.extend([lm.x, lm.y, lm.z])

        # Reshape landmarks for CNN
        landmarks_array = np.array(landmarks, dtype=np.float32)
        reshaped_landmarks = reshape_landmarks_for_cnn(landmarks_array)
        
        # Make prediction
        with torch.no_grad():
            input_tensor = torch.tensor(reshaped_landmarks, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()
            probabilities = apply_landmark_heuristics(probabilities, hand_landmarks)

        predicted_class_idx = int(np.argmax(probabilities))
        confidence_value = float(probabilities[predicted_class_idx])
        top_candidates = get_top_candidates(probabilities)
        response['confidence'] = confidence_value
        response['raw_prediction'] = top_candidates[0]['label']
        response['top_candidates'] = top_candidates

        if predicted_class_idx in idx_to_class:
            raw_class_name = idx_to_class[predicted_class_idx]
            class_name = format_class_name(raw_class_name)

            second_best_confidence = top_candidates[1]['confidence'] if len(top_candidates) > 1 else 0.0
            confidence_margin = confidence_value - second_best_confidence
            frame_is_reliable = (
                confidence_value >= frame_confidence_threshold
                and confidence_margin >= margin_threshold
            )

            if frame_is_reliable:
                prediction_history.append(class_name)
                probability_history.append(probabilities)
            elif prediction_history:
                prediction_history.popleft()
                probability_history.popleft()

            if prediction_history and probability_history:
                averaged_probabilities = np.mean(np.stack(probability_history), axis=0)
                stabilized_idx = int(np.argmax(averaged_probabilities))
                stabilized_label = format_class_name(idx_to_class[stabilized_idx])
                stabilized_confidence = float(averaged_probabilities[stabilized_idx])
                averaged_top_candidates = get_top_candidates(averaged_probabilities, top_k=2)
                stabilized_margin = (
                    averaged_top_candidates[0]['confidence'] - averaged_top_candidates[1]['confidence']
                    if len(averaged_top_candidates) > 1 else averaged_top_candidates[0]['confidence']
                )
                prediction_counts = Counter(prediction_history)
                stabilized_count = prediction_counts[stabilized_label]
                stability_ratio = stabilized_count / len(prediction_history)
                response['stability'] = float(stability_ratio)
                
                if (
                    len(prediction_history) >= min_history_for_prediction
                    and stabilized_confidence >= display_confidence_threshold
                    and stabilized_margin >= margin_threshold / 2
                    and stability_ratio >= stability_threshold
                ):
                    response['prediction'] = stabilized_label
                    response['confidence'] = stabilized_confidence
                    response['state'] = 'stable'
                    response['message'] = 'Stable gesture detected'

                    current_time = time.time()
                    if (
                        frame_is_reliable
                        and stabilized_label == class_name
                        and stabilized_label != last_spoken_text
                        and current_time - last_prediction_time > cooldown_period
                    ):
                        last_spoken_text = stabilized_label
                        last_prediction_time = current_time
                        response['speak'] = True
                else:
                    response['state'] = 'warming_up' if frame_is_reliable else 'uncertain'
                    response['message'] = (
                        'Hold the gesture steady for a moment'
                        if frame_is_reliable else 'Gesture is not clear yet'
                    )
            else:
                response['state'] = 'warming_up' if frame_is_reliable else 'uncertain'
                response['message'] = (
                    'Hold the gesture steady for a moment'
                    if frame_is_reliable else 'Gesture is not clear yet'
                )
    else:
        clear_prediction_state()
        response['prediction'] = 'No hand detected'
        response['state'] = 'waiting'
        response['message'] = 'Place one hand inside the frame'

    
    return jsonify(response)

if __name__ == '__main__':
    os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
    app.run(debug=True)
