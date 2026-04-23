from gtts import gTTS
import os

# Set of gesture class names (replace with your actual gesture labels)
gesture_labels = ["hello", "yes", "no", "thank_you", "ok", "stop"]

# Output directory (inside static folder)
output_dir = os.path.join("static", "voices")
os.makedirs(output_dir, exist_ok=True)

for label in gesture_labels:
    tts = gTTS(text=label.replace('_', ' '), lang='en')
    tts.save(os.path.join(output_dir, f"{label}.mp3"))
    print(f"Saved {label}.mp3")
