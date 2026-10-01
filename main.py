import numpy as np
import os
from tensorflow.keras.models import load_model

# Import your utils functions (utils.py must be in the same directory)
from utils import extract_audio_features, extract_video_features

# Load trained models
try:
    audio_model = load_model("models/audio_emotion_model.keras")
    video_model = load_model("models/violence_detection_bilstm.h5")

except Exception as e:
    print("Error loading models:", e)
    audio_model = None
    video_model = None

# Emotion label mapping (update as per your audio model)
label_dict = {
    0: "Neutral", 1: "Calm", 2: "Happy", 3: "Sad",
    4: "Angry", 5: "Fearful", 6: "Disgust", 7: "Surprised"
    # Adjust indices as per your model's output!
}

def predict_harassment(audio_path=None, video_path=None):
    """
    Predicts harassment for given audio and/or video files.
    Either audio_path or video_path must be provided.
    """
    audio_pred = None
    audio_class = None
    video_pred = None

    # Process audio
    if audio_path and audio_model:
        audio_features = extract_audio_features(audio_path)
        if audio_features is not None:
            # Reshape features as required by your audio model
            # Example: audio_features = np.expand_dims(audio_features, axis=0)
            # Adjust as per your model's input shape!
            audio_features = audio_features.reshape(1, -1)
            audio_probs = audio_model.predict(audio_features, verbose=0)
            audio_pred = np.argmax(audio_probs)
            audio_class = label_dict.get(audio_pred, "Unknown")

    # Process video
    if video_path and video_model:
        video_features = extract_video_features(video_path)
        if video_features is not None:
            # Reshape features as required by your video model
            # Example: video_features = video_features.reshape(1, -1)
            # For BiLSTM, you may need to reshape to (1, timesteps, features)
            # Adjust as per your model's input shape!
            # If your video model expects (batch, timesteps, features), you might need:
            # video_features = video_features.reshape(1, 1, -1)
            # Or, if your extract_video_features returns a sequence, you might do:
            # video_features = video_features.reshape(1, *video_features.shape)
            # You must adjust this to match your model's expected input!
            video_probs = video_model.predict(video_features.reshape(1, -1), verbose=0)
            video_pred = np.argmax(video_probs)

    # Determine result
    if video_pred is not None and video_pred == 1:  # Assuming 1 is violence
        return "⚠️ Harassment Likely Detected (Video)"
    elif audio_pred is not None and audio_pred in [4, 5, 6]:  # Angry, Fearful, Disgust
        return f"⚠️ Harassment Likely Detected (Audio: {audio_class})"                                              
    elif video_pred is not None and audio_pred is None:
        return "⚠️ Video indicates possible harassment (audio unreadable)"
    else:
        return "✅ No Harassment Detected"

if __name__ == "__main__":
    # Example usage (modify paths as needed)
    audio_file = "example.wav"  # Set to None if no audio
    video_file = "example.mp4"  # Set to None if no video

    result = predict_harassment(audio_file, video_file)
    print(result)
