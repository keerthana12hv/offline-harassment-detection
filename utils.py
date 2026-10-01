# utils.py
import os
import numpy as np
import librosa
import cv2

# MobileNet imports
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model

# --------- Audio feature extraction ---------
def extract_audio_features(file_path,
                           sr=22050,
                           n_mfcc=40,
                           max_frames=130):
    """
    Extract MFCC audio features shaped (n_mfcc, max_frames, 1).
    Pads/truncates frames to exactly max_frames.
    Returns: numpy array dtype float32 or None on error.
    """
    try:
        # load whole audio (let librosa use default sampling if sr=None is desired)
        audio, sr_used = librosa.load(file_path, sr=sr)
        # compute MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=sr_used, n_mfcc=n_mfcc)
        # pad or truncate time dimension to max_frames
        if mfcc.shape[1] < max_frames:
            pad_width = max_frames - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
        else:
            mfcc = mfcc[:, :max_frames]
        # ensure dtype float32 and add channel axis
        mfcc = mfcc.astype(np.float32)
        mfcc = np.expand_dims(mfcc, axis=-1)   # (n_mfcc, max_frames, 1)
        return mfcc
    except Exception as e:
        # Helpful debug print — in production you may want to log instead
        print(f"Audio Feature Extraction Error ({file_path}): {e}")
        # Return zeros of the correct shape so downstream code doesn't crash
        return np.zeros((n_mfcc, max_frames, 1), dtype=np.float32)

# --------- Video feature extraction (MobileNetV2) ---------
# Cache the MobileNetV2 model to avoid re-loading
_mobilenet_extractor = None

def _get_mobilenet_extractor():
    global _mobilenet_extractor
    if _mobilenet_extractor is None:
        # include_top=False and pooling='avg' gives a 1280-d vector per image
        _mobilenet_extractor = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    return _mobilenet_extractor

def extract_video_features(video_path,
                           max_frames=60,
                           target_size=(224, 224),
                           frame_sample_rate=1):
    """
    Extract per-frame CNN features using MobileNetV2.
    - max_frames: number of frames (timesteps) to output (pads with zeros if fewer).
    - target_size: (width, height) to resize frame for MobileNet.
    - frame_sample_rate: sample one frame every `frame_sample_rate` frames (1 = every frame).
    Returns: np.array shape (max_frames, 1280), dtype float32
    """
    try:
        extractor = _get_mobilenet_extractor()
        cap = cv2.VideoCapture(video_path)
        frames_feats = []
        total_frames_read = 0
        extracted_count = 0

        if not cap.isOpened():
            raise ValueError("Unable to open video file")

        while extracted_count < max_frames:
            # Read frames in a loop; we will sample according to frame_sample_rate
            ret, frame = cap.read()
            if not ret:
                break
            total_frames_read += 1

            # sample according to frame_sample_rate
            if (total_frames_read - 1) % frame_sample_rate != 0:
                continue

            # convert BGR -> RGB
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception:
                # if frame conversion fails, skip this frame
                continue

            # resize to mobilenet input
            frame_resized = cv2.resize(frame_rgb, target_size)
            # preprocess and predict feature
            x = np.asarray(frame_resized, dtype=np.float32)
            x = preprocess_input(x)          # mobilenet preprocessing
            x = np.expand_dims(x, axis=0)    # shape (1, H, W, 3)
            feat = extractor.predict(x, verbose=0)  # shape (1,1280)
            feat = feat.reshape(-1)          # (1280,)
            frames_feats.append(feat)
            extracted_count += 1

            # safety: break if video is extremely long but we've got enough frames
            if extracted_count >= max_frames:
                break

        cap.release()

        # If we extracted fewer frames than max_frames, pad with zeros
        if len(frames_feats) < max_frames:
            n_missing = max_frames - len(frames_feats)
            # If no frames extracted at all, return zero array
            if len(frames_feats) == 0:
                return np.zeros((max_frames, 1280), dtype=np.float32)
            pad = [np.zeros(1280, dtype=np.float32) for _ in range(n_missing)]
            frames_feats.extend(pad)

        # Truncate if longer (shouldn't be needed)
        frames_feats = frames_feats[:max_frames]

        features = np.array(frames_feats, dtype=np.float32)  # (max_frames, 1280)
        return features

    except Exception as e:
        print(f"Video Feature Extraction Error ({video_path}): {e}")
        # return zeros of expected shape to avoid crashing downstream; consider returning None instead if preferred
        return np.zeros((max_frames, 1280), dtype=np.float32)

# --------- small helper functions (optional) ---------
def get_video_input_shape_for_model():
    """
    Returns expected shape for the video model's single sample (timesteps, features).
    """
    return (60, 1280)

def get_audio_input_shape_for_model():
    """
    Returns expected shape for the audio model input (n_mfcc, frames, channels).
    """
    return (40, 130, 1)
