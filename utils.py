
import os
import numpy as np
import librosa
import cv2

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input
)


# =====================================================
# AUDIO FEATURE EXTRACTION
# =====================================================

def extract_audio_features(
    file_path,
    sr=22050,
    n_mfcc=40,
    max_frames=130
):
    """
    Extract MFCC audio features.

    Output shape: (40, 130, 1)
    """

    try:
        audio, sr_used = librosa.load(
            file_path,
            sr=sr
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr_used,
            n_mfcc=n_mfcc
        )

        # Pad or truncate to 130 frames
        if mfcc.shape[1] < max_frames:
            pad_width = max_frames - mfcc.shape[1]

            mfcc = np.pad(
                mfcc,
                ((0, 0), (0, pad_width)),
                mode="constant"
            )

        else:
            mfcc = mfcc[:, :max_frames]

        # Convert to float32
        mfcc = mfcc.astype(np.float32)

        # Add channel dimension
        mfcc = np.expand_dims(mfcc, axis=-1)

        return mfcc

    except Exception as e:
        print(f"Audio Feature Extraction Error: {e}")
        return None


# =====================================================
# VIDEO FEATURE EXTRACTION USING MOBILENETV2
# =====================================================

_mobilenet_extractor = None


def _get_mobilenet_extractor():
    """
    Load MobileNetV2 without global average pooling.

    Output feature map for a 224x224 image:
    (7, 7, 1280)

    Flattened output:
    (62720,)
    """

    global _mobilenet_extractor

    if _mobilenet_extractor is None:

        _mobilenet_extractor = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling=None,
            input_shape=(224, 224, 3)
        )

    return _mobilenet_extractor


def extract_video_features(
    video_path,
    max_frames=60,
    target_size=(224, 224),
    frame_sample_rate=1
):
    """
    Extract video features using MobileNetV2.

    Steps:
    1. Read video frames.
    2. Resize frames to 224x224.
    3. Apply MobileNetV2 preprocessing.
    4. Extract CNN feature maps.
    5. Flatten each feature map.
    6. Pad or truncate to 60 frames.

    Output shape:
    (60, 62720)
    """

    cap = None

    try:

        if frame_sample_rate < 1:
            raise ValueError(
                "frame_sample_rate must be at least 1"
            )

        # Load MobileNetV2
        extractor = _get_mobilenet_extractor()

        # Calculate flattened feature dimension
        feature_dim = int(
            np.prod(extractor.output_shape[1:])
        )

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(
                "Unable to open video file"
            )

        frames_feats = []
        total_frames_read = 0

        while len(frames_feats) < max_frames:

            ret, frame = cap.read()

            if not ret:
                break

            total_frames_read += 1

            # Sample frames
            if (
                (total_frames_read - 1)
                % frame_sample_rate != 0
            ):
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Resize frame
            frame_resized = cv2.resize(
                frame_rgb,
                target_size
            )

            # Convert to float32
            x = np.asarray(
                frame_resized,
                dtype=np.float32
            )

            # MobileNetV2 preprocessing
            x = preprocess_input(x)

            # Add batch dimension
            x = np.expand_dims(x, axis=0)

            # Extract feature map
            feat = extractor.predict(
                x,
                verbose=0
            )

            # Flatten (7, 7, 1280) to (62720,)
            feat = feat.reshape(-1)

            frames_feats.append(feat)

        # Release video
        cap.release()
        cap = None

        if len(frames_feats) == 0:
            raise ValueError(
                "No valid frames extracted from video"
            )

        # Pad missing frames with zeros
        while len(frames_feats) < max_frames:

            frames_feats.append(
                np.zeros(
                    feature_dim,
                    dtype=np.float32
                )
            )

        # Keep exactly 60 frames
        frames_feats = frames_feats[:max_frames]

        # Convert to NumPy array
        features = np.array(
            frames_feats,
            dtype=np.float32
        )

        print(
            "Video feature shape:",
            features.shape
        )

        return features

    except Exception as e:

        print(
            f"Video Feature Extraction Error: {e}"
        )

        return None

    finally:

        if cap is not None:
            cap.release()


# =====================================================
# INPUT SHAPE HELPER FUNCTIONS
# =====================================================

def get_video_input_shape_for_model():
    """
    Expected video model input:
    60 frames x 62720 features
    """

    return (60, 62720)


def get_audio_input_shape_for_model():
    """
    Expected audio model input:
    40 MFCC x 130 frames x 1 channel
    """

    return (40, 130, 1)
