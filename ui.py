
import streamlit as st
import time
from pathlib import Path
import tempfile
import os
import numpy as np
import pandas as pd
import zipfile
import shutil
from datetime import datetime
import plotly.express as px

from tensorflow.keras.models import load_model
from utils import extract_audio_features, extract_video_features


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Harassment Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# CUSTOM CSS
# =====================================================

def load_custom_css():

    st.markdown("""
    <style>

    .stApp {
        background: linear-gradient(
            135deg, #667eea 0%, #764ba2 100%
        );
    }

    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }

    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 40px;
        text-align: center;
        background: rgba(255, 255, 255, 0.9);
        margin: 20px 0;
    }

    .result-safe {
        background: linear-gradient(90deg, #56ab2f, #a8e6cf);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .result-danger {
        background: linear-gradient(90deg, #ff416c, #ff4757);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .stat-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px;
        box-shadow: 0 4px 15px rgba(31, 38, 135, 0.2);
    }

    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 10px 30px;
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    </style>
    """, unsafe_allow_html=True)


load_custom_css()


# =====================================================
# LOAD KERAS MODELS
# =====================================================

@st.cache_resource
def load_models():

    base = Path(__file__).resolve().parent / "data"

    audio_path = base / "audio_emotion_model.keras"
    video_path = base / "violence_detection_bilstm.keras"

    missing = [
        p.name
        for p in (audio_path, video_path)
        if not p.exists()
    ]

    if missing:
        st.error(
            f"Missing model file(s) in data/: {', '.join(missing)}"
        )
        return None, None

    try:

        audio_model = load_model(str(audio_path))
        video_model = load_model(str(video_path))

        st.success("Audio and video models loaded successfully!")

        print("Audio model input shape:", audio_model.input_shape)
        print("Video model input shape:", video_model.input_shape)

        return audio_model, video_model

    except Exception as e:

        st.error(f"Error loading models: {e}")

        return None, None


audio_model, video_model = load_models()


# =====================================================
# VIDEO PREDICTION
# =====================================================

def get_video_prediction(model, features):

    if features is None:
        raise ValueError("Video feature extraction failed.")

    # Add batch dimension
    features = np.expand_dims(features, axis=0)

    print("Video input shape:", features.shape)
    print("Expected model shape:", model.input_shape)

    # Check input dimensions
    expected_shape = tuple(model.input_shape[1:])
    actual_shape = tuple(features.shape[1:])

    if actual_shape != expected_shape:
        raise ValueError(
            f"Video shape mismatch! "
            f"Model expects {expected_shape}, "
            f"but extracted features have {actual_shape}."
        )

    # Predict
    prob = float(model.predict(features, verbose=0)[0][0])

    print("Video model probability:", prob)

    is_violence = prob > 0.5

    return int(is_violence), prob


# =====================================================
# AUDIO PREDICTION
# =====================================================

def get_audio_prediction(model, features):

    if features is None:
        raise ValueError("Audio feature extraction failed.")

    # Add batch dimension
    features = np.expand_dims(features, axis=0)

    print("Audio input shape:", features.shape)
    print("Expected model shape:", model.input_shape)

    expected_shape = tuple(model.input_shape[1:])
    actual_shape = tuple(features.shape[1:])

    if actual_shape != expected_shape:
        raise ValueError(
            f"Audio shape mismatch! "
            f"Model expects {expected_shape}, "
            f"but extracted features have {actual_shape}."
        )

    probs = model.predict(features, verbose=0)[0]

    pred_class = int(np.argmax(probs))
    confidence = float(np.max(probs))

    print("Audio model probabilities:", probs)

    return pred_class, confidence


# =====================================================
# HARASSMENT DETECTION
# =====================================================

def detect_harassment(file_path, filename):

    ext = os.path.splitext(filename)[1].lower()

    # ---------------- VIDEO ANALYSIS ----------------

    if ext in [".mp4", ".mkv", ".mov", ".avi"]:

        if video_model is None:
            raise RuntimeError("Video model is not loaded.")

        video_feat = extract_video_features(file_path)

        if video_feat is None:
            raise ValueError("Unable to extract video features.")

        video_pred, video_conf = get_video_prediction(
            video_model,
            video_feat
        )

        is_harassment = bool(video_pred)

        return {
            "filename": filename,
            "type": "Video",
            "is_harassment": is_harassment,
            "classification": (
                "Violence Detected"
                if is_harassment
                else "Safe Content"
            ),
            "confidence": video_conf,
            "explanation": (
                "Aggressive visual patterns detected"
                if is_harassment
                else "No threatening behavior identified"
            )
        }

    # ---------------- AUDIO ANALYSIS ----------------

    elif ext in [".wav", ".mp3", ".m4a"]:

        if audio_model is None:
            raise RuntimeError("Audio model is not loaded.")

        audio_feat = extract_audio_features(file_path)

        if audio_feat is None:
            raise ValueError("Unable to extract audio features.")

        audio_pred, audio_conf = get_audio_prediction(
            audio_model,
            audio_feat
        )

        is_harassment = audio_pred in [4, 5, 6]

        return {
            "filename": filename,
            "type": "Audio",
            "is_harassment": is_harassment,
            "classification": f"Emotional State: {audio_pred}",
            "confidence": audio_conf,
            "explanation": (
                f"Audio analysis indicates emotional class {audio_pred}"
            )
        }

    # ---------------- UNSUPPORTED FORMAT ----------------

    return {
        "filename": filename,
        "type": "Unsupported",
        "is_harassment": False,
        "classification": "Unsupported Format",
        "confidence": 0.0,
        "explanation": "File format not supported"
    }


# =====================================================
# ZIP FILE EXTRACTION
# =====================================================

def extract_files_from_zip(zip_file):

    extracted_files = []
    temp_dir = tempfile.mkdtemp()

    try:

        with zipfile.ZipFile(zip_file, "r") as zip_ref:

            for member in zip_ref.infolist():

                # Ignore directories and unsafe paths
                if member.is_dir():
                    continue

                destination = os.path.realpath(
                    os.path.join(temp_dir, member.filename)
                )

                if os.path.commonpath(
                    [temp_dir, destination]
                ) != os.path.realpath(temp_dir):
                    continue

                zip_ref.extract(member, temp_dir)

        for root, _, files in os.walk(temp_dir):

            for file in files:

                if file.lower().endswith(
                    (
                        ".mp4", ".mov", ".avi", ".mkv",
                        ".wav", ".mp3", ".m4a"
                    )
                ):

                    full_path = os.path.join(root, file)

                    extracted_files.append(
                        (full_path, file)
                    )

        return extracted_files, temp_dir

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


# =====================================================
# SAVE MULTIPLE UPLOADED FILES
# =====================================================

def save_uploaded_files(uploaded_files):

    temp_dir = tempfile.mkdtemp()

    saved_files = []

    try:

        for uploaded_file in uploaded_files:

            safe_name = os.path.basename(uploaded_file.name)

            file_path = os.path.join(
                temp_dir,
                safe_name
            )

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            saved_files.append(
                (file_path, safe_name)
            )

        return saved_files, temp_dir

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


# =====================================================
# DISPLAY SINGLE FILE RESULT
# =====================================================

def display_single_result(
    result,
    show_confidence=True,
    show_explanation=True
):

    st.markdown("#### Detection Results")

    confidence_html = (
        f"<p><strong>Confidence:</strong> "
        f"{result['confidence']:.2f}</p>"
        if show_confidence else ""
    )

    explanation_html = (
        f"<p><strong>Details:</strong> "
        f"{result['explanation']}</p>"
        if show_explanation else ""
    )

    if result["is_harassment"]:

        st.markdown(f"""
        <div class="result-danger">

            <h3>⚠️ Harassment Detected</h3>

            <p><strong>Filename:</strong>
            {result['filename']}</p>

            <p><strong>Type:</strong>
            {result['type']}</p>

            <p><strong>Classification:</strong>
            {result['classification']}</p>

            {confidence_html}

            {explanation_html}

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-safe">

            <h3>✅ Content Safe</h3>

            <p><strong>Filename:</strong>
            {result['filename']}</p>

            <p><strong>Type:</strong>
            {result['type']}</p>

            <p><strong>Classification:</strong>
            {result['classification']}</p>

            {confidence_html}

            {explanation_html}

        </div>
        """, unsafe_allow_html=True)


# =====================================================
# DISPLAY BATCH RESULTS
# =====================================================

def display_batch_results(
    results,
    show_confidence=True,
    show_explanation=True
):

    df = pd.DataFrame(results)

    df["Status"] = df["is_harassment"].apply(
        lambda x: "🚨 Violence" if x else "✅ Safe"
    )

    df["confidence"] = df["confidence"].apply(
        lambda x: f"{x:.2f}"
    )

    display_columns = [
        "filename",
        "type",
        "Status",
        "classification"
    ]

    if show_confidence:
        display_columns.append("confidence")

    st.dataframe(
        df[display_columns],
        use_container_width=True,
        hide_index=True
    )

    if st.button("📥 Export Results"):

        csv = df.to_csv(index=False)

        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name=(
                f"harassment_results_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            ),
            mime="text/csv"
        )

    if len(df) > 0:

        fig = px.pie(
            df,
            names="Status",
            title="Content Analysis Overview",
            color_discrete_sequence=[
                "#2ed573",
                "#ff4757"
            ]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =====================================================
# ABOUT SYSTEM INFORMATION
# =====================================================

def display_info_panel():

    st.markdown("""
    <div class="custom-card">

        <h3>ℹ️ About This System</h3>

        <p>
        This AI-powered system uses advanced machine
        learning algorithms to detect harassment and
        violence in multimedia content.
        </p>

        <h4>🎯 Supported Formats</h4>

        <ul>
            <li><strong>Video:</strong>
            MP4, MOV, AVI, MKV</li>

            <li><strong>Audio:</strong>
            WAV, MP3, M4A</li>
        </ul>

        <h4>🔬 Analysis Methods</h4>

        <ul>
            <li><strong>Video:</strong> MobileNetV2 + BiLSTM</li>

            <li><strong>Audio:</strong>
            MFCC Features + CNN</li>
        </ul>

    </div>
    """, unsafe_allow_html=True)


# =====================================================
# SINGLE FILE PROCESSING
# =====================================================

def process_single_file(
    uploaded_file,
    show_confidence=True,
    show_explanation=True
):

    file_path = None

    try:

        safe_name = os.path.basename(uploaded_file.name)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=Path(safe_name).suffix
        ) as temp_file:

            temp_file.write(uploaded_file.getbuffer())

            file_path = temp_file.name

        ext = os.path.splitext(safe_name)[1].lower()

        if ext in [".mp4", ".mov", ".mkv", ".avi"]:
            st.video(file_path)

        elif ext in [".wav", ".mp3", ".m4a"]:
            st.audio(file_path)

        with st.spinner("🔍 Analyzing file..."):

            result = detect_harassment(
                file_path,
                safe_name
            )

        display_single_result(
            result,
            show_confidence,
            show_explanation
        )

    except Exception as e:

        st.error(f"Analysis failed: {str(e)}")

    finally:

        if file_path and os.path.exists(file_path):
            os.unlink(file_path)


# =====================================================
# MULTIPLE FILE PROCESSING
# =====================================================

def process_multiple_files(
    uploaded_files,
    show_confidence=True,
    show_explanation=True
):

    st.markdown(
        f"#### 📊 Analyzing {len(uploaded_files)} files"
    )

    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []

    saved_files, temp_dir = save_uploaded_files(
        uploaded_files
    )

    try:

        for i, (file_path, filename) in enumerate(saved_files):

            status_text.text(
                f"Processing: {filename}"
            )

            try:

                result = detect_harassment(
                    file_path,
                    filename
                )

            except Exception as e:

                result = {
                    "filename": filename,
                    "type": "Error",
                    "is_harassment": False,
                    "classification": "Analysis Failed",
                    "confidence": 0.0,
                    "explanation": str(e)
                }

            results.append(result)

            progress_bar.progress(
                (i + 1) / len(saved_files)
            )

            time.sleep(0.1)

        status_text.text("✅ Analysis Complete!")

        display_batch_results(
            results,
            show_confidence,
            show_explanation
        )

    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# =====================================================
# ZIP FOLDER PROCESSING
# =====================================================

def process_folder_upload(
    uploaded_zip,
    show_confidence=True,
    show_explanation=True
):

    temp_dir = None

    try:

        extracted_files, temp_dir = extract_files_from_zip(
            uploaded_zip
        )

        if not extracted_files:

            st.warning(
                "No supported media files found in the ZIP archive."
            )

            return

        st.markdown(
            f"#### 📊 Found {len(extracted_files)} media files"
        )

        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []

        for i, (file_path, filename) in enumerate(extracted_files):

            status_text.text(
                f"Processing: {filename}"
            )

            try:

                result = detect_harassment(
                    file_path,
                    filename
                )

            except Exception as e:

                result = {
                    "filename": filename,
                    "type": "Error",
                    "is_harassment": False,
                    "classification": "Analysis Failed",
                    "confidence": 0.0,
                    "explanation": str(e)
                }

            results.append(result)

            progress_bar.progress(
                (i + 1) / len(extracted_files)
            )

            time.sleep(0.1)

        status_text.text("✅ Analysis Complete!")

        display_batch_results(
            results,
            show_confidence,
            show_explanation
        )

    except Exception as e:

        st.error(
            f"Error processing ZIP file: {str(e)}"
        )

    finally:

        if temp_dir:
            shutil.rmtree(
                temp_dir,
                ignore_errors=True
            )


# =====================================================
# MAIN STREAMLIT APPLICATION
# =====================================================

def main():

    st.markdown("""
    <div class="main-header">

        <h1>🛡️ AI-Powered Harassment Detection System</h1>

        <p>
        Advanced machine learning system for detecting
        violence and harassment in multimedia content
        </p>

    </div>
    """, unsafe_allow_html=True)

    # ---------------- SIDEBAR ----------------

    with st.sidebar:

        st.markdown("### 📋 Analysis Options")

        analysis_mode = st.radio(
            "Choose Analysis Mode:",
            [
                "Single File Upload",
                "Multiple Files Upload",
                "Folder Upload (ZIP)"
            ],
            key="analysis_mode"
        )

        st.markdown("---")

        st.markdown("### ⚙️ Settings")

        show_confidence = st.checkbox(
            "Show Confidence Scores",
            value=True
        )

        show_explanation = st.checkbox(
            "Show Detailed Explanations",
            value=True
        )

        if st.button("🔄 Reset Analysis"):
            st.rerun()

    # ---------------- MAIN CONTENT ----------------

    st.markdown(
        '<div class="custom-card">',
        unsafe_allow_html=True
    )

    # SINGLE FILE UPLOAD

    if analysis_mode == "Single File Upload":

        st.markdown("### 📁 Single File Analysis")

        uploaded_file = st.file_uploader(
            "Choose a file",
            type=[
                "mp4", "wav", "mp3",
                "mov", "mkv", "avi", "m4a"
            ],
            help="Upload an audio or video file for harassment detection"
        )

        if uploaded_file:

            process_single_file(
                uploaded_file,
                show_confidence,
                show_explanation
            )

    # MULTIPLE FILE UPLOAD

    elif analysis_mode == "Multiple Files Upload":

        st.markdown("### 📁 Multiple Files Analysis")

        uploaded_files = st.file_uploader(
            "Choose multiple files",
            type=[
                "mp4", "wav", "mp3",
                "mov", "mkv", "avi", "m4a"
            ],
            accept_multiple_files=True,
            help="Upload multiple audio or video files for batch analysis"
        )

        if uploaded_files:

            process_multiple_files(
                uploaded_files,
                show_confidence,
                show_explanation
            )

    # ZIP FOLDER UPLOAD

    else:

        st.markdown("### 📁 Folder Analysis (ZIP Upload)")

        uploaded_zip = st.file_uploader(
            "Upload a ZIP folder containing media files",
            type=["zip"],
            help="Upload a ZIP file containing your video/audio files"
        )

        if uploaded_zip:

            process_folder_upload(
                uploaded_zip,
                show_confidence,
                show_explanation
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # ---------------- SIDEBAR INFORMATION ----------------

    with st.sidebar:
        display_info_panel()


# =====================================================
# APPLICATION ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
