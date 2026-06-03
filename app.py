import os
import tempfile

import joblib
import numpy as np
import streamlit as st

from demo import extract_features, analyze_pitch
from feedback import get_feedback


MODEL_PATH = "models/vocal_technique_model.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Make sure models/vocal_technique_model.pkl exists.")
        st.stop()

    model_package = joblib.load(MODEL_PATH)

    if isinstance(model_package, dict):
        model = model_package["model"]
        model_name = model_package.get("model_name", "Unknown model")
        labels = model_package.get("labels", [])
    else:
        model = model_package
        model_name = "Legacy model"
        labels = []

    return model, model_name, labels


def save_uploaded_file(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name


def predict_audio(file_path, model):
    features = extract_features(file_path)
    prediction = model.predict(features)[0]

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))

    pitch_info = analyze_pitch(file_path)

    return prediction, confidence, pitch_info


def format_pitch_info(pitch_info):
    if not pitch_info.get("available", False):
        return {
            "Average pitch": "N/A",
            "Pitch range": "N/A",
            "Pitch variation": "N/A",
            "Pitch stability": "N/A",
        }

    return {
        "Average pitch": f"{pitch_info['average_pitch_hz']:.1f} Hz",
        "Pitch range": f"{pitch_info['min_pitch_hz']:.1f}–{pitch_info['max_pitch_hz']:.1f} Hz",
        "Pitch variation": f"{pitch_info['variation_percent']:.1f}%",
        "Pitch stability": pitch_info["stability"],
    }


def compare_pitch(reference_pitch, user_pitch):
    if not reference_pitch.get("available", False) or not user_pitch.get("available", False):
        return "Pitch comparison was not available because one of the files did not have enough stable pitch information."

    avg_diff = abs(reference_pitch["average_pitch_hz"] - user_pitch["average_pitch_hz"])
    variation_diff = abs(reference_pitch["variation_percent"] - user_pitch["variation_percent"])

    if avg_diff < 25:
        pitch_match = "strong"
    elif avg_diff < 75:
        pitch_match = "moderate"
    else:
        pitch_match = "weak"

    return (
        f"Pitch similarity: {pitch_match}. "
        f"The average pitch difference was about {avg_diff:.1f} Hz. "
        f"The pitch variation difference was about {variation_diff:.1f}%."
    )


def compare_quality(reference_label, user_label):
    if reference_label == user_label:
        return (
            f"Vocal quality match: strong. "
            f"Both the reference and user attempt were detected as {reference_label}."
        )

    return (
        f"Vocal quality match: needs work. "
        f"The reference was detected as {reference_label}, while the user attempt was detected as {user_label}."
    )


def main():
    st.set_page_config(page_title="AI Vocal Coach", page_icon="🎤", layout="wide")

    st.title("🎤 AI Vocal Coach")
    st.write(
        "Upload a reference vocal clip and your own attempt. "
        "The coach compares vocal quality and pitch behavior, then gives feedback."
    )

    model, model_name, labels = load_model()

    st.sidebar.header("Model")
    st.sidebar.write(f"Using: **{model_name}**")
    st.sidebar.write("Labels: breathy, straight, vibrato")

    st.header("1. Upload Audio")

    col1, col2 = st.columns(2)

    with col1:
        reference_file = st.file_uploader(
            "Reference audio",
            type=["wav", "mp3", "flac"],
            key="reference"
        )

    with col2:
        attempt_input_method = st.radio(
            "How do you want to provide your attempt?",
            ["Upload file", "Record now"],
            horizontal=True
        )

        if attempt_input_method == "Upload file":
            user_file = st.file_uploader(
                "Your attempt",
                type=["wav", "mp3", "flac"],
                key="user_upload"
            )
        else:
            user_file = st.audio_input(
                "Record your attempt",
                key="user_recording"
            )

    target_style = st.selectbox(
        "What are you trying to match?",
        [
            "match reference style",
            "clear tone",
            "breathy tone",
            "controlled vibrato",
            "straight tone",
            "not sure",
        ]
    )

    if st.button("Analyze"):
        if reference_file is None or user_file is None:
            st.warning("Please provide both a reference audio file and your attempt. Your attempt can be uploaded or recorded.")
            return

        reference_path = save_uploaded_file(reference_file)
        user_path = save_uploaded_file(user_file)

        with st.spinner("Analyzing audio..."):
            reference_label, reference_conf, reference_pitch = predict_audio(reference_path, model)
            user_label, user_conf, user_pitch = predict_audio(user_path, model)

        st.header("2. Results")

        ref_col, user_col = st.columns(2)

        with ref_col:
            st.subheader("Reference")
            st.audio(reference_file)
            st.metric("Detected vocal quality", reference_label)

            if reference_conf is not None:
                st.metric("Confidence", f"{reference_conf:.2f}")

            st.write("Pitch analysis")
            st.json(format_pitch_info(reference_pitch))

        with user_col:
            st.subheader("Your attempt")
            st.audio(user_file)
            st.metric("Detected vocal quality", user_label)

            if user_conf is not None:
                st.metric("Confidence", f"{user_conf:.2f}")

            st.write("Pitch analysis")
            st.json(format_pitch_info(user_pitch))

        st.header("3. Comparison")

        quality_comparison = compare_quality(reference_label, user_label)
        pitch_comparison = compare_pitch(reference_pitch, user_pitch)

        st.write(quality_comparison)
        st.write(pitch_comparison)

        st.header("4. Coaching Feedback")

        if target_style == "match reference style":
            style_goal = reference_label
        else:
            style_goal = target_style

        feedback = get_feedback(user_label, style_goal, user_pitch)

        st.write(feedback)

        st.info(
            "Note: This demo compares pitch behavior and vocal quality. "
            "It does not yet perform full note-by-note pitch accuracy against a melody."
        )


if __name__ == "__main__":
    main()