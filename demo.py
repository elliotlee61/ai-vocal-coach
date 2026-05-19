import os
import warnings
import librosa
import numpy as np
import joblib

from feedback import get_feedback


warnings.filterwarnings("ignore")

MODEL_PATH = "models/vocal_technique_model.pkl"
CONFIDENCE_THRESHOLD = 0.55


def safe_stats(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.zeros(6)

    return np.array([
        np.mean(values),
        np.std(values),
        np.min(values),
        np.max(values),
        np.median(values),
        np.percentile(values, 75) - np.percentile(values, 25),
    ])


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    y, _ = librosa.effects.trim(y)

    if len(y) == 0:
        return np.zeros(100).reshape(1, -1)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)

    mfcc_features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1),
        np.std(mfcc_delta, axis=1),
    ])

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rms = librosa.feature.rms(y=y)[0]

    spectral_features = np.concatenate([
        safe_stats(centroid),
        safe_stats(bandwidth),
        safe_stats(rolloff),
        safe_stats(flatness),
        safe_stats(zcr),
        safe_stats(rms),
    ])

    try:
        f0 = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )

        f0 = f0[np.isfinite(f0)]
        f0 = f0[(f0 > 50) & (f0 < 2000)]

        if len(f0) > 3:
            pitch_diff = np.diff(f0)

            pitch_features = np.array([
                np.mean(f0),
                np.std(f0),
                np.min(f0),
                np.max(f0),
                np.max(f0) - np.min(f0),
                np.mean(np.abs(pitch_diff)),
                np.std(pitch_diff),
                np.percentile(f0, 75) - np.percentile(f0, 25),
            ])
        else:
            pitch_features = np.zeros(8)

    except Exception:
        pitch_features = np.zeros(8)

    features = np.concatenate([
        mfcc_features,
        spectral_features,
        pitch_features
    ])

    return features.reshape(1, -1)


def choose_target_style():
    print("\nWhat is your target singing goal?")
    print("You can type one of these options:")
    print("- clear tone")
    print("- breathy / soft tone")
    print("- controlled vibrato")
    print("- straight tone")
    print("- match reference style")
    print("- not sure")

    target_style = input("\nType your target style: ").strip().lower()

    if target_style in ["clear", "clear tone", "clean", "clean tone"]:
        return "clear tone"

    elif target_style in ["breathy", "breathy tone", "soft", "soft tone", "breathy / soft tone"]:
        return "breathy tone"

    elif target_style in ["vibrato", "controlled vibrato"]:
        return "controlled vibrato"

    elif target_style in ["straight", "straight tone", "stable", "stable tone"]:
        return "straight tone"

    elif target_style in ["match reference", "match reference style", "reference", "not sure", "unsure"]:
        return "match reference style"

    else:
        print("\nI did not recognize that target style.")
        print("Defaulting to: match reference style")
        return "match reference style"


def main():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train_model.py first.")
        return

    model_package = joblib.load(MODEL_PATH)

    # Supports both old and new saved model formats
    if isinstance(model_package, dict):
        model = model_package["model"]
        model_name = model_package.get("model_name", "Unknown model")
        labels = model_package.get("labels", [])
    else:
        model = model_package
        model_name = "Legacy model"
        labels = []

    file_path = input("Enter path to a singing audio file: ").strip()

    if not os.path.exists(file_path):
        print("File not found.")
        return

    target_style = choose_target_style()

    features = extract_features(file_path)
    prediction = model.predict(features)[0]

    confidence = None
    probabilities = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities)

    print("\nAI Vocal Coach Demo")
    print("-------------------")
    print(f"Model used: {model_name}")
    print(f"Audio file: {file_path}")
    print(f"Target style: {target_style}")

    if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
        print(f"Detected vocal quality: uncertain / low confidence")
        print(f"Closest prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}")

        if labels and probabilities is not None:
            print("\nClass probabilities:")
            for label, prob in sorted(zip(labels, probabilities), key=lambda x: x[1], reverse=True):
                print(f"- {label}: {prob:.2f}")

        print("\nGoal-aware coach feedback:")
        print(
            "The model is not confident enough to give a strong technique label. "
            "Try using a clearer or longer recording, or compare the clip against your intended target style."
        )

    else:
        print(f"Detected vocal quality: {prediction}")

        if confidence is not None:
            print(f"Confidence: {confidence:.2f}")

        print("\nGoal-aware coach feedback:")
        print(get_feedback(prediction, target_style))


if __name__ == "__main__":
    main()