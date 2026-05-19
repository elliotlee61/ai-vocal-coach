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

    pitch_features = extract_pitch_features(y, sr)

    features = np.concatenate([
        mfcc_features,
        spectral_features,
        pitch_features
    ])

    return features.reshape(1, -1)


def extract_pitch_features(y, sr):
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

    return pitch_features


def analyze_pitch(file_path):
    """
    Gives human-readable pitch information for the vocal coach.
    This does not measure pitch accuracy against a song yet.
    It measures pitch behavior and stability inside the uploaded clip.
    """

    y, sr = librosa.load(file_path, sr=22050)
    y, _ = librosa.effects.trim(y)

    if len(y) == 0:
        return {
            "available": False,
            "message": "No usable audio found for pitch analysis."
        }

    try:
        f0 = librosa.yin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )

        f0 = f0[np.isfinite(f0)]
        f0 = f0[(f0 > 50) & (f0 < 2000)]

        if len(f0) < 5:
            return {
                "available": False,
                "message": "Not enough stable pitch information was detected."
            }

        avg_pitch = float(np.mean(f0))
        min_pitch = float(np.min(f0))
        max_pitch = float(np.max(f0))
        pitch_range = max_pitch - min_pitch
        pitch_std = float(np.std(f0))

        # Variation as percentage of average pitch
        variation_percent = (pitch_std / avg_pitch) * 100 if avg_pitch > 0 else 0

        if variation_percent < 3:
            stability = "high"
            stability_note = "Your pitch stayed relatively steady."
        elif variation_percent < 8:
            stability = "medium"
            stability_note = "Your pitch had some movement, but it was not extremely unstable."
        else:
            stability = "low"
            stability_note = "Your pitch moved around a lot in this clip."

        return {
            "available": True,
            "average_pitch_hz": avg_pitch,
            "min_pitch_hz": min_pitch,
            "max_pitch_hz": max_pitch,
            "pitch_range_hz": pitch_range,
            "pitch_std_hz": pitch_std,
            "variation_percent": variation_percent,
            "stability": stability,
            "stability_note": stability_note
        }

    except Exception as e:
        return {
            "available": False,
            "message": f"Pitch analysis failed: {e}"
        }


def print_pitch_report(pitch_info):
    print("\nPitch analysis:")
    print("----------------")

    if not pitch_info["available"]:
        print(pitch_info["message"])
        return

    print(f"Estimated average pitch: {pitch_info['average_pitch_hz']:.1f} Hz")
    print(f"Pitch range: {pitch_info['min_pitch_hz']:.1f} Hz – {pitch_info['max_pitch_hz']:.1f} Hz")
    print(f"Pitch variation: {pitch_info['variation_percent']:.1f}%")
    print(f"Pitch stability: {pitch_info['stability']}")
    print(f"Note: {pitch_info['stability_note']}")


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

    pitch_info = analyze_pitch(file_path)

    print("\nAI Vocal Coach Demo")
    print("-------------------")
    print(f"Model used: {model_name}")
    print(f"Audio file: {file_path}")
    print(f"Target style: {target_style}")

    if confidence is not None and confidence < CONFIDENCE_THRESHOLD:
        print("Detected vocal quality: uncertain / low confidence")
        print(f"Closest prediction: {prediction}")
        print(f"Confidence: {confidence:.2f}")

        if labels and probabilities is not None:
            print("\nClass probabilities:")
            for label, prob in sorted(zip(labels, probabilities), key=lambda x: x[1], reverse=True):
                print(f"- {label}: {prob:.2f}")

        print_pitch_report(pitch_info)

        print("\nGoal-aware coach feedback:")
        print(
            "The model is not confident enough to give a strong technique label. "
            "Use the pitch analysis above as a guide, and try recording a clearer or longer phrase."
        )

    else:
        print(f"Detected vocal quality: {prediction}")

        if confidence is not None:
            print(f"Confidence: {confidence:.2f}")

        print_pitch_report(pitch_info)

        print("\nGoal-aware coach feedback:")
        print(get_feedback(prediction, target_style, pitch_info))


if __name__ == "__main__":
    main()