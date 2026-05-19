import os
import librosa
import numpy as np
import joblib

from feedback import get_feedback

MODEL_PATH = "models/vocal_technique_model.pkl"

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    y, _ = librosa.effects.trim(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

    extra_features = np.array([
        np.mean(spectral_centroid),
        np.std(spectral_centroid),
        np.mean(spectral_bandwidth),
        np.std(spectral_bandwidth),
        np.mean(zero_crossing_rate),
        np.std(zero_crossing_rate),
    ])

    features = np.concatenate([mfcc_mean, mfcc_std, extra_features])
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

    model = joblib.load(MODEL_PATH)

    file_path = input("Enter path to a singing audio file: ").strip()

    if not os.path.exists(file_path):
        print("File not found.")
        return

    target_style = choose_target_style()

    features = extract_features(file_path)
    prediction = model.predict(features)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = max(probabilities)

    print("\nAI Vocal Coach Demo")
    print("-------------------")
    print(f"Audio file: {file_path}")
    print(f"Target style: {target_style}")
    print(f"Detected vocal quality: {prediction}")

    if confidence is not None:
        print(f"Confidence: {confidence:.2f}")

    print("\nGoal-aware coach feedback:")
    print(get_feedback(prediction, target_style))

if __name__ == "__main__":
    main()
