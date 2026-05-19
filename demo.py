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
    print("\nChoose your target singing goal:")
    print("1. Clear / straight tone")
    print("2. Breathy / soft tone")
    print("3. Controlled vibrato")
    print("4. Match reference style / not sure")

    choice = input("Enter 1, 2, 3, or 4: ").strip()

    if choice == "1":
        return "clear tone"
    elif choice == "2":
        return "breathy tone"
    elif choice == "3":
        return "controlled vibrato"
    elif choice == "4":
        return "match reference style"
    else:
        print("Invalid choice. Defaulting to match reference style.")
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
