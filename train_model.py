import os
import warnings
import librosa
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")

DATA_DIR = "data"
MODEL_PATH = "models/vocal_technique_model.pkl"


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

    # Trim silence
    y, _ = librosa.effects.trim(y)

    if len(y) == 0:
        return np.zeros(100)

    # MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)

    mfcc_features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(mfcc_delta, axis=1),
        np.std(mfcc_delta, axis=1),
    ])

    # Spectral / timbre features
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

    # Pitch features
    # Vibrato and straight tone are strongly related to pitch movement.
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

    return features


def load_dataset():
    X = []
    y = []

    for label in sorted(os.listdir(DATA_DIR)):
        label_path = os.path.join(DATA_DIR, label)

        if not os.path.isdir(label_path):
            continue

        for file_name in sorted(os.listdir(label_path)):
            if file_name.lower().endswith((".wav", ".mp3", ".flac")):
                file_path = os.path.join(label_path, file_name)

                try:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

    return np.array(X), np.array(y)


def evaluate_model(name, model, X_train, X_test, y_train, y_test, labels):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\n" + "=" * 60)
    print(f"Model: {name}")
    print("=" * 60)
    print(classification_report(y_test, predictions))
    print("Confusion matrix:")
    print(labels)
    print(confusion_matrix(y_test, predictions, labels=labels))
    print(f"Overall accuracy: {accuracy:.3f}")

    return accuracy, model


def main():
    X, y = load_dataset()

    if len(X) == 0:
        print("No audio files found. Add clips to data/breathy, data/vibrato, and data/straight.")
        return

    labels = sorted(set(y))

    print(f"Loaded {len(X)} audio files.")
    print(f"Labels: {labels}")

    for label in labels:
        print(f"{label}: {sum(y == label)} clips")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            max_features="sqrt"
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=700,
            random_state=42,
            class_weight="balanced",
            max_features="sqrt"
        ),

        "SVM RBF": make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=42
            )
        ),
    }

    best_name = None
    best_accuracy = -1
    best_model = None

    for name, model in models.items():
        accuracy, trained_model = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            labels
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_name = name
            best_model = trained_model

    os.makedirs("models", exist_ok=True)

    model_package = {
        "model": best_model,
        "model_name": best_name,
        "labels": labels,
        "feature_version": "mfcc_spectral_pitch_v2"
    }

    joblib.dump(model_package, MODEL_PATH)

    print("\n" + "=" * 60)
    print(f"Best model: {best_name}")
    print(f"Best accuracy: {best_accuracy:.3f}")
    print(f"Saved model package to {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()