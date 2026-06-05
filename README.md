# AI Vocal Coach

AI Vocal Coach is a final project for Musical AI. It is a web-based vocal coaching system that compares a singer’s performance to a reference vocal recording.

The user uploads a reference vocal clip and either uploads or records their own attempt. The system analyzes both recordings, detects vocal quality, compares pitch behavior, calculates similarity scores, displays a pitch contour graph, and gives reference-based coaching feedback.

The purpose of the project is not to judge a vocal technique as automatically good or bad. Breathy tone, straight tone, and vibrato can all be valid depending on the performance. Instead, the system evaluates how closely the user’s attempt matches the reference.

---

## Main Features

* Upload a reference vocal recording
* Upload or record a singing attempt
* Detect vocal quality in both recordings
* Display model confidence
* Analyze pitch behavior
* Compare reference and user pitch contours
* Calculate pitch similarity
* Calculate vocal quality match
* Calculate an overall match score
* Display a practice priority
* Generate reference-matching coaching feedback

---

## Supported Vocal Qualities

The current classifier detects three vocal qualities:

* breathy
* straight
* vibrato

The system analyzes both the reference and the user attempt using the same classifier.

Example:

```text
Reference vocal quality: breathy
User vocal quality: straight
```

The coach then explains how the user’s vocal quality differs from the reference and suggests an adjustment.

---

## How the System Works

```text
Reference audio
        ↓
Vocal quality detection
Pitch analysis
        ↓

User attempt
        ↓
Vocal quality detection
Pitch analysis
        ↓

Reference comparison
        ↓
Pitch similarity score
Vocal quality match score
Overall match score
Pitch contour visualization
Practice priority
Coaching feedback
```

---

## Machine Learning Model

The vocal quality classifier was trained using a subset of the VocalSet dataset.

Training data:

* 200 breathy clips
* 200 straight clips
* 199 vibrato clips
* 599 total clips

The training script compared three machine learning classifiers:

* Random Forest
* Extra Trees
* Support Vector Machine with an RBF kernel

The best-performing model was:

```text
SVM with RBF kernel
```

Evaluation result:

```text
Test clips: 150
Accuracy: 89.3%
```

Class precision:

```text
Breathy: 0.87
Straight: 0.85
Vibrato: 0.96
```

The trained model is stored at:

```text
models/vocal_technique_model.pkl
```

This allows the web app to run without retraining the model every time.

---

## Audio Features

The model does not classify raw audio directly. Each recording is converted into numerical audio features.

The extracted features include:

* MFCCs
* MFCC delta features
* spectral centroid
* spectral bandwidth
* spectral rolloff
* spectral flatness
* RMS energy
* zero-crossing rate
* pitch-based features

MFCC and spectral features help describe vocal tone and timbre.

Pitch-based features help distinguish between straight tone and vibrato because straight tone is generally more stable, while vibrato includes repeated pitch movement.

---

## Pitch Analysis

The system estimates the fundamental frequency, or F0, across time.

This creates a pitch contour for both the reference and the user attempt.

The pitch analysis includes:

* estimated average pitch
* minimum pitch
* maximum pitch
* pitch range
* pitch variation
* pitch movement or stability
* pitch contour graph

Extreme pitch-tracking outliers are filtered before the pitch values are displayed and scored.

The app compares the reference and user attempt using:

* average pitch difference
* pitch variation difference
* pitch contour behavior

Important note: the current system performs pitch behavior comparison. It does not yet perform full note-by-note melody alignment or identify every individual sharp or flat note.

---

## Scoring

The app generates three main scores.

### Pitch Similarity

Pitch similarity compares the pitch behavior of the reference and the user attempt.

The score is based on:

* average pitch difference
* pitch variation difference

The score labels are:

```text
80–100: Strong
60–79: Moderate
40–59: Partial
0–39: Weak
```

### Vocal Quality Match

The vocal quality match compares the detected vocal quality labels.

Current scoring:

```text
Same vocal quality: 100
Straight and vibrato mismatch: 60
Other vocal quality mismatch: 40
```

### Overall Match

The overall score combines pitch similarity and vocal quality match.

```text
Overall match =
55% pitch similarity
+
45% vocal quality match
```

Pitch is weighted slightly more because the user is attempting to match a reference performance.

---

## Practice Priority

The system gives the user a short practice priority based on the comparison.

Examples:

```text
Good match overall. Focus on consistency and control.
```

```text
Priority: improve pitch similarity while keeping the same vocal style.
```

```text
Priority: match the reference vocal quality.
```

```text
Priority: slow down the phrase and practice pitch and vocal quality separately.
```

---

## Coaching Feedback

The coaching feedback is based on:

* the vocal quality detected in the reference
* the vocal quality detected in the user attempt
* pitch behavior from the user attempt
* how closely the two recordings match

Example:

```text
The reference was detected as breathy, while the user attempt was detected as straight.

Your attempt sounds straighter than the reference. Try allowing slightly more air into the tone while keeping the pitch controlled.
```

If the detected vocal qualities match, the coach suggests focusing on consistency and control rather than changing the style.

---

## User Interface

The web interface was built with Streamlit.

Streamlit is a Python library used to create interactive web applications. It provides the interface for:

* audio uploading
* microphone recording
* audio playback
* buttons
* scores
* graphs
* feedback text

Streamlit is the interface layer. The trained SVM model and audio-analysis functions perform the actual analysis.

---

## Project Files

```text
app.py
Streamlit web interface

demo.py
Terminal version, feature extraction, and pitch analysis

feedback.py
Reference-matching coaching feedback

train_model.py
Model training and evaluation

requirements.txt
Python dependencies

models/vocal_technique_model.pkl
Trained vocal quality classifier

screenshots/
Images of the working application
```

---

## How to Run the Web App

### 1. Clone the repository

```bash
git clone https://github.com/elliotlee61/ai-vocal-coach.git
cd ai-vocal-coach
```

You can also open the repository directly in GitHub Codespaces.

### 2. Install the required Python packages

```bash
pip install -r requirements.txt
```

### 3. Confirm that the trained model exists

The application expects the trained model at:

```text
models/vocal_technique_model.pkl
```

If the trained model is included in the repository, no retraining is required.

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

You can also use:

```bash
python -m streamlit run app.py
```

### 5. Open the application

When running locally, Streamlit should open automatically in a browser.

When using GitHub Codespaces:

1. Run `streamlit run app.py`
2. Open the **Ports** tab
3. Find port `8501`
4. Click the browser or globe icon
5. Open the forwarded Streamlit page

---

## How to Use the Application

1. Upload a reference vocal recording.
2. Choose how to provide your attempt:

   * upload an audio file
   * record directly in the app
3. Provide the user attempt.
4. Click **Analyze**.
5. Review:

   * reference vocal quality
   * user vocal quality
   * confidence scores
   * pitch analysis
   * pitch similarity
   * vocal quality match
   * overall match
   * pitch contour graph
   * practice priority
   * coaching feedback

Recommended audio:

* short vocal-only recordings
* WAV format when possible
* minimal background noise
* little or no instrumental accompaniment
* similar phrases for the reference and user attempt

---

## Retraining the Model

Retraining is not required to use the web application if the trained model file is included.

To retrain the classifier:

```bash
python train_model.py
```

Expected local dataset structure:

```text
data/
    breathy/
    straight/
    vibrato/
```

The training script:

1. loads the labeled audio clips
2. extracts audio and pitch features
3. creates a training and test split
4. trains Random Forest, Extra Trees, and SVM RBF models
5. evaluates each model
6. selects the best model
7. saves it to `models/vocal_technique_model.pkl`

The VocalSet audio files are not included in this repository because of their size.

---

## Current Limitations

The current final class product has several limitations:

* only three vocal quality labels are supported
* pitch comparison is not full note-by-note melody accuracy
* reference and user timing are not automatically aligned
* coaching text is rule-based after model prediction
* the classifier has not yet been evaluated using completely held-out singers
* noisy audio and background music can reduce reliability
* the overall score is a project-designed similarity score, not a standardized professional singing score

---

## Future Improvements

Possible future improvements include:

* note-by-note pitch accuracy
* automatic time alignment
* timing and rhythm comparison
* more vocal quality labels
* held-out singer evaluation
* improved vibrato rate and depth analysis
* more detailed exercises
* progress tracking
* improved support for real-world recordings
* deployment as a permanent public web application

---

## Final Product Summary

AI Vocal Coach is a complete minimum viable vocal-coaching product for the class.

It includes:

```text
reference audio upload
+ user upload or recording
+ trained vocal quality classifier
+ pitch analysis
+ reference comparison
+ similarity scores
+ pitch contour visualization
+ practice priority
+ coaching feedback
```

The final system demonstrates how a trained audio classifier can be combined with pitch analysis and a user interface to create a practical reference-matching AI vocal coach.
