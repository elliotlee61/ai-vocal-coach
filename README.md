# AI Vocal Coach

AI Vocal Coach is a final project for Musical AI. It is a lightweight web-based vocal coaching system that lets a user compare their own singing attempt against a reference vocal clip.

The purpose of the project is to give singers feedback that is more useful than a simple karaoke score. Instead of only judging whether a singer matched the exact pitch, the system analyzes both **vocal quality** and **pitch behavior**, then gives feedback based on what the user is trying to match.

The system is not designed to say that a technique is automatically good or bad. A breathy tone, straight tone, or vibrato can all be valid depending on the musical goal. The coach compares the user's attempt to a reference clip or selected target style and gives feedback based on that context.

---

## Current Product

The current version is a Streamlit web app.

The user provides:

1. a reference vocal audio file
2. their own singing attempt, either by uploading an audio file or recording directly in the app
3. a target goal, such as matching the reference style, clear tone, breathy tone, straight tone, or controlled vibrato

The system outputs:

* detected vocal quality for the reference
* detected vocal quality for the user's attempt
* model confidence scores
* pitch analysis for both recordings
* pitch similarity score
* vocal quality match score
* overall match score
* pitch contour comparison plot
* practice priority
* goal-aware coaching feedback

---

## Main Features

### Reference-Based Vocal Coaching

The user uploads a reference clip and their own attempt. The system compares the two recordings and gives feedback on how closely the user's attempt matches the reference.

### Upload or Record Attempt

The user can either upload their attempt as an audio file or record directly inside the web app.

### Vocal Quality Detection

The system uses a trained machine learning classifier to detect vocal quality.

The current supported labels are:

* breathy
* straight
* vibrato

### Pitch Behavior Analysis

The app estimates the pitch contour of both the reference and user attempt.

The pitch analysis includes:

* estimated average pitch
* pitch range
* pitch variation
* pitch movement/stability rating
* pitch contour comparison plot

### Scoring

The app gives three main scores:

* pitch similarity
* vocal quality match
* overall match

### Practice Priority

The coach gives a short practice priority based on the comparison.

Example:

```text
Good match overall. Focus on consistency and control.
```

or:

```text
Priority: first slow down the phrase and focus on matching the reference pitch and vocal quality separately.
```

### Goal-Aware Feedback

The feedback depends on the user's goal. For example, a breathy tone is not automatically treated as bad. If the reference or target style is breathy, the coach treats breathiness as a valid stylistic goal. If the target is clear tone, the coach suggests ways to make the sound more connected.

---

## How the System Works

The system follows this pipeline:

```text
Reference audio
        ↓
Vocal quality detection + pitch analysis

User attempt audio
        ↓
Vocal quality detection + pitch analysis

Comparison
        ↓
Pitch similarity score
Vocal quality match score
Overall match score
Pitch contour visualization
Practice priority
Goal-aware coaching feedback
```

---

## Model

The vocal quality classifier was trained on a local VocalSet subset organized into three labels:

* breathy
* straight
* vibrato

Dataset subset used:

* 200 breathy clips
* 200 straight clips
* 199 vibrato clips
* 599 total clips

The best-performing model was:

```text
SVM with RBF kernel
```

Training result:

```text
Test clips: 150
Best accuracy: 0.893
```

Class precision:

```text
breathy: 0.87
straight: 0.85
vibrato: 0.96
```

The trained model is saved at:

```text
models/vocal_technique_model.pkl
```

This allows the web app to run without retraining the model every time.

---

## Audio Features

The model uses extracted audio features rather than raw audio directly.

Features include:

* MFCCs
* MFCC deltas
* spectral centroid
* spectral bandwidth
* spectral rolloff
* spectral flatness
* RMS energy
* zero-crossing rate
* pitch-based features

Pitch-based features help the model distinguish between straight tone and vibrato because vibrato involves pitch movement over time, while straight tone is generally more stable.

---

## Pitch Analysis

The app estimates pitch using a pitch-tracking method and creates a pitch contour over time.

The pitch contour is used to compare the reference and user attempt. The system calculates differences in average pitch and pitch variation, then converts those differences into a pitch similarity score.

Important note: this version does not yet perform full note-by-note pitch accuracy against a melody. It compares pitch behavior and pitch contour similarity. A future version could add reference alignment and note-level pitch scoring.

---

## Files

```text
app.py                              Streamlit web interface
demo.py                             Terminal version and audio analysis functions
feedback.py                         Goal-aware coaching feedback
train_model.py                      Model training and evaluation script
requirements.txt                    Python dependencies
models/vocal_technique_model.pkl    Trained vocal quality model
screenshots/                        App and terminal screenshots
```

---

## How to Run the Web App

### 1. Clone or open the repository

If using GitHub Codespaces, open the repository in a Codespace.

If running locally:

```bash
git clone https://github.com/elliotlee61/ai-vocal-coach.git
cd ai-vocal-coach
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Make sure the trained model exists

The app expects the model file here:

```text
models/vocal_technique_model.pkl
```

If the model file is already included, the app can run directly.

If the model file is missing, retrain the model:

```bash
python train_model.py
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

### 5. Open the app

If running locally, Streamlit should open in the browser automatically.

If using GitHub Codespaces:

1. run `streamlit run app.py`
2. open the **Ports** tab
3. find the forwarded Streamlit port, usually `8501`
4. click the browser/globe icon to open the app

---

## How to Use the App

1. Upload a reference vocal clip.
2. Upload your own attempt or select **Record now** to record directly in the app.
3. Choose what you are trying to match:

   * match reference style
   * clear tone
   * breathy tone
   * controlled vibrato
   * straight tone
   * not sure
4. Click **Analyze**.
5. Review the results:

   * detected vocal quality
   * confidence score
   * pitch analysis
   * pitch similarity score
   * vocal quality match score
   * overall match score
   * pitch contour plot
   * practice priority
   * coaching feedback

---

## Example Output

```text
Reference detected quality: breathy
User detected quality: breathy

Pitch similarity: 97/100 (strong)
Vocal quality match: 100/100
Overall match: 98/100

Practice priority:
Good match overall. Focus on consistency and control.

Feedback:
The system detects a breathy tone, which usually means the sound has more air mixed into it.

This matches your breathy or soft-tone goal. Next, focus on keeping the breathiness controlled so the pitch and words stay clear.
```

---

## Terminal Version

The project also includes a terminal version of the coach.

Run:

```bash
python demo.py
```

Then enter an audio path when prompted.

Example:

```text
data/breathy/m9_scales_breathy_a.wav
```

Then enter a target style, such as:

```text
clear tone
```

or:

```text
breathy
```

or:

```text
controlled vibrato
```

The terminal version outputs:

* detected vocal quality
* confidence score
* pitch analysis
* goal-aware coaching feedback

---

## Training the Model

To retrain the model, run:

```bash
python train_model.py
```

The training script:

1. loads VocalSet clips from the local `data/` folders
2. extracts audio and pitch features
3. trains multiple classifiers
4. evaluates the models
5. saves the best model to `models/vocal_technique_model.pkl`

Expected local data folder structure:

```text
data/
  breathy/
  straight/
  vibrato/
```

The VocalSet audio files are not included because they are large.

---

## Current Limitations

This is a final class product, but it is still a lightweight academic prototype.

Current limitations:

* only three vocal quality labels are supported
* pitch comparison is based on pitch behavior, not full note-by-note melody accuracy
* the system does not yet perform detailed timing alignment between reference and attempt
* the model has not yet been evaluated with held-out singers
* the feedback is rule-based after the model prediction
* noisy recordings or recordings with background instruments may reduce accuracy

---

## Future Improvements

Possible future improvements include:

* held-out singer evaluation
* more vocal quality labels
* better reference alignment
* note-by-note pitch accuracy
* timing comparison
* improved pitch tracking
* more detailed practice plans
* a more polished user interface

The current version focuses on a complete minimum viable product:

```text
reference upload
+ attempt upload or recording
+ vocal quality detection
+ pitch behavior comparison
+ scoring
+ visualization
+ coaching feedback
```
