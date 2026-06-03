# AI Vocal Coach

This project is a proposal-stage AI vocal coach demo for Musical AI.

The goal is to help a singer compare their own vocal attempt to a reference vocal clip. The system analyzes both recordings, detects vocal quality, compares pitch behavior, and gives goal-aware coaching feedback.

The system is not meant to say that a vocal technique is automatically good or bad. A breathy tone, straight tone, or vibrato can all be valid depending on the singer's goal. Instead, the coach compares the detected vocal quality and pitch behavior to the reference or target style.

---

## Current Demo

The current demo is a lightweight Streamlit web app.

The user provides:

1. a reference vocal audio file
2. their own attempt, either by uploading an audio file or recording directly in the app
3. a target goal, such as matching the reference style, clear tone, breathy tone, or controlled vibrato

The system outputs:

- detected vocal quality for the reference
- detected vocal quality for the user's attempt
- model confidence scores
- pitch analysis for both recordings
- pitch similarity score
- vocal quality match score
- overall match score
- pitch contour comparison plot
- practice priority
- goal-aware coaching feedback

---

## Vocal Quality Classifier

The current model detects three vocal quality labels:

- breathy
- straight
- vibrato

The classifier was trained using a VocalSet subset organized into these three labels.

Current dataset subset:

- 200 breathy clips
- 200 straight clips
- 199 vibrato clips
- 599 total clips

The current best model is:

- SVM with RBF kernel

Training result:

- 150 test clips
- best accuracy: 0.893

Class precision:

- breathy: 0.87
- straight: 0.85
- vibrato: 0.96

This is still a prototype result. A stronger final evaluation should use held-out singers to check whether the model is learning vocal technique rather than singer-specific patterns.

---

## Pitch Analysis

The app estimates the pitch contour of both the reference audio and the user's attempt.

The pitch analysis includes:

- estimated average pitch
- pitch range
- pitch variation
- pitch movement/stability rating
- pitch contour plot

The system compares the reference and attempt using pitch behavior, such as average pitch difference and pitch variation difference.

Important note: this demo does **not** yet perform full note-by-note pitch accuracy against a melody. It compares pitch behavior and pitch contour similarity.

---

## Scoring

The app produces three main scores.

### Pitch Similarity

Compares pitch behavior between the reference and user attempt.

Example:

```text
Pitch similarity: 97/100 (strong)
