# AI Vocal Coach

This project is a proposal-stage AI vocal coach demo for Musical AI.

The goal is to analyze a short singing clip, detect a vocal quality, and give goal-aware coaching feedback. The system is not meant to say that a technique is automatically good or bad. Instead, it compares the detected vocal quality to the user's intended target style.

## Current Demo

The current demo uses a trained vocal technique classifier with three labels:

- breathy
- straight
- vibrato

The user provides:

1. a path to a singing audio file
2. a target singing goal, such as clear tone, breathy tone, or controlled vibrato

The system outputs:

- detected vocal quality
- confidence score
- goal-aware coaching feedback

## Example

```text
Input audio: data/breathy/m9_scales_breathy_a.wav
Target style: clear tone

Detected vocal quality: breathy
Confidence: 0.82

Feedback:
The system detects a breathy tone. Your tone sounds breathy compared to your clear-tone goal. Try using steadier airflow and keeping the sound more connected.
