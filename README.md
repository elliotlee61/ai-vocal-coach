# AI Vocal Coach

This project is a proposal-stage AI vocal coach demo for Musical AI.

The goal is to analyze a short singing clip, detect a vocal quality, assess basic pitch behavior, and give goal-aware coaching feedback. The system is not meant to say that a technique is automatically good or bad. Instead, it compares the detected vocal quality and pitch behavior to the user's intended target style.

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
- pitch analysis
- pitch stability rating
- goal-aware coaching feedback

## Pitch Analysis

The demo estimates the singer's pitch contour from the audio file. It does not yet measure full pitch accuracy against a reference melody. Instead, it measures pitch behavior inside the clip.

The pitch analysis includes:

- estimated average pitch
- minimum and maximum pitch
- pitch range
- pitch variation percentage
- pitch stability rating: high, medium, or low

This helps the coach give more grounded feedback. For example, if the user chooses a clear or straight-tone goal, the system expects the pitch to be relatively stable. If the user chooses controlled vibrato, some pitch movement is expected, but the movement should still sound intentional and controlled.

## Example Demo Output

Example run:

```text
Enter path to a singing audio file: data/breathy/m9_scales_breathy_a.wav

What is your target singing goal?
You can type one of these options:
- clear tone
- breathy / soft tone
- controlled vibrato
- straight tone
- match reference style
- not sure

Type your target style: clear
