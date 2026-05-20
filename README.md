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

## Example Demo Output

Example run:

```text
Enter path to a singing audio file:
data/breathy/m9_scales_breathy_a.wav

What is your target singing goal?
- clear tone
- breathy / soft tone
- controlled vibrato
- straight tone
- match reference style
- not sure

Type your target style:
clear
