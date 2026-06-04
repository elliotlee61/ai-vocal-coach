def get_feedback(predicted_label, target_style, pitch_info=None):
    predicted_label = predicted_label.lower().strip()
    target_style = target_style.lower().strip()

    target_map = {
        "clear": "straight",
        "clear tone": "straight",
        "breathy": "breathy",
        "breathy tone": "breathy",
        "soft": "breathy",
        "soft tone": "breathy",
        "straight": "straight",
        "straight tone": "straight",
        "stable": "straight",
        "stable tone": "straight",
        "vibrato": "vibrato",
        "controlled vibrato": "vibrato",
    }

    normalized_target = target_map.get(target_style, target_style)

    detected_explanations = {
        "breathy": "The system detects a breathy tone, which usually means the sound has more air mixed into it.",
        "straight": "The system detects a mostly straight tone, which usually means the pitch and tone are relatively stable.",
        "vibrato": "The system detects vibrato, which means the pitch is moving or oscillating around the note.",
    }

    matching_feedback = {
        "breathy": (
            "This matches the reference's breathy or soft vocal quality. "
            "Next, focus on keeping the breathiness controlled while maintaining clear pitch and words."
        ),
        "straight": (
            "This matches the reference's clear or straight vocal quality. "
            "Next, focus on keeping the tone consistent across the whole phrase."
        ),
        "vibrato": (
            "This matches the reference's vibrato quality. "
            "Next, focus on keeping the vibrato rate even and avoiding excessive pitch wobble."
        ),
    }

    adjustment_feedback = {
        ("breathy", "straight"): (
            "Your attempt sounds breathier than the reference. "
            "Try using steadier airflow and keeping the sound more connected."
        ),
        ("breathy", "vibrato"): (
            "Your attempt sounds breathy, while the reference was detected as vibrato. "
            "Try first getting a clearer sustained note, then add a gentle and even vibrato."
        ),
        ("straight", "breathy"): (
            "Your attempt sounds straighter than the reference. "
            "Try allowing a little more air into the tone while keeping the pitch steady."
        ),
        ("straight", "vibrato"): (
            "Your attempt sounds mostly straight, while the reference used vibrato. "
            "Try sustaining the note first, then gently vary the pitch in a controlled, even way."
        ),
        ("vibrato", "straight"): (
            "Your attempt has more vibrato than the reference. "
            "Try holding the note more steadily and reducing extra pitch movement."
        ),
        ("vibrato", "breathy"): (
            "Your attempt has vibrato, while the reference was detected as breathier or softer. "
            "Try reducing pitch movement and focusing on a lighter, airier sound."
        ),
    }

    detected_text = detected_explanations.get(
        predicted_label,
        "The system detected a vocal quality, but the explanation for this label is still being developed."
    )

    if predicted_label == normalized_target:
        coaching_text = matching_feedback.get(
            predicted_label,
            "This appears to match your target style. Keep practicing for consistency."
        )
    else:
        coaching_text = adjustment_feedback.get(
            (predicted_label, normalized_target),
            (
                f"Your attempt was detected as '{predicted_label}', while the reference target was '{target_style}'. "
                "Try comparing your recording to your intended sound and adjust one element at a time."
            )
        )

    pitch_text = build_pitch_feedback(predicted_label, normalized_target, pitch_info)

    return detected_text + "\n\n" + coaching_text + pitch_text


def build_pitch_feedback(predicted_label, normalized_target, pitch_info):
    if not pitch_info or not pitch_info.get("available", False):
        return ""

    stability = pitch_info.get("stability", "unknown")
    variation = pitch_info.get("variation_percent", None)
    pitch_range = pitch_info.get("pitch_range_hz", None)

    pitch_lines = []

    pitch_lines.append("\n\nPitch note:")

    if variation is not None and pitch_range is not None:
        pitch_lines.append(
            f"The pitch analysis estimated {variation:.1f}% pitch variation "
            f"with a range of about {pitch_range:.1f} Hz."
        )

    if normalized_target == "straight":
        if stability == "high":
            pitch_lines.append(
                "This supports your clear or straight-tone goal because the pitch stayed fairly stable."
            )
        elif stability == "medium":
            pitch_lines.append(
                "For a clearer straight tone, try to reduce extra pitch movement and hold the note more evenly."
            )
        else:
            pitch_lines.append(
                "For a clear or straight-tone goal, the pitch is moving quite a lot. Try sustaining one note more steadily."
            )

    elif normalized_target == "vibrato":
        if stability == "high":
            pitch_lines.append(
                "For a vibrato goal, the pitch may be too steady. Try adding a gentle, controlled pitch oscillation."
            )
        elif stability == "medium":
            pitch_lines.append(
                "This may fit a controlled vibrato goal if the pitch movement sounds even and intentional."
            )
        else:
            pitch_lines.append(
                "There is noticeable pitch movement. For controlled vibrato, focus on keeping the movement even rather than random."
            )

    elif normalized_target == "breathy":
        if stability == "low":
            pitch_lines.append(
                "Even for a breathy style, the pitch should still feel controlled. Try keeping the airy tone while stabilizing the note."
            )
        else:
            pitch_lines.append(
                "The pitch stability looks usable for a breathy style. Next, focus on keeping the tone soft without losing clarity."
            )

    else:
        pitch_lines.append(
            "Use this pitch information as a guide for stability, but a reference melody would be needed for true pitch accuracy."
        )

    return "\n".join(pitch_lines)
