def get_feedback(predicted_label, target_style):
    """
    Goal-aware feedback.
    Predicted_label: model prediction, such as breathy, straight, vibrato
    target_style: user's intended goal, such as clear, breathy, straight, vibrato
    """
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
            "This matches your breathy or soft-tone goal. "
            "Next, focus on keeping the breathiness controlled so the pitch and words stay clear."
        ),
        "straight": (
            "This matches your clear or straight-tone goal. "
            "Next, focus on keeping the tone consistent across the whole phrase."
        ),
        "vibrato": (
            "This matches your vibrato goal. "
            "Next, focus on keeping the vibrato rate even and avoiding excessive pitch wobble."
        ),
    }

    adjustment_feedback = {
        ("breathy", "straight"): (
            "Your tone sounds breathy compared to your clear-tone goal. "
            "Try using steadier airflow and keeping the sound more connected."
        ),
        ("breathy", "vibrato"): (
            "Your tone sounds breathy, but your goal was controlled vibrato. "
            "Try first getting a clearer sustained note, then add a gentle and even vibrato."
        ),
        ("straight", "breathy"): (
            "Your tone sounds mostly straight, but your goal was a breathy or softer style. "
            "Try allowing a little more air into the tone while keeping the pitch steady."
        ),
        ("straight", "vibrato"): (
            "Your tone sounds mostly straight, but your goal was vibrato. "
            "Try sustaining the note first, then gently vary the pitch in a controlled, even way."
        ),
        ("vibrato", "straight"): (
            "The system detects vibrato, but your goal was a clearer straight tone. "
            "Try holding the note more steadily and reducing pitch movement."
        ),
        ("vibrato", "breathy"): (
            "The system detects vibrato, but your goal was a breathy or soft tone. "
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
                f"The detected quality was '{predicted_label}', but your target style was '{target_style}'. "
                "Try comparing your recording to your intended sound and adjust one element at a time."
            )
        )

    return detected_text + "\n\n" + coaching_text
