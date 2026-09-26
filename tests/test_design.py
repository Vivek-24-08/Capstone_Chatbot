from frontend.design import (
    DEFAULT_UI_SCALE,
    MAX_UI_SCALE,
    MIN_UI_SCALE,
    clamp_ui_scale,
    ensure_ui_preferences,
)


def test_zoom_is_clamped_to_readable_range():
    assert clamp_ui_scale(0) == MIN_UI_SCALE
    assert clamp_ui_scale(99) == MAX_UI_SCALE
    assert clamp_ui_scale(1.149) == 1.15


def test_display_preferences_receive_safe_defaults_without_overwriting_choices():
    state = {"ui_high_contrast": True, "ui_scale": 4}
    ensure_ui_preferences(state)
    assert state["ui_scale"] == MAX_UI_SCALE
    assert state["ui_high_contrast"] is True
    assert state["ui_reduce_motion"] is False
    assert state["ui_expand_sources"] is False
    assert DEFAULT_UI_SCALE == 1.0
