"""scenarios.yaml is hand-edited; catch mistakes before they surface as a failed call."""

from datetime import date

import pytest
import yaml

from caller.config import SCENARIOS_FILE, VOICES
from caller.persona import _calendar, build_instructions, load_scenarios

RAW = yaml.safe_load(SCENARIOS_FILE.read_text(encoding="utf-8"))
SCENARIO_IDS = [s["id"] for s in RAW["scenarios"]]


def test_scenario_ids_are_unique():
    assert len(SCENARIO_IDS) == len(set(SCENARIO_IDS))


@pytest.mark.parametrize("scenario", RAW["scenarios"], ids=SCENARIO_IDS)
def test_scenario_is_complete(scenario):
    assert scenario["persona"] in RAW["personas"], "unknown persona"
    for key in ("goal", "probe", "expected"):
        assert str(scenario.get(key, "")).strip(), f"missing {key}"
    behavior = scenario.get("behavior") or {}
    if "interrupt_on" in behavior:
        assert str(behavior.get("interrupt_line", "")).strip(), "interrupt_on needs interrupt_line"


@pytest.mark.parametrize("name", RAW["personas"])
def test_persona_has_a_voice_for_every_tts_provider(name):
    persona = RAW["personas"][name]
    for key in ("name", "dob", "phone", "insurance", "notes"):
        assert str(persona.get(key, "")).strip(), f"missing {key}"
    for provider, voices in VOICES.items():
        assert persona["voice"] in voices, f"no {provider} voice for {persona['voice']!r}"


def test_load_scenarios_builds_every_scenario():
    assert list(load_scenarios()) == SCENARIO_IDS


def test_calendar_starts_on_monday_and_covers_six_weeks():
    lines = _calendar(date(2026, 11, 5)).splitlines()  # a Thursday
    assert len(lines) == 6
    assert lines[0].startswith("Mon Nov 2, Tue Nov 3")
    assert lines[1].startswith("Mon Nov 9")


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_instructions_render_cleanly(scenario_id):
    text = build_instructions(load_scenarios()[scenario_id], today=date(2026, 11, 5))
    assert "Today is Thursday, November 5, 2026." in text
    assert "<hangup>" in text  # the patient must be told how to end the call
    assert "{" not in text and "}" not in text  # no unformatted template fields
