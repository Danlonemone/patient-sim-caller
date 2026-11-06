"""Project paths, environment loading and the call-safety check shared by every module."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Keep downloaded model weights (turn detector) inside the project rather than the user profile.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

# The number to call, plus an explicit allowlist it must appear on. Two settings on purpose: a
# typo or a stray env var can't send the bot to a stranger's phone. Only call agents you own or
# are authorised to test.
TARGET_NUMBER = os.getenv("TARGET_NUMBER", "").strip()
ALLOWED_TARGETS = {n.strip() for n in os.getenv("ALLOWED_TARGETS", "").split(",") if n.strip()}
E164 = re.compile(r"^\+[1-9]\d{6,14}$")

AGENT_NAME = "patient-caller"
CALLS_DIR = ROOT / "calls"
SCENARIOS_FILE = ROOT / "caller" / "scenarios.yaml"

# Proper nouns the speech-to-text keeps mishearing (practice and doctor names), from the
# `keyterms` list in scenarios.yaml. Only add a term after confirming its spelling by listening
# to a recording; a wrong hint turns a transcription error into a fake agent bug.
STT_KEYTERMS: list[str] = list(
    (yaml.safe_load(SCENARIOS_FILE.read_text(encoding="utf-8")) or {}).get("keyterms") or []
)

PATIENT_LLM_MODEL = os.getenv("PATIENT_LLM_MODEL", "claude-haiku-5-5")
ANALYSIS_MODEL = os.getenv("ANALYSIS_MODEL", "claude-opus-5-5")

# Text-to-speech provider: Deepgram Aura-2 (default; shares the speech-to-text key) or
# Cartesia Sonic-3 (the most natural-sounding option, needs its own key).
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "deepgram")

# Voices per persona type, chosen for natural conversational American voices.
VOICES = {
    "cartesia": {
        "female": "2747b6cf-fa34-460c-97db-267566918881",  # Allie - Natural Conversationalist
        "female_alt": "e8e5fffb-252c-436d-b842-8879b84445b6",  # Cathy - Coworker
        "male": "a5136bf9-224c-4d76-b823-52bd5efcffcc",  # Jameson - Easygoing Support
        "male_alt": "bbee10a8-4f08-4c5c-8282-e69299115055",  # Ben - Helpful Man (middle-aged)
        "older_female": "8634bd27-0acf-4056-b014-4fea0385ed9e",  # Suzanne - Laidback Aunt
        "older_male": "0ad65e7f-006c-47cf-bd31-52279d487913",  # Rupert - Caring Dad
    },
    "deepgram": {
        "female": "aura-2-andromeda-en",  # casual, expressive
        "female_alt": "aura-2-luna-en",  # young adult, friendly
        "male": "aura-2-orion-en",  # approachable, calm
        "male_alt": "aura-2-arcas-en",  # natural, smooth
        "older_female": "aura-2-athena-en",  # mature, calm
        "older_male": "aura-2-atlas-en",  # mature, approachable
    },
}


def require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable {name} (see .env.example)")
    return value


def checked_target() -> str:
    """Return TARGET_NUMBER, or raise if it isn't a valid E.164 number listed in ALLOWED_TARGETS."""
    number = TARGET_NUMBER
    if not E164.match(number):
        raise RuntimeError(f"TARGET_NUMBER {number!r} is missing or not in E.164 form (+15555550100)")
    if number not in ALLOWED_TARGETS:
        raise RuntimeError(f"refusing to dial {number}: it is not listed in ALLOWED_TARGETS")
    return number
