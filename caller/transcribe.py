"""Turn a call's stereo recording into a timestamped, two-speaker transcript.

The recording has the clinic agent on channel 0 and our patient bot on channel 1, so Deepgram's
multichannel mode gives perfect speaker separation, and the timestamps line up exactly with the
audio file you'll listen to when checking a finding.

Usage: python -m caller.transcribe calls/01-new-booking [calls/02-...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

from .config import STT_KEYTERMS, require

SPEAKERS = {0: "AGENT", 1: "PATIENT"}
DEEPGRAM_URL = "https://api.deepgram.com/v1/listen"


def transcribe(call_dir: Path) -> Path:
    audio = (call_dir / "recording.ogg").read_bytes()
    response = httpx.post(
        DEEPGRAM_URL,
        params=[
            ("model", "nova-3"),
            ("multichannel", "true"),
            ("utterances", "true"),
            ("smart_format", "true"),
            ("punctuate", "true"),
            *[("keyterm", term) for term in STT_KEYTERMS],
        ],
        headers={
            "Authorization": f"Token {require('DEEPGRAM_API_KEY')}",
            "Content-Type": "audio/ogg",
        },
        content=audio,
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()
    (call_dir / "deepgram.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    utterances = sorted(result["results"].get("utterances", []), key=lambda u: u["start"])
    lines = [_header(call_dir), ""]
    lines += [
        f"[{_mmss(u['start'])}] {SPEAKERS.get(u['channel'], '?')}: {u['transcript']}"
        for u in utterances
        if u["transcript"].strip()
    ]
    out = call_dir / "transcript.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _header(call_dir: Path) -> str:
    meta_path = call_dir / "meta.json"
    if not meta_path.exists():
        return f"# {call_dir.name}"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    duration = meta.get("duration_s")
    length = _mmss(duration) if duration is not None else "?"
    return (
        f"# {meta['call_id']} | scenario: {meta['scenario']} | patient: {meta['patient']} | "
        f"{meta.get('answered_at', '')[:16].replace('T', ' ')} UTC | duration {length} | "
        f"ended: {meta.get('end_reason', '?')}\n"
        "# AGENT = clinic's AI receptionist, PATIENT = our caller bot. "
        "Timestamps match recording.ogg."
    )


def _mmss(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(transcribe(Path(arg)))
