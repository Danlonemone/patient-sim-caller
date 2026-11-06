"""Latency report from events.jsonl: how quickly the patient bot answers the clinic agent.

Usage: python -m caller.latency [calls/01-new-booking ...]   (default: every call)
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from .config import CALLS_DIR


def turn_latencies(call_dir: Path) -> dict[str, list[float]]:
    events = [json.loads(line) for line in (call_dir / "events.jsonl").open(encoding="utf-8")]
    turns = [e for e in events if e["type"] == "turn"]
    out: dict[str, list[float]] = {"response": [], "end_of_turn": [], "llm_ttft": [], "tts_ttfb": []}
    for prev, turn in zip(turns, turns[1:]):
        m, pm = turn["metrics"], prev["metrics"]
        if turn["speaker"] != "PATIENT":
            continue
        if prev["speaker"] == "AGENT" and "stopped_speaking_at" in pm and "started_speaking_at" in m:
            out["response"].append(m["started_speaking_at"] - pm["stopped_speaking_at"])
        if prev["speaker"] == "AGENT" and "end_of_turn_delay" in pm:
            out["end_of_turn"].append(pm["end_of_turn_delay"])
        if "llm_node_ttft" in m:
            out["llm_ttft"].append(m["llm_node_ttft"])
        if "tts_node_ttfb" in m:
            out["tts_ttfb"].append(m["tts_node_ttfb"])
    return out


def _fmt(values: list[float]) -> str:
    if not values:
        return "n/a"
    values = sorted(values)
    p90 = values[min(len(values) - 1, int(0.9 * len(values)))]
    return f"p50 {statistics.median(values):.2f}s  p90 {p90:.2f}s  (n={len(values)})"


def main(dirs: list[Path]) -> None:
    totals: dict[str, list[float]] = {}
    for d in dirs:
        lat = turn_latencies(d)
        print(f"{d.name}: response {_fmt(lat['response'])}")
        for key, values in lat.items():
            totals.setdefault(key, []).extend(values)
    print("\nAll calls")
    labels = {
        "response": "clinic stops -> patient starts",
        "end_of_turn": "end-of-turn detection",
        "llm_ttft": "LLM time to first token",
        "tts_ttfb": "TTS time to first byte",
    }
    for key, label in labels.items():
        print(f"  {label:32} {_fmt(totals.get(key, []))}")


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]]
    main(args or sorted(p.parent for p in CALLS_DIR.glob("*/events.jsonl")))
