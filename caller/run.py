"""Single entry point: place scenario calls one after another, then transcribe and analyze them.

Usage:
  python -m caller.run all
  python -m caller.run new-booking refill-controlled
  python -m caller.run all --no-analyze        # skip the Claude review step
  python -m caller.run new-booking --no-worker # use a worker you started yourself
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from livekit import api

from . import analyze, transcribe
from .config import AGENT_NAME, CALLS_DIR, ROOT, checked_target, require
from .persona import load_scenarios

GAP_BETWEEN_CALLS_S = 20
CALL_TIMEOUT_S = 8 * 60  # ringing + the 4.5 minute in-call limit + slack


def start_worker() -> subprocess.Popen[str]:
    """Start `python -m caller.agent start` and block until it has registered with LiveKit."""
    log_path = CALLS_DIR / "_worker.log"
    CALLS_DIR.mkdir(exist_ok=True)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    proc = subprocess.Popen(
        [sys.executable, "-m", "caller.agent", "start"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=flags,
    )
    registered = threading.Event()

    def pump() -> None:
        assert proc.stdout is not None
        with log_path.open("a", encoding="utf-8") as log:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                if "registered worker" in line:
                    registered.set()
                if "ERROR" in line or "Traceback" in line:
                    print(f"[worker] {line.rstrip()}")

    threading.Thread(target=pump, daemon=True).start()
    print("Starting worker...")
    if not registered.wait(timeout=90):
        stop_worker(proc)
        raise SystemExit(f"Worker did not register within 90s, see {log_path}")
    print("Worker registered.")
    return proc


def stop_worker(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        proc.send_signal(signal.CTRL_BREAK_EVENT)  # graceful: lets the worker close its job processes
    else:
        proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def next_call_id(scenario_id: str) -> str:
    numbers = [int(m.group(1)) for d in CALLS_DIR.glob("*") if (m := re.match(r"^(\d+)-", d.name))]
    return f"{max(numbers, default=0) + 1:02d}-{scenario_id}"


async def place_call(scenario_id: str, call_id: str) -> None:
    room = f"call-{call_id}-{uuid.uuid4().hex[:6]}"
    lkapi = api.LiveKitAPI()
    try:
        await lkapi.room.create_room(api.CreateRoomRequest(name=room, empty_timeout=60))
        await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room,
                metadata=json.dumps({"scenario": scenario_id, "call_id": call_id}),
            )
        )
    finally:
        await lkapi.aclose()


def wait_for_call(call_dir: Path) -> dict | None:
    meta_path = call_dir / "meta.json"
    deadline = time.time() + CALL_TIMEOUT_S
    while time.time() < deadline:
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        time.sleep(2)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Place patient-simulator calls to TARGET_NUMBER.")
    parser.add_argument("scenarios", nargs="+", help='scenario ids from scenarios.yaml, or "all"')
    parser.add_argument("--no-analyze", action="store_true", help="skip the Claude review step")
    parser.add_argument("--no-worker", action="store_true", help="don't start a worker subprocess")
    args = parser.parse_args()

    for name in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET", "LIVEKIT_SIP_TRUNK_ID"):
        require(name)
    try:
        target = checked_target()
    except RuntimeError as e:
        raise SystemExit(f"{e} (see TARGET_NUMBER / ALLOWED_TARGETS in .env.example)") from None
    print(f"Target: {target}")

    scenarios = load_scenarios()
    ids = list(scenarios) if args.scenarios == ["all"] else args.scenarios
    unknown = [s for s in ids if s not in scenarios]
    if unknown:
        raise SystemExit(f"Unknown scenario(s): {', '.join(unknown)}. Known: {', '.join(scenarios)}")

    worker = None if args.no_worker else start_worker()
    try:
        for i, scenario_id in enumerate(ids):
            if i:
                time.sleep(GAP_BETWEEN_CALLS_S)
            call_id = next_call_id(scenario_id)
            call_dir = CALLS_DIR / call_id
            print(f"\n=== {call_id}: dialing ===")
            asyncio.run(place_call(scenario_id, call_id))

            meta = wait_for_call(call_dir)
            if meta is None:
                print(f"{call_id}: timed out waiting for the call to finish")
                continue
            print(f"{call_id}: ended ({meta['end_reason']}), {meta.get('duration_s', 0)}s")
            if not (call_dir / "recording.ogg").exists():
                continue

            print(f"{call_id}: transcript -> {transcribe.transcribe(call_dir)}")
            if not args.no_analyze:
                print(f"{call_id}: analysis   -> {analyze.analyze_call(call_dir)}")
    finally:
        if worker:
            stop_worker(worker)

    if not args.no_analyze:
        print(f"\nSummary -> {analyze.summarize()}")


if __name__ == "__main__":
    main()
