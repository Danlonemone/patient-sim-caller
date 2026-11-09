"""LiveKit worker: each job dials the clinic line once and plays one patient scenario.

Pipeline: Deepgram Nova-3 (STT) -> Claude Haiku 5.5 (LLM) -> Deepgram Aura-2 or Cartesia Sonic-3
(TTS, see config.TTS_PROVIDER), with Silero VAD and LiveKit's semantic turn detector deciding when
the clinic agent has finished speaking.

Run modes:
  python -m caller.agent start     # worker used by `python -m caller.run`
  python -m caller.agent console   # talk to the patient locally with your mic (no phone call)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.protobuf.duration_pb2 import Duration
from livekit import api
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    StopResponse,
    cli,
    room_io,
)
from livekit.agents import llm
from livekit.agents.llm import ChatMessage
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.recorder_io import RecorderIO
from livekit.plugins import anthropic, cartesia, deepgram, silero
from livekit.plugins.anthropic import llm as anthropic_llm
from livekit.plugins.turn_detector.english import EnglishModel

from .config import (
    AGENT_NAME,
    CALLS_DIR,
    PATIENT_LLM_MODEL,
    STT_KEYTERMS,
    TARGET_NUMBER,
    TTS_PROVIDER,
    VOICES,
    checked_target,
    require,
)
from .persona import Scenario, build_instructions, load_scenarios
from .speech import HANGUP_MARKER, SpeechFilter

logger = logging.getLogger("patient-caller")

CLINIC_IDENTITY = "clinic"
GREETING_GRACE_S = 8.0  # the clinic normally speaks first; nudge only if it stays silent
GREETING_MAX_S = 15.0  # stop waiting for the clinic's "how can I help?" after this long
DEFAULT_MAX_MINUTES = 4.0
HANGUP_WAIT_FOR_CLINIC_S = 5.0  # let the clinic finish its closing line before we hang up

# livekit-plugins-anthropic 1.8.3 only knows that the 4.6 models reject assistant-message prefill;
# Haiku 5.5 does too, so make the plugin end every request on a user turn for it as well.
anthropic_llm._NO_PREFILL_PATTERNS = (*anthropic_llm._NO_PREFILL_PATTERNS, "claude-haiku-5-5")


class FastClaude(anthropic.LLM):
    """Anthropic LLM with thinking switched off.

    Haiku 5.5 thinks adaptively unless told otherwise, which adds first-token latency we can't
    afford mid-conversation. The plugin has no constructor option for it, but chat() forwards
    extra_kwargs straight into messages.create().
    """

    def chat(self, *, extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN, **kwargs: Any):
        extra = dict(extra_kwargs) if is_given(extra_kwargs) else {}
        extra.setdefault("thinking", {"type": "disabled"})
        return super().chat(extra_kwargs=extra, **kwargs)


class PatientAgent(Agent):
    """The patient. It ends the call by appending HANGUP_MARKER to its goodbye.

    We use a text marker rather than a function tool because sending even one tool definition
    added 0.2-0.4 s to Claude's time-to-first-token on every turn (measured), and the patient only
    needs to hang up once per call. The marker is stripped here, before TTS and the chat history.
    """

    def __init__(self, instructions: str) -> None:
        super().__init__(instructions=instructions)
        self.wants_to_hang_up = False
        self.answered_at: float | None = None  # set once the call connects
        self.greeted = False

    async def on_user_turn_completed(self, turn_ctx, new_message):  # type: ignore[override]
        # The clinic opens with a recording disclaimer, then pauses ("Thanks... for calling...")
        # before its greeting. A real caller waits for "how can I help?" instead of answering the
        # disclaimer, so hold replies until the clinic asks a question (or GREETING_MAX_S passes).
        if self.greeted or self.answered_at is None:
            return
        asked = (new_message.text_content or "").rstrip().endswith("?")
        if asked or time.time() - self.answered_at > GREETING_MAX_S:
            self.greeted = True
            return
        raise StopResponse()

    async def llm_node(self, chat_ctx, tools, model_settings):  # type: ignore[override]
        self.wants_to_hang_up = False
        speech = SpeechFilter()  # strips the hang-up marker and filler words (see speech.py)
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            text = chunk.delta.content if isinstance(chunk, llm.ChatChunk) and chunk.delta else None
            if text is None:
                yield chunk
                continue
            out = speech.feed(text)
            if speech.hung_up:
                self.wants_to_hang_up = True
            if out:
                yield out
        tail = speech.flush()
        if tail:
            yield tail


class CallLog:
    """Writes events.jsonl while the call runs and meta.json once it is over."""

    def __init__(self, call_dir: Path, meta: dict[str, Any]) -> None:
        call_dir.mkdir(parents=True, exist_ok=True)
        self.dir = call_dir
        self.meta = meta
        self.t0 = time.time()
        self._events = (call_dir / "events.jsonl").open("a", encoding="utf-8")

    def mark_answered(self) -> None:
        self.t0 = time.time()
        self.meta["answered_at"] = _iso(self.t0)

    def event(self, kind: str, **data: Any) -> None:
        record = {"t": round(time.time() - self.t0, 2), "type": kind, **data}
        self._events.write(json.dumps(record, default=str) + "\n")
        self._events.flush()

    def finish(self, end_reason: str) -> None:
        self.meta["end_reason"] = end_reason
        self.meta["ended_at"] = _iso(time.time())
        if "answered_at" in self.meta:
            self.meta["duration_s"] = round(time.time() - self.t0, 1)
        self._events.close()
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def make_tts(voice_type: str) -> Any:
    voice = VOICES[TTS_PROVIDER][voice_type]
    if TTS_PROVIDER == "cartesia":
        return cartesia.TTS(model="sonic-3", voice=voice)
    return deepgram.TTS(model=voice)


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


# Explicit dispatch only: the worker joins a room when run.py asks for this agent by name.
os.environ.setdefault("LIVEKIT_AGENT_NAME", AGENT_NAME)
server = AgentServer(setup_fnc=prewarm)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    job = json.loads(ctx.job.metadata or "{}")
    scenario = load_scenarios()[job.get("scenario") or os.getenv("SCENARIO", "new-booking")]
    console = ctx.is_fake_job()

    # first writer wins: patient goodbye, time limit, silence; otherwise the clinic hung up
    end_reason: dict[str, str] = {}

    instructions = build_instructions(scenario)
    agent = PatientAgent(instructions)
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en-US", keyterm=STT_KEYTERMS),
        llm=FastClaude(model=PATIENT_LLM_MODEL, caching="ephemeral", max_tokens=300),
        tts=make_tts(scenario.persona["voice"]),
        vad=ctx.proc.userdata["vad"],
        turn_handling={
            "turn_detection": EnglishModel(),
            "endpointing": {"min_delay": 0.5, "max_delay": 3.0},
            "interruption": {"min_duration": 0.5},
            "preemptive_generation": {"enabled": True, "preemptive_tts": True},
        },
        user_away_timeout=20.0,
    )

    if console:
        await session.start(agent=agent, room=ctx.room)
        return

    call_id = job["call_id"]
    log = CallLog(
        CALLS_DIR / call_id,
        meta={
            "call_id": call_id,
            "scenario": scenario.id,
            "patient": scenario.persona["name"],
            "room": ctx.room.name,
            "from": require("CALLER_ID_NUMBER"),
            "to": TARGET_NUMBER,
            "llm": PATIENT_LLM_MODEL,
            "tts": f"{TTS_PROVIDER}:{VOICES[TTS_PROVIDER][scenario.persona['voice']]}",
            "started_at": _iso(time.time()),
        },
    )

    _attach_event_logging(session, log)

    max_minutes = float(scenario.behavior.get("max_minutes", DEFAULT_MAX_MINUTES))
    try:
        # RoomIO needs the room connected before it can link to a specific participant.
        await ctx.connect()
        await session.start(
            agent=agent,
            room=ctx.room,
            room_options=room_io.RoomOptions(participant_identity=CLINIC_IDENTITY),
            record=False,  # we run our own recorder below so the file lands in calls/<id>/
        )
        await _dial(ctx, max_minutes)
    except api.SipCallError as e:
        logger.error("call failed: %s %s", e.sip_status_code, e.sip_status)
        log.meta["sip_status"] = f"{e.sip_status_code} {e.sip_status}"
        log.finish("dial_failed")
        ctx.shutdown(reason="dial failed")
        return
    except Exception:
        log.finish("setup_error")  # unblock run.py instead of letting it wait out the timeout
        raise

    log.mark_answered()
    log.event("answered")
    agent.answered_at = log.t0

    # Record only once the call is answered, so 00:00 in the file is pickup time.
    recorder = RecorderIO(agent_session=session)
    session.input.audio = recorder.record_input(session.input.audio)
    session.output.audio = recorder.record_output(session.output.audio)
    await recorder.start(output_path=log.dir / "recording.ogg")

    async def _finalize() -> None:
        await recorder.aclose()
        log.finish(end_reason.get("reason", "remote_hangup"))

    ctx.add_shutdown_callback(_finalize)

    @session.on("close")
    def _on_close(ev: Any) -> None:
        log.event("session_closed", reason=str(getattr(ev, "reason", "")))
        ctx.shutdown(reason="session closed")

    _start_call_guards(ctx, session, agent, scenario, instructions, log, max_minutes, end_reason)


async def _dial(ctx: JobContext, max_minutes: float) -> None:
    number = checked_target()  # belt and braces: run.py checks too, but never dial off-list

    logger.info("dialing %s", number)
    await ctx.api.sip.create_sip_participant(
        api.CreateSIPParticipantRequest(
            room_name=ctx.room.name,
            sip_trunk_id=require("LIVEKIT_SIP_TRUNK_ID"),
            sip_call_to=number,
            sip_number=require("CALLER_ID_NUMBER"),
            participant_identity=CLINIC_IDENTITY,
            participant_name="Clinic",
            wait_until_answered=True,
            ringing_timeout=Duration(seconds=40),
            # server-side backstop in case this process dies mid-call
            max_call_duration=Duration(seconds=int(max_minutes * 60) + 60),
        )
    )


def _attach_event_logging(session: AgentSession, log: CallLog) -> None:
    @session.on("conversation_item_added")
    def _on_item(ev: Any) -> None:
        item = ev.item
        if isinstance(item, ChatMessage) and item.role in ("user", "assistant"):
            log.event(
                "turn",
                speaker="AGENT" if item.role == "user" else "PATIENT",
                text=item.text_content,
                interrupted=item.interrupted,
                metrics=dict(item.metrics),
            )

    @session.on("user_state_changed")
    def _on_user_state(ev: Any) -> None:
        log.event("clinic_state", state=ev.new_state)

    @session.on("agent_state_changed")
    def _on_agent_state(ev: Any) -> None:
        log.event("patient_state", state=ev.new_state)

    @session.on("function_tools_executed")
    def _on_tools(ev: Any) -> None:
        log.event("tools", calls=[c.name for c in ev.function_calls])

    @session.on("agent_false_interruption")
    def _on_false_interruption(ev: Any) -> None:
        log.event("false_interruption", resumed=ev.resumed)

    @session.on("error")
    def _on_error(ev: Any) -> None:
        log.event("error", error=str(ev.error))


def _start_call_guards(
    ctx: JobContext,
    session: AgentSession,
    agent: PatientAgent,
    scenario: Scenario,
    instructions: str,
    log: CallLog,
    max_minutes: float,
    end_reason: dict[str, str],
) -> None:
    """Hang-up, greeting nudge, silence handling, time limits and the optional scripted barge-in."""
    state = {"clinic_spoke": False, "away_nudges": 0, "barged_in": False}
    tasks: set[asyncio.Task[Any]] = set()

    def spawn(coro: Any) -> None:
        task = asyncio.create_task(coro)
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def cancel_guards() -> None:
        for task in list(tasks):
            task.cancel()

    ctx.add_shutdown_callback(cancel_guards)

    def hang_up(reason: str) -> None:
        end_reason.setdefault("reason", reason)
        ctx.delete_room()
        ctx.shutdown(reason=reason)

    async def hang_up_after_goodbye() -> None:
        # the goodbye has finished playing; don't cut off the clinic's own closing line
        await asyncio.sleep(0.8)
        deadline = time.time() + HANGUP_WAIT_FOR_CLINIC_S
        while session.user_state == "speaking" and time.time() < deadline:
            await asyncio.sleep(0.2)
        hang_up("patient_goodbye")

    @session.on("conversation_item_added")
    def _on_item(ev: Any) -> None:
        # fires once our message has been played out; only then is the goodbye actually said
        if isinstance(ev.item, ChatMessage) and ev.item.role == "assistant" and agent.wants_to_hang_up:
            agent.wants_to_hang_up = False
            log.event("hangup_requested")
            spawn(hang_up_after_goodbye())

    async def greeting_nudge() -> None:
        await asyncio.sleep(GREETING_GRACE_S)
        if not state["clinic_spoke"]:
            session.say("Hi, hello?")

    async def clock() -> None:
        await asyncio.sleep((max_minutes - 0.5) * 60)
        log.event("wrap_up_requested")
        await agent.update_instructions(
            instructions
            + "\n## Time check\nYou need to go very soon. Wrap up politely in your next turn: "
            f"say goodbye and end that reply with {HANGUP_MARKER}."
        )
        await asyncio.sleep(60)
        hang_up("max_duration")

    trigger = str(scenario.behavior.get("interrupt_on", "")).lower()

    @session.on("user_input_transcribed")
    def _on_clinic_words(ev: Any) -> None:
        # Scripted barge-in: cut in the moment the clinic says the trigger word mid-utterance
        # (e.g. while offering Tuesday slots), so the interruption is real and on-topic.
        if (
            trigger
            and not state["barged_in"]
            and session.user_state == "speaking"
            and trigger in ev.transcript.lower()
        ):
            state["barged_in"] = True
            log.event("barge_in", heard=ev.transcript, line=scenario.behavior["interrupt_line"])
            session.say(scenario.behavior["interrupt_line"], allow_interruptions=False)

    @session.on("user_state_changed")
    def _on_user_state(ev: Any) -> None:
        if ev.new_state == "speaking":
            state["clinic_spoke"] = True
        elif ev.new_state == "away":
            state["away_nudges"] += 1
            if state["away_nudges"] == 1:
                session.say("Hello? Are you still there?")
            else:
                hang_up("silence")

    spawn(greeting_nudge())
    spawn(clock())


if __name__ == "__main__":
    cli.run_app(server)
