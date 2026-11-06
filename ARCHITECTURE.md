# Architecture

`python -m caller.run` starts a LiveKit Agents worker and places one call per scenario. For each
call it creates a LiveKit Cloud room and dispatches the worker into it. The worker starts a
**pipeline** `AgentSession` and dials the target line through a Twilio Elastic SIP trunk, using
LiveKit's outbound SIP (`create_sip_participant`, which waits for the answer). Once the line
connects, the clinic's audio flows through Silero VAD and Deepgram Nova-3 streaming STT. LiveKit's
semantic turn detector decides when the clinic has finished its turn. Claude Haiku 5.5 then
writes the patient's reply from a persona prompt (identity, goal, hidden probe, a calendar), and
the reply is spoken with Deepgram Aura-2 TTS. A `RecorderIO` tap writes a stereo OGG (clinic on
the left, patient on the right) containing only audio that actually played, so it stays correct
through interruptions. After hang-up, the recording is re-transcribed with Deepgram's
multichannel mode, giving a two-speaker transcript whose timestamps match the audio exactly.
Claude Opus 5.5 then reviews it against the scenario's expected behaviour, and a cross-call pass
compares every fact the agent stated (hours, address, coverage) across calls.

The design is built around what the other side of the call hears: a caller that waits its turn,
answers quickly, stays on task and hangs up like a person. So most of the effort went into
turn-taking and latency rather than infrastructure, and almost every behaviour below came from
listening to real calls and measuring, not from the first draft.

## Key decisions and trade-offs

**Telephony: Twilio Elastic SIP → LiveKit Cloud outbound trunk.** LiveKit's own phone numbers
are inbound-only, so outbound calls need a SIP provider. Twilio is the best-documented pairing.
Telnyx is cheaper per minute but slower to verify. LiveKit Cloud rather than self-hosting:
nothing to run locally (no Docker), and the free tier covers this volume.

**STT: Deepgram Nova-3 (streaming).** Low streaming latency (~0.25 s from end of speech to a
final transcript in my logs), robust on 8 kHz phone audio, and keyterm prompting. Keyterms are
only for proper nouns confirmed by listening: an unusual doctor's name was transcribed as a
different name, which at first looked like an agent bug. The same key also does the post-call
multichannel transcription.

**LLM: Claude Haiku 5.5 live, Claude Opus 5.5 offline.** The live model's job is a believable
caller with short turns, where time-to-first-token matters more than depth. Haiku with thinking
disabled answers in ~0.5 s. Bug-finding runs after the call, where latency doesn't matter, so it
gets the strongest model with structured output. One measured surprise: sending even a single
tool definition (`end_call`) added **0.2–0.4 s** to every Haiku response. The patient now ends
calls with a `<hangup>` text marker that is stripped from the stream before TTS, instead of a
function tool.

**TTS: Deepgram Aura-2 by default, Cartesia Sonic-3 optional.** Cartesia sounded the most
natural. Aura-2 runs on the same Deepgram key as speech-to-text, streams first audio in ~0.15 s,
and is selectable via `TTS_PROVIDER`. Trade-off: Aura-2 mispronounces filler words ("um" came out
as "Bob"), so the persona avoids them and any that slip through are stripped before TTS.

**Turn-taking.**
- Silero VAD, LiveKit's English turn-detector model, and a 0.5–3.0 s endpointing window.
- A clinic AI pauses mid-utterance while it "checks the schedule", and the semantic detector
  is what stops the patient jumping into those gaps.
- The clinic speaks first, as on a real call. A greeting gate suppresses replies until the clinic
  asks its opening question, because a recording disclaimer with a 3 s pause was being answered.
- Hang-up waits until the goodbye has played and the clinic has stopped talking.

**Interruptions.**
- Barge-in by the clinic is on (0.5 s minimum, false-interruption resume).
- To test the clinic's handling of interruptions, the `barge-in` scenario cuts in on purpose
  when the clinic's live transcript contains a trigger word ("Tuesday"), so the interruption is
  real and on-topic. An earlier timer-based version fired out of context.

**Latency.**
- Clinic stops speaking → patient starts speaking, measured from per-turn metrics
  (`python -m caller.latency`):
  - after the fixes: **p50 ~1.3–1.4 s**, with end-of-turn ~0.60 s + LLM first token ~0.6 s +
    TTS first audio ~0.15–0.2 s;
  - before the tool fix: p50 1.7–2.0 s.
- Preemptive generation and preemptive TTS are on.
- The largest remaining fixed cost is the 0.5 s minimum endpointing delay. I kept it on purpose.
  Even at 0.5 s the patient occasionally cut into the clinic's mid-utterance pauses, and a
  shorter window would make that more likely. Talking over the other side sounds worse than half
  a second of thinking time.

**Safety rails.**
- The bot only dials `TARGET_NUMBER`, and only if it's also listed in `ALLOWED_TARGETS`; this is
  checked before the run starts and again right before dialing.
- Each call has a 3:30 wrap-up instruction, a 4:30 hard stop, a server-side
  `max_call_duration` backstop, and silence handling (nudge, then hang up).

## What listening to real calls taught me

- **Tools cost latency.** One function definition added 0.2–0.4 s to every reply, so hanging up
  became a text marker instead of a tool call.
- **Callers don't answer disclaimers.** The bot replied into the pause in the clinic's recording
  notice; a greeting gate now waits for the first real question.
- **TTS has opinions about "um".** Aura-2 read filler words as other words, so the prompt bans
  them and a filter strips any that slip through.
- **Check the transcript before blaming the agent.** A misheard doctor's name looked like an
  agent bug until I listened to the recording. Keyterm hints are added only after that check.
- **Scripted interruptions must be on-topic.** A timer-based barge-in cut in at random moments;
  triggering on a word the clinic is actually saying makes it a real test.
- **Half a second of silence beats talking over someone.** Shorter endpointing made the caller
  faster on paper and ruder on the phone.
