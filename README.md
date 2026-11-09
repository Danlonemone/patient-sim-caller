# Patient Simulator Caller

An automated caller that stress-tests AI phone receptionists. It phones a medical office's voice
agent, plays a realistic patient (scheduling, refills, insurance questions, emergencies, privacy
probes, interruptions), records both sides of the call, transcribes it, and drafts a QA review of
how the receptionist behaved.

Built on **LiveKit Agents in pipeline mode**: Deepgram Nova-3 (speech-to-text) → Claude Haiku 5.5
(patient brain) → Deepgram Aura-2 or Cartesia Sonic-3 (text-to-speech), dialing out over a Twilio
SIP trunk. See [ARCHITECTURE.md](ARCHITECTURE.md) for how it works and why.

## Why I built this

Voice agents are hard to test. Text evals miss what goes wrong on a real phone line: talking over
each other, mishearing a date, losing the thread after an interruption, or confidently confirming
an appointment that can't exist. Testing by hand means calling the agent yourself, over and over,
and trying to remember what it said. I wanted a caller that behaves like a real patient (it waits
its turn, answers briefly, pushes on the edge cases a careful tester would) and leaves behind a
recording, a transcript with timestamps and a draft bug list for every call.

## Responsible use

- The bot only dials `TARGET_NUMBER`, and only if that number is also listed in
  `ALLOWED_TARGETS`. Both are set in `.env`; there's no way to pass a number on the command line.
- Only point it at agents you own or have permission to test.
- Every call has a wrap-up prompt at 3:30, a hard stop at 4:30, a server-side maximum call
  length, and hangs up after prolonged silence.

## Run it

After the one-time setup below:

```bash
python -m caller.run all
```

That starts the LiveKit worker, places one call per scenario (one at a time), and for each call
writes `calls/NN-<scenario>/`:

| File | What it is |
|---|---|
| `recording.ogg` | Stereo recording, receptionist on the left channel, patient bot on the right |
| `transcript.txt` | Timestamped two-speaker transcript; timestamps match the recording |
| `analysis.json` | Claude Opus 5.5 review: candidate findings, facts the agent stated, bot issues |
| `events.jsonl` | Turn-by-turn log with latency metrics (end-of-turn delay, LLM TTFT, TTS TTFB) |
| `meta.json` | Scenario, timings, how the call ended |

It finishes with `reports/findings-draft.md`, which ranks every candidate finding and flags facts
the agent contradicted across calls. Treat it as a draft: listen to the audio before you believe
a finding.

Other useful commands:

```bash
python -m caller.run new-booking barge-in    # specific scenarios
python -m caller.run all --no-analyze        # calls + transcripts only
python -m caller.agent console               # talk to the patient yourself with a mic (no phone call)
python -m caller.transcribe calls/03-cancel  # re-run one step on an existing call
python -m caller.analyze --summary
python -m caller.latency                     # response-latency report across calls
```

Scenarios live in [caller/scenarios.yaml](caller/scenarios.yaml): each has a persona, a goal, a
hidden *probe* (what the patient pushes on), and the *expected* correct behaviour used by the
analyzer. Change `clinic`, `clinic_type` and the personas to match the agent you're testing.

## Results

I ran 20 calls against a demo AI receptionist for an orthopedic practice, covering 17 scenarios,
and kept 13 findings after checking each one against the audio.

**Latency.** From the moment the receptionist stops talking to the moment the patient starts, the
median is about **1.3–1.4 s**: roughly 0.6 s end-of-turn detection, 0.6 s to the LLM's first token
and 0.15–0.2 s to the first audio. Early versions sat at 1.7–2.0 s; most of the gain came from
removing a single tool definition (see [ARCHITECTURE.md](ARCHITECTURE.md)).

**What it catches.** The most useful findings were the kind a scripted test wouldn't reach:

- saying it accepts insurance plans it can't know about, including a made-up one
- giving out a patient's appointment details to a family member who hadn't been verified
- missing blood-clot warning signs until the caller asked about them directly
- reading the caller's phone number back as their insurance member ID
- silently skipping over a date that doesn't exist instead of saying so
- greeting callers by the on-file patient's name based on caller ID alone

One finding turned out to be the caller's own speech-to-text mishearing a doctor's name, which is
why every draft finding is checked against the recording before it counts.

**What it taught me about the caller.** Most of the work went into making the bot sound like a
person rather than a test script. Every behaviour described in ARCHITECTURE.md came from listening
to a call that went wrong.

## One-time setup

### 1. Python environment (Python 3.10–3.14; developed on 3.12)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python -m caller.agent download-files   # turn-detector + VAD model weights (~460 MB)
cp .env.example .env                    # then fill it in (steps 2-5)
```

### 2. Provider keys

| Variable | Where to get it |
|---|---|
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | [LiveKit Cloud](https://cloud.livekit.io) → project → Settings → Keys |
| `DEEPGRAM_API_KEY` | [Deepgram console](https://console.deepgram.com) (speech-to-text and the default text-to-speech) |
| `CARTESIA_API_KEY` | Only if `TTS_PROVIDER=cartesia`: [Cartesia](https://play.cartesia.ai) → API keys |
| `ANTHROPIC_API_KEY` | [Claude Console](https://platform.claude.com) → API keys |

### 3. Twilio Elastic SIP trunk (outbound calling)

1. Upgrade the Twilio account (trial accounts can only call verified numbers) and buy one US
   phone number. Put it in `CALLER_ID_NUMBER` in E.164 form, e.g. `+18055550100`.
2. **Elastic SIP Trunking → Trunks → Create**. Under **Termination**, pick a Termination SIP URI
   (e.g. `my-caller.pstn.twilio.com`) → `TWILIO_TERMINATION_URI`.
3. Under Termination → **Authentication → Credential Lists**, create a username/password →
   `TWILIO_SIP_USERNAME`, `TWILIO_SIP_PASSWORD`.
4. Under **Numbers**, attach the number from step 1 to the trunk.

### 4. Register the trunk with LiveKit

```bash
python -m caller.setup_trunk
```

Creates the LiveKit outbound SIP trunk and writes `LIVEKIT_SIP_TRUNK_ID` into `.env`.

### 5. Choose the target

Set `TARGET_NUMBER` to the receptionist line you're testing and add the same number to
`ALLOWED_TARGETS`.

## Tests

```bash
pip install pytest
pytest
```

The tests cover the logic that can break silently without a phone call: the hang-up marker is
never spoken even when it streams in split across chunks, filler words are stripped, the bot
refuses any number that isn't valid and allowlisted, and every scenario in `scenarios.yaml` is
complete and renders into a clean prompt. They run in CI on every push.

## Repository layout

```
caller/
  agent.py         LiveKit worker: pipeline session, dial-out, recording, end-call, safety limits
  speech.py        strips the hang-up marker and filler words from the reply stream before TTS
  persona.py       loads scenarios and builds the patient's system prompt
  scenarios.yaml   clinic, personas + test scenarios
  run.py           single entry point (worker + calls + transcripts + analysis)
  transcribe.py    Deepgram multichannel transcription of the stereo recording
  analyze.py       Claude Opus 5.5 post-call review and cross-call contradiction check
  latency.py       response-latency report from the per-turn metrics
  setup_trunk.py   one-time LiveKit outbound trunk registration
  config.py        env loading, paths, the target-number allowlist check
tests/             pytest suite (no API keys or LiveKit install needed)
calls/             one folder per call (created on first run; not committed)
reports/           machine-generated findings draft (not committed)
```

## License

[MIT](LICENSE)
