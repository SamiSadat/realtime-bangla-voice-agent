# Real-time Bangla voice agent, with the latency broken down

A speech-to-speech agent that holds a Bangla phone conversation — VAD, streaming
Whisper, an LLM with tool calls, and Bangla TTS on a [Pipecat](https://pipecat.ai)
pipeline — and **measures where every millisecond of its response time goes**.

The pipeline is the ordinary part. The part worth reading is
[`src/bvagent/profiling/`](src/bvagent/profiling/) and
[`src/bvagent/bench/`](src/bvagent/bench/): a frame observer that reconstructs a
per-turn timeline, a replay harness that drives the real pipeline from recorded
audio so runs are reproducible, and a sweep tool that compares configurations
against the identical call.

Two scenarios, both drawn from what Bangla voice deployments actually do:
**loan payment reminders** and **branch appointment booking**, each with real
tool calls against a backing store.

---

## Why time-to-first-audio

A voice agent is judged on one number: how long the caller sits in silence after
they stop talking, before they hear a reply. That is **time-to-first-audio
(TTFA)**, and it is the only latency a caller can perceive. Component
benchmarks — "our STT is 200ms" — do not add up to it, because the stages
overlap, and because the largest contributor is usually not a model at all.

So everything here measures TTFA first and attributes it second.

---

## The measurement

`TurnProfiler` ([`profiling/observer.py`](src/bvagent/profiling/observer.py)) is
a Pipecat `BaseObserver`. It watches frames flow between processors and never
pushes any, so it cannot perturb what it measures. Each user turn becomes a
`TurnTrace` with a timeline:

```
caller falls silent ──┐
                      ├── turn-end wait (VAD stop_secs + turn_stop_timeout)
                      │   └── STT decode runs *inside* this window
   turn handed to LLM ─┤
                      ├── LLM time-to-first-token
                      ├── LLM → first chunk TTS will speak
                      ├── TTS time-to-first-byte
                      └── transport
   bot voice starts ──┘   = TTFA
```

Three things this gets right that a naive breakdown does not:

**1. Timing is corrected back to the speech, not the frame.** Silero reports
"user stopped" only after hearing `stop_secs` of silence — so the frame arrives
half a second after the caller actually stopped. Timing from the frame credits
the pipeline with latency the caller already spent waiting. Every VAD moment is
corrected by the `stop_secs` / `start_secs` the frame carries. Getting this
wrong understated TTFA by ~500ms in an early version of this repo.

**2. STT is concurrent, not sequential.** Transcription runs while the
endpointing timer counts down. Adding it to the critical path would make the
parts sum to more than the measured whole. It is reported separately, with
`stt_on_critical_path` — the part that actually outlasted the window — and with
`stt_binding_fraction`, the share of turns whose end was set by the transcript
rather than by the timer. Both are needed: at the default setting the first is
0ms, meaning a faster STT would buy nothing; shorten the endpointing window far
enough and the first is *still* 0ms while the second climbs to 1, because STT
has become the constraint without ever appearing on the critical path. See
[the sweep below](#what-actually-moved-the-needle).

**3. Barge-in is its own measurement.** How long the bot keeps talking after
being interrupted is a different number from response latency, and on a phone
call it is the one that makes an agent feel deaf. Pipecat broadcasts an
`InterruptionFrame` on *every* turn start, whether or not the bot held the
floor; counting those made the "barge-in" figure really the length of an
uninterrupted reply. Only interruptions landing while the bot is actually
speaking are counted.

Traces are kept as data, not logged and lost, so a run yields percentiles.
p95 is the caller who was about to hang up; a mean hides them.

---

## Reproducible runs

`python -m bvagent.bench.run` replays a scripted Bangla call through the real
pipeline — same VAD, same models, same observer — with a WAV file in place of a
person.

Two details make the numbers trustworthy:

- **Audio is fed at realtime pace, and captured at realtime pace.** Pipecat's
  output stage does not sleep between audio frames; it relies on the transport
  taking as long as the audio lasts, which is what a phone line does. A capture
  transport that returns instantly drains a reply in microseconds, which both
  fakes the turn timing and leaves no window in which a barge-in could land.
- **The script waits for the reply to *start*, not merely for the bot to be
  idle.** Between an utterance ending and a reply beginning, the bot is idle and
  still flagged finished from the previous turn. A script that takes that as its
  cue speaks a fraction of a second into the reply and interrupts every turn —
  destroying the latency it exists to measure.

Caller audio is synthesized with the same MMS voice the agent speaks with. That
is a deliberate trade: it makes the input byte-identical across runs, so a
change in the numbers is a change in the system. It also means **these runs
measure latency, not transcription accuracy** — synthetic speech is cleaner than
a phone line, and no WER claim should be made from it. Point the harness at real
recordings in `data/audio/` for accuracy work.

---

## Results

All numbers below were produced by `make bench CALLS=4` on this repo — 24 turns
of the loan-reminder scenario, replayed identically. Hardware: RTX PRO 6000
(Blackwell, 96GB), Ryzen 9 9950X. Everything local; nothing over the network.

| component | configuration |
| --- | --- |
| VAD | Silero, `stop_secs=0.5` + `turn_stop_timeout=0.4` |
| STT | faster-whisper `large-v3`, cuda/float16, `beam=1`, `lang=bn` |
| LLM | Qwen3.8-27B via Ollama, `max_tokens=160`, reasoning off |
| TTS | `facebook/mms-tts-ben`, cuda |

### Time to first audio

| metric | n | p50 | p90 | p95 |
| --- | ---: | ---: | ---: | ---: |
| **TTFA** — what the caller experiences | 24 | **1953 ms** | 2912 ms | 2945 ms |
| TTFA excluding the endpointing wait | 24 | 847 ms | 1771 ms | 2037 ms |

### Where it goes

| stage | p50 | p95 | share of median TTFA | what moves it |
| --- | ---: | ---: | ---: | --- |
| Turn-end wait (VAD) | 901 ms | 901 ms | **46%** | `stop_secs` + `turn_stop_timeout` |
| Context → LLM accepted | 0 ms | 1 ms | 0% | model resident, server not busy |
| LLM time-to-first-token | 656 ms | 1824 ms | **34%** | model, prompt length, device |
| LLM → first TTS chunk | 152 ms | 249 ms | 8% | sentence vs. token streaming |
| TTS time-to-first-byte | 72 ms | 97 ms | 4% | chunk size, model, device |
| Transport / playout | 3 ms | 5 ms | 0% | chunk size, network |
| _unattributed_ | 0 ms | 1 ms | 0% | frame hops, scheduling |

Concurrent with the turn-end wait, and therefore **not** added to TTFA:

| | p50 | p95 |
| --- | ---: | ---: |
| STT decode | 689 ms | 757 ms |
| …of which the caller waited on | **0 ms** | **0 ms** |
| …measured from the VAD stop frame (TTFS) | 189 ms | 257 ms |

### Interruption and tools

| | n | p50 | p95 |
| --- | ---: | ---: | ---: |
| **Barge-in** — caller talks over bot → bot audio stops | 4 | **202 ms** | 202 ms |
| Tool call round trip | 8 | 51 ms | 52 ms |

The tool backend simulates a 50ms dependency, so that row is the harness
checking itself: it confirms tool time is attributed to tools rather than
smeared into LLM latency.

---

## What actually moved the needle

**The biggest cost is not a model.** The turn-end wait is 46% of median TTFA and
pure configuration — no faster GPU shortens it. But it does not shorten
indefinitely either. `make sweep DIM=vad`, 15 turns per row, identical call:

| `stop_secs` / `turn_stop` | configured wait | **measured turn-end** | STT decode | TTFA p50 |
| --- | ---: | ---: | ---: | ---: |
| 0.2 / 0.2 | 400 ms | **737 ms** | **736 ms** | 2626 ms |
| 0.35 / 0.3 | 650 ms | 1048 ms | 611 ms | 2807 ms |
| 0.5 / 0.4 | 900 ms | 901 ms | 687 ms | 2050 ms |
| 0.8 / 0.5 | 1300 ms | 1301 ms | 992 ms | 2001 ms |

**Look at the top row.** The configured wait is 400ms; the measured wait is
737ms — and the STT decode is 736ms. The turn cannot end without a transcript,
so once the endpointing window is cut below the decode time, the pipeline simply
waits for Whisper instead and the knob stops doing anything. At 0.5 and 0.8 the
configured and measured waits agree to within a millisecond; at 0.2 they do not.

That crossover is invisible to the obvious metric. `stt_on_critical_path` reads
**0ms on every row**, including the one where STT has become the constraint,
because the transcript still arrives just before the turn closes. So the report
carries a second figure — `stt_binding_fraction`, the share of turns whose end
was set by the transcript rather than by the timer. While it is 0, shortening
the window shortens TTFA; once it climbs, the lever has quietly become STT speed.

The coupling runs the other way too: decode time *rises* with `stop_secs`
(611ms at 0.35 → 992ms at 0.8), because the trailing silence is part of the
segment Whisper is handed. So the window is tied to STT at both ends — too short
and the turn waits on the decode, too long and it makes the decode slower.

This is the whole argument for profiling a pipeline rather than its parts. Two
stages that look independent — a VAD setting and a model's decode time — are
coupled in both directions, and neither coupling shows up in a component
benchmark.

The default is 0.5/0.4: just above the floor, where configuration still sets the
wait and the number is predictable (p50 = p90 = p95 = 901 ms). Note what the
sweep *cannot* see: the replayed caller is synthetic and never hesitates, so it
measures the latency a short window buys and never the callers it cuts off
mid-clause. TTFA p50 across rows is also dominated by LLM run-to-run variance
(405–822 ms), which is why the turn-end column, not the TTFA column, is the one
to read here.

**At the default setting, a faster STT would buy nothing.** Whisper `large-v3`
decodes in 689ms, and on **100% of turns that finished inside the endpointing
window** — the caller waited on 0ms of it. Optimising it is wasted work until
the endpointing wait is cut first, at which point it immediately becomes the
constraint. Which of the two to work on is not a judgement call; it is whichever
the report says is binding.

**Bigger Whisper is both more accurate *and* faster on Bangla.** Measured
directly on the same clips:

| model | Bangla output | decode (2.9s clip) | RTF |
| --- | --- | ---: | ---: |
| `small` | Devanagari/Telugu garbage | 2923 ms | 0.72 |
| `medium` | wrong script, unusable | 2011 ms | 0.68 |
| **`large-v3`** | correct Bengali script | **255 ms** | **0.06** |

The smaller models do not fail gracefully on Bangla — they flail, emitting long
garbage sequences, and every garbage token is another decode step. The usual
size-versus-speed intuition is inverted here, and it only shows up if you
measure the language you are actually deploying in.

**Whisper's Bangla failure mode is a repetition loop, and it is a latency bug
before it is an accuracy bug.** On one utterance `large-v3` emitted
`আমি গুন্সান শান্দের` eleven times over, taking **15 seconds to decode 2.1
seconds of audio** (RTF 7.0). By the time that transcript landed, the turn had
already been closed without one, so the agent never replied at all — the caller
got 120 seconds of silence, not a wrong answer.

The cost is not in the loop itself but in Whisper's recovery: its quality checks
fail, so it re-decodes the window at each of six fallback temperatures. Three
guards bound it, in `STTConfig`:

| guard | value | what it does |
| --- | --- | --- |
| `repetition_penalty` | 1.15 | makes the loop unattractive to the decoder |
| `temperature` | `(0.0, 0.2)` | two fallback passes instead of six |
| `max_tokens_per_second` | 45 | hard ceiling on decode length per second of audio |

Sizing that last one needs measuring, not guessing: Bengali script runs 15–22
tokens per second of speech, and an initial cap of 28 truncated real sentences
mid-word. 45 leaves roughly 2x headroom and still bounds a runaway.

Fixing this took the booking scenario's worst turn from **130.7s to 4.5s**.

**Tell the turn strategy when a transcript is final.** A segmented STT decodes a
whole utterance in one pass, so its transcript is never revised — but Pipecat's
`TranscriptionFrame.finalized` defaults to `False`, so the turn-stop strategy
waits out its STT safety-net timer on every single turn. Setting `finalized=True`
lets it short-circuit.

**Name the gap between the turn closing and the LLM being asked.** An early run
had a turn at 130s TTFA with 121s of it filed under "unattributed overhead",
which said nothing about what had happened. It was the LLM server reloading an
evicted 27B model. The breakdown now has a `Context -> LLM accepted` row — 1ms
when the model is resident — and the benchmark warms the LLM before measuring,
the same way the STT and TTS services warm themselves.

**Turning reasoning off is fiddlier than it looks.** Ollama's `think: false` is
ignored on the OpenAI-compatible route, and the `/no_think` prompt suffix is not
honoured by every build — both still emitted ~300 reasoning tokens per turn.
`reasoning_effort: "none"` is the one that actually works.


---

## Running it

Requires Python 3.11+, and an OpenAI-compatible LLM endpoint (Ollama or vLLM
locally, or any hosted API). A CUDA GPU is optional — everything falls back to
CPU, and the report says which device produced each number.

```bash
make setup                      # venv, deps, and prefetch the models
make test                       # 42 tests, no models or GPU needed

make bench                      # replay a scripted call, print the report
make bench CALLS=4              # more turns, meaningful percentiles
make sweep DIM=vad              # compare endpointing settings
make sweep DIM=stt              # compare Whisper sizes
make serve                      # live agent on ws://localhost:8765
```

For a live call, run `make serve`, then open `client/index.html` in a browser
and allow the microphone. Speak Bangla; talk over the agent to test barge-in.
The browser and the pipeline exchange bare 16-bit PCM
([`serializer.py`](src/bvagent/serializer.py)) — no codec, no protobuf, so the
transport row of the report measures the network rather than a serializer.

Every knob that moves latency is an environment variable; see
[`.env.example`](.env.example).

---

## The stack, and why

| stage | choice | why |
| --- | --- | --- |
| VAD | Silero, timeout-based turn end | see below — the default is broken for Bangla |
| STT | faster-whisper `large-v3`, `language=bn` | more accurate *and* faster than smaller sizes on Bangla |
| LLM | any OpenAI-compatible endpoint | so a local and a hosted model can be compared on one harness |
| TTS | `facebook/mms-tts-ben` (VITS) | end-to-end, no separate vocoder pass, offline, 16 kHz |
| transport | WebSocket, raw PCM | what a telephony gateway and a browser both already speak |

Everything runs locally. For Bangla telephony that is usually the requirement
rather than a preference: no per-minute STT/TTS cost, and call audio never
leaves the box.

### Pipecat's default turn detection does not work in Bangla

Pipecat 1.10 defaults to `LocalSmartTurnAnalyzerV3`, a *semantic* end-of-turn
model, for deciding when the user has finished. It is trained on English. On
Bangla speech it returns `INCOMPLETE` for every utterance, indefinitely — so the
turn never ends, the aggregator holds it open, and an entire call accumulates
into a single user message that is only flushed at hangup. The agent appears to
ignore the caller completely.

This pipeline sets the stop strategy explicitly to
`SpeechTimeoutUserTurnStopStrategy`, which has no language model in it and so
has no opinion about Bangla. The cost is that turn-end becomes a fixed timeout
rather than a semantic judgement — which is exactly the `turn_end_wait` row in
the report, and why that row is charged to configuration rather than to a model.

### Other Bangla-specific handling

- **Sentence boundaries.** Bangla ends sentences with the danda `।`, not a full
  stop. Pipecat's boundary matcher handles it; NLTK's `punkt`, which it falls
  back on, does not split reliably on it. TTS chunking
  ([`mms_tts.py`](src/bvagent/services/mms_tts.py)) splits on `।` `॥` directly.
- **Numerals.** Callers and the LLM both produce Bengali (`৪৪২১`) or ASCII
  (`4421`) digits, sometimes in one utterance. Every identifier is normalized on
  the way in; every amount and date is rendered back in Bengali numerals, because
  ASCII digits make the TTS switch phoneme sets mid-sentence, which is audible.
- **Reply length is a latency figure.** The system prompt
  ([`prompts.py`](src/bvagent/agent/prompts.py)) caps replies at one or two
  sentences. That does more for perceived latency than any model swap.
- **The greeting is a fixed string**, spoken via `TTSSpeakFrame` rather than
  generated, so the call opens at TTS latency instead of LLM + TTS latency.

---

## Layout

```
src/bvagent/
  profiling/      trace.py      per-turn timeline and derived spans
                  observer.py   Pipecat observer that fills it in
                  report.py     percentiles, text and markdown reports
  bench/          replay.py     realtime-paced transport that replays WAVs
                  corpus.py     the scripted Bangla caller
                  run.py        one scripted call -> a report
                  sweep.py      compare configurations on the same call
  services/       whisper_stt.py  faster-whisper as a Pipecat STT service
                  mms_tts.py      MMS-TTS as a Pipecat TTS service, chunked
                  llm.py          OpenAI-compatible LLM with tool calls
  agent/          tools.py, backend.py, prompts.py
  pipeline.py     assembly, turn strategy, barge-in
  bot.py          live WebSocket server
client/index.html browser mic client
```

## Limitations

- Whisper `large-v3` on Bangla still garbles words; account numbers come through
  because they are digits. A Bangla fine-tune can be swapped in via
  `BV_STT_MODEL` and measured against stock on the same harness.
- The tool backend is in-memory, with a configurable simulated call latency so
  that tool time shows up in the report rather than hiding in it.
- The WebSocket transport serves one caller at a time; concurrency would need a
  per-call worker and a batched STT/TTS server.
