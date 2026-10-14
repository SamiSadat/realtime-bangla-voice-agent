"""Compare configurations against the same scripted call.

A single latency number is only worth as much as what it is compared against.
This runs the identical call under several configurations and prints them side
by side, so a claim like "the endpointing wait dominates" is demonstrated
rather than asserted.

    python -m bvagent.bench.sweep --dimension vad
    python -m bvagent.bench.sweep --dimension stt --calls 2

Every variant replays byte-identical caller audio through the same pipeline, so
the only thing that differs between rows is the setting named in the row.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

from loguru import logger

from bvagent.bench.run import run_call
from bvagent.config import Settings, load_settings
from bvagent.profiling.report import LatencyReport, _ms
from bvagent.profiling.trace import TurnTrace

# A variant is a name plus a function that returns modified settings.
Variant = tuple[str, Callable[[Settings], Settings]]


def _with_vad(stop_secs: float, turn_stop: float) -> Callable[[Settings], Settings]:
    def apply(s: Settings) -> Settings:
        s = replace(s)
        s.vad = replace(s.vad, stop_secs=stop_secs, turn_stop_timeout=turn_stop)
        return s

    return apply


def _with_stt(model: str, beam: int = 1) -> Callable[[Settings], Settings]:
    def apply(s: Settings) -> Settings:
        s = replace(s)
        s.stt = replace(s.stt, model=model, beam_size=beam)
        return s

    return apply


def _with_stt_device(device: str, compute: str) -> Callable[[Settings], Settings]:
    def apply(s: Settings) -> Settings:
        s = replace(s)
        s.stt = replace(s.stt, device=device, compute_type=compute)
        return s

    return apply


def _with_tts_chunk(max_chars: int) -> Callable[[Settings], Settings]:
    def apply(s: Settings) -> Settings:
        s = replace(s)
        s.tts = replace(s.tts, max_chars_per_chunk=max_chars)
        return s

    return apply


def _with_thinking(enabled: bool) -> Callable[[Settings], Settings]:
    def apply(s: Settings) -> Settings:
        s = replace(s)
        s.llm = replace(s.llm, disable_thinking=not enabled)
        return s

    return apply


# Each dimension isolates one lever from the stage-breakdown table.
DIMENSIONS: dict[str, list[Variant]] = {
    # The turn-end wait is usually the largest single slice of TTFA, and it is
    # pure configuration — no faster hardware shortens it.
    "vad": [
        ("stop=0.2 turn=0.2", _with_vad(0.2, 0.2)),
        ("stop=0.35 turn=0.3", _with_vad(0.35, 0.3)),
        ("stop=0.5 turn=0.4", _with_vad(0.5, 0.4)),
        ("stop=0.8 turn=0.5", _with_vad(0.8, 0.5)),
    ],
    # Whether a smaller Whisper is worth it. On Bangla it generally is not:
    # the small models produce garbage, and garbage takes more decode steps.
    "stt": [
        ("small", _with_stt("small")),
        ("medium", _with_stt("medium")),
        ("large-v3", _with_stt("large-v3")),
        ("large-v3 beam=5", _with_stt("large-v3", beam=5)),
    ],
    "stt_device": [
        ("cuda/float16", _with_stt_device("cuda", "float16")),
        ("cpu/int8", _with_stt_device("cpu", "int8")),
    ],
    # How much text VITS is handed before it can emit its first sample.
    "tts_chunk": [
        ("chunk<=60", _with_tts_chunk(60)),
        ("chunk<=120", _with_tts_chunk(120)),
        ("chunk<=160", _with_tts_chunk(160)),
        ("chunk<=400 (no split)", _with_tts_chunk(400)),
    ],
    # Reasoning tokens are silence the caller sits through.
    "thinking": [
        ("thinking off", _with_thinking(False)),
        ("thinking on", _with_thinking(True)),
    ],
}


def sweep(
    dimension: str,
    *,
    settings: Settings,
    calls: int,
    with_barge_in: bool,
    out_dir: Path,
) -> list[tuple[str, LatencyReport]]:
    variants = DIMENSIONS[dimension]
    results: list[tuple[str, LatencyReport]] = []

    for name, apply in variants:
        logger.info(f"--- sweep {dimension}: {name} ---")
        variant_settings = apply(settings)
        traces: list[TurnTrace] = []
        for index in range(calls):
            try:
                report = asyncio.run(
                    run_call(
                        variant_settings,
                        with_barge_in=with_barge_in,
                        repeats=1,
                        out_dir=out_dir / dimension,
                        label=f"{_slug(name)}-call{index + 1}",
                    )
                )
                traces.extend(report.traces)
            except Exception:
                # One variant failing (a model that will not load, say) should
                # not cost the whole sweep the variants that did work.
                logger.exception(f"sweep variant {name!r} failed")
                break
        results.append(
            (
                name,
                LatencyReport.build(
                    traces, label=name, stack=variant_settings.stack_description()
                ),
            )
        )
    return results


def render(dimension: str, results: list[tuple[str, LatencyReport]], width: int = 92) -> str:
    lines: list[str] = []
    rule = "=" * width
    lines.append(rule)
    lines.append(f"  Sweep: {dimension}   (identical scripted call for every row)")
    lines.append(rule)
    header = (
        f"  {'variant':<24}{'n':>4}{'TTFA p50':>11}{'TTFA p95':>11}"
        f"{'turn-end':>10}{'LLM':>8}{'STT':>8}{'TTS':>8}"
    )
    lines.append(header)
    lines.append("  " + "-" * (width - 4))

    baseline: float | None = None
    for name, report in results:
        ttfa = report.ttfa()
        if not ttfa.n:
            lines.append(f"  {name:<24}{'-':>4}   (no measured turns)")
            continue
        stages = {d.name: d for d in report.stage_distributions()}
        concurrent = {d.name: d for d in report.concurrent_distributions()}
        if baseline is None:
            baseline = ttfa.p50
        lines.append(
            f"  {name:<24}{ttfa.n:>4}{_ms(ttfa.p50):>11}{_ms(ttfa.p95):>11}"
            f"{_ms(stages['turn_end_wait'].p50):>10}"
            f"{_ms(stages['llm_ttft'].p50):>8}"
            f"{_ms(concurrent['stt'].p50):>8}"
            f"{_ms(stages['tts_ttfb'].p50):>8}"
        )

    lines.append("  " + "-" * (width - 4))
    if baseline:
        best = min(
            (r for _, r in results if r.ttfa().n),
            key=lambda r: r.ttfa().p50 or float("inf"),
            default=None,
        )
        if best is not None and best.ttfa().p50 is not None:
            delta = baseline - best.ttfa().p50
            lines.append(
                f"  Best p50 TTFA: {best.label} at {_ms(best.ttfa().p50)}"
                + (f"  ({_ms(delta)} under the first row)" if delta > 0 else "")
            )
    lines.append(rule)
    return "\n".join(lines)


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in name).strip("-")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dimension", choices=sorted(DIMENSIONS), default="vad", help="Which lever to sweep."
    )
    parser.add_argument("--scenario", choices=["loan", "booking"], default=None)
    parser.add_argument("--calls", type=int, default=1, help="Calls per variant.")
    parser.add_argument("--barge-in", action="store_true", help="Include the interruption turn.")
    parser.add_argument("--out", default="runs/sweeps")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    settings = load_settings()
    if args.scenario:
        settings.scenario = args.scenario

    out_dir = Path(args.out)
    results = sweep(
        args.dimension,
        settings=settings,
        calls=args.calls,
        with_barge_in=args.barge_in,
        out_dir=out_dir,
    )

    text = render(args.dimension, results)
    print()
    print(text)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.dimension}.json").write_text(
        json.dumps(
            {
                "dimension": args.dimension,
                "variants": [
                    {"name": name, **report.to_dict(include_turns=False)}
                    for name, report in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    (out_dir / f"{args.dimension}.txt").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
