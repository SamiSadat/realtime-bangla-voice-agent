"""Aggregation and presentation of turn traces.

A single turn's latency is an anecdote. What a deployment is judged on is the
distribution: p95 TTFA is the caller who was about to hang up, and a mean hides
them. Everything here works on a list of :class:`TurnTrace`.
"""

from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from bvagent.profiling.trace import TurnTrace

# Stage order is pipeline order, so a printed breakdown reads as the path a
# turn actually takes.
STAGE_LABELS: list[tuple[str, str]] = [
    ("turn_end_wait", "Turn-end wait (VAD)"),
    ("llm_queue", "Context -> LLM accepted"),
    ("llm_ttft", "LLM time-to-first-token"),
    ("llm_to_sentence", "LLM -> first TTS chunk"),
    ("tts_ttfb", "TTS time-to-first-byte"),
    ("transport", "Transport / playout"),
]

# Measured, but overlapping the critical path rather than adding to it.
CONCURRENT_LABELS: list[tuple[str, str]] = [
    ("stt", "STT decode (Whisper)"),
    ("stt_on_critical_path", "  ...of which the caller waited on"),
    ("stt_ttfs", "  ...measured from the VAD stop frame"),
]

# What each stage responds to. Printed with the breakdown, because a latency
# table without this is a list of numbers nobody can act on.
STAGE_LEVERS: dict[str, str] = {
    "turn_end_wait": "VAD stop_secs + turn_stop_timeout (vs. cutting callers off)",
    "llm_queue": "model already resident, server not busy",
    "llm_ttft": "model size, prompt length, KV cache reuse, device",
    "llm_to_sentence": "TTS text aggregation: sentence vs. token streaming",
    "tts_ttfb": "vocoder chunk size, model, device",
    "transport": "output chunk size, network",
}


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile. ``q`` in [0, 100]."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (q / 100.0)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[int(pos)]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass
class Distribution:
    """Summary of one measured quantity across turns."""

    name: str
    n: int
    mean: float | None
    p50: float | None
    p90: float | None
    p95: float | None
    p99: float | None
    maximum: float | None

    @classmethod
    def of(cls, name: str, values: Iterable[float | None]) -> "Distribution":
        xs = [v for v in values if v is not None]
        if not xs:
            return cls(name, 0, None, None, None, None, None, None)
        return cls(
            name=name,
            n=len(xs),
            mean=sum(xs) / len(xs),
            p50=percentile(xs, 50),
            p90=percentile(xs, 90),
            p95=percentile(xs, 95),
            p99=percentile(xs, 99),
            maximum=max(xs),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n": self.n,
            **{
                k: (None if v is None else round(v * 1000, 1))
                for k, v in (
                    ("mean_ms", self.mean),
                    ("p50_ms", self.p50),
                    ("p90_ms", self.p90),
                    ("p95_ms", self.p95),
                    ("p99_ms", self.p99),
                    ("max_ms", self.maximum),
                )
            },
        }


@dataclass
class LatencyReport:
    """A run's worth of traces, summarized."""

    label: str
    traces: list[TurnTrace]
    stack: dict[str, str]
    created_at: str
    host: dict[str, str]

    @classmethod
    def build(
        cls,
        traces: Sequence[TurnTrace],
        *,
        label: str = "run",
        stack: dict[str, str] | None = None,
    ) -> "LatencyReport":
        resolved = dict(stack or {})
        if not resolved and traces:
            resolved = dict(traces[0].stack)
        return cls(
            label=label,
            traces=list(traces),
            stack=resolved,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            host={
                "platform": platform.platform(),
                "python": platform.python_version(),
                "machine": platform.machine(),
            },
        )

    # -- computed views --------------------------------------------------

    @property
    def measured(self) -> list[TurnTrace]:
        """Turns that produced a TTFA. The denominator for every stage stat."""
        return [t for t in self.traces if t.complete]

    def stage_distributions(self) -> list[Distribution]:
        return [
            Distribution.of(key, (t.stages()[key] for t in self.measured))
            for key, _ in STAGE_LABELS
        ]

    def concurrent_distributions(self) -> list[Distribution]:
        return [
            Distribution.of(key, (t.concurrent_stages()[key] for t in self.measured))
            for key, _ in CONCURRENT_LABELS
        ]

    def ttfa(self) -> Distribution:
        return Distribution.of("ttfa", (t.ttfa_secs for t in self.measured))

    def ttfa_engineered(self) -> Distribution:
        return Distribution.of(
            "ttfa_engineered", (t.ttfa_engineered_secs for t in self.measured)
        )

    def barge_in(self) -> Distribution:
        return Distribution.of("barge_in", (t.barge_in_secs for t in self.traces))

    def tool(self) -> Distribution:
        return Distribution.of("tool", (t.tool_secs for t in self.traces))

    def unattributed(self) -> Distribution:
        return Distribution.of("unattributed", (t.unattributed_secs for t in self.measured))

    # -- serialization ---------------------------------------------------

    def to_dict(self, *, include_turns: bool = True) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "label": self.label,
            "created_at": self.created_at,
            "host": self.host,
            "stack": self.stack,
            "turns_recorded": len(self.traces),
            "turns_measured": len(self.measured),
            "ttfa": self.ttfa().to_dict(),
            "ttfa_engineered": self.ttfa_engineered().to_dict(),
            "stages": [d.to_dict() for d in self.stage_distributions()],
            "concurrent": [d.to_dict() for d in self.concurrent_distributions()],
            "unattributed": self.unattributed().to_dict(),
            "tool": self.tool().to_dict(),
            "barge_in": self.barge_in().to_dict(),
        }
        if include_turns:
            doc["turns"] = [t.to_dict() for t in self.traces]
        return doc

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))
        return p

    # -- presentation ----------------------------------------------------

    def to_text(self, *, width: int = 78) -> str:
        """Plain-text report. No dependency on a terminal renderer, so it can
        be pasted into a PR, a README, or a CI log unchanged."""
        out: list[str] = []
        rule = "=" * width
        out.append(rule)
        out.append(f"  Latency report: {self.label}")
        out.append(f"  {self.created_at}   turns measured: {len(self.measured)}"
                   f" / {len(self.traces)} recorded")
        out.append(rule)

        if self.stack:
            out.append("")
            out.append("  Stack under test")
            for k, v in self.stack.items():
                out.append(f"    {k:<16} {v}")

        measured = self.measured
        if not measured:
            out.append("")
            out.append("  No turn produced a first-audio measurement.")
            out.append(rule)
            return "\n".join(out)

        ttfa = self.ttfa()
        out.append("")
        out.append("  Time to first audio (caller stops speaking -> bot voice starts)")
        out.append(f"    {_fmt_dist(ttfa)}")
        eng = self.ttfa_engineered()
        out.append("  Excluding the VAD endpointing wait (the part models control)")
        out.append(f"    {_fmt_dist(eng)}")

        out.append("")
        out.append("  Stage breakdown (median, and share of median TTFA)")
        out.append(f"    {'stage':<26}{'p50':>9}{'p95':>9}{'share':>8}   lever")
        out.append(f"    {'-' * (width - 6)}")
        total_p50 = ttfa.p50 or 0.0
        for dist, (key, label) in zip(self.stage_distributions(), STAGE_LABELS):
            share = ""
            if dist.p50 is not None and total_p50 > 0:
                share = f"{100 * dist.p50 / total_p50:.0f}%"
            out.append(
                f"    {label:<26}{_ms(dist.p50):>9}{_ms(dist.p95):>9}{share:>8}   "
                f"{STAGE_LEVERS.get(key, '')}"
            )
        un = self.unattributed()
        if un.p50 is not None:
            share = f"{100 * un.p50 / total_p50:.0f}%" if total_p50 > 0 else ""
            out.append(
                f"    {'(unattributed overhead)':<26}{_ms(un.p50):>9}"
                f"{_ms(un.p95):>9}{share:>8}   frame hops, scheduling"
            )

        out.append("")
        out.append("  Concurrent with the turn-end wait (not added to TTFA)")
        for dist, (_, label) in zip(self.concurrent_distributions(), CONCURRENT_LABELS):
            out.append(f"    {label:<26}{_ms(dist.p50):>9}{_ms(dist.p95):>9}")
        hidden = self.stt_hidden_fraction()
        if hidden is not None:
            out.append(
                f"    STT finished inside the endpointing window on "
                f"{hidden * 100:.0f}% of turns"
            )
        binding = self.stt_binding_fraction()
        if binding is not None and binding > 0:
            out.append(
                f"    STT was the binding constraint on {binding * 100:.0f}% of turns "
                f"(the turn waited on the transcript, not on the timer)"
            )
        ttfs = Distribution.of("stt_ttfs", (t.stt_ttfs_secs for t in self.measured))
        if ttfs.p99 is not None:
            out.append(
                f"    Measured TTFS p99 {_ms(ttfs.p99)} -> set BV_STT_TTFS_P99="
                f"{ttfs.p99:.2f} (turn strategy sizes its timer from this)"
            )

        tool = self.tool()
        if tool.n:
            out.append("")
            out.append(f"  Tool calls: {tool.n} turn(s) invoked a tool")
            out.append(f"    {_fmt_dist(tool)}")

        barge = self.barge_in()
        if barge.n:
            out.append("")
            out.append("  Barge-in (caller talks over bot -> bot audio stops)")
            out.append(f"    {_fmt_dist(barge)}")
        else:
            out.append("")
            out.append("  Barge-in: not exercised in this run")

        out.append(rule)
        return "\n".join(out)

    def stt_hidden_fraction(self) -> float | None:
        """Share of turns where STT cost the caller nothing.

        The single number that says whether a faster STT model would help at
        all: while this is 100%, it would not.
        """
        values = [
            t.stt_on_critical_path_secs
            for t in self.measured
            if t.stt_on_critical_path_secs is not None
        ]
        if not values:
            return None
        return sum(1 for v in values if v <= 0.001) / len(values)

    def stt_binding_fraction(self) -> float | None:
        """Share of turns whose end was set by STT rather than by the VAD config.

        While this is 0, shortening the endpointing window shortens TTFA. Once
        it climbs, the window has been cut below the decode time and the lever
        has quietly become STT speed instead.
        """
        values = [t.stt_is_binding for t in self.measured if t.stt_is_binding is not None]
        if not values:
            return None
        return sum(1 for v in values if v) / len(values)

    def to_markdown(self) -> str:
        """Markdown tables, for dropping into a README or a PR description."""
        lines: list[str] = [f"### Latency report — {self.label}", ""]
        if self.stack:
            lines.append("| component | configuration |")
            lines.append("| --- | --- |")
            for k, v in self.stack.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")
        measured = self.measured
        lines.append(
            f"Turns measured: **{len(measured)}** of {len(self.traces)} recorded."
        )
        lines.append("")
        if not measured:
            lines.append("_No first-audio measurement in this run._")
            return "\n".join(lines)

        lines.append("| metric | n | p50 | p90 | p95 | max |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for dist, label in (
            (self.ttfa(), "**TTFA** (caller-perceived)"),
            (self.ttfa_engineered(), "TTFA excl. endpointing"),
        ):
            lines.append(
                f"| {label} | {dist.n} | {_ms(dist.p50)} | {_ms(dist.p90)} "
                f"| {_ms(dist.p95)} | {_ms(dist.maximum)} |"
            )
        for dist, (_, label) in zip(self.stage_distributions(), STAGE_LABELS):
            lines.append(
                f"| {label} | {dist.n} | {_ms(dist.p50)} | {_ms(dist.p90)} "
                f"| {_ms(dist.p95)} | {_ms(dist.maximum)} |"
            )
        for dist, (_, label) in zip(self.concurrent_distributions(), CONCURRENT_LABELS):
            lines.append(
                f"| _{label.strip()}_ (concurrent) | {dist.n} | {_ms(dist.p50)} "
                f"| {_ms(dist.p90)} | {_ms(dist.p95)} | {_ms(dist.maximum)} |"
            )
        barge = self.barge_in()
        if barge.n:
            lines.append(
                f"| Barge-in stop | {barge.n} | {_ms(barge.p50)} | {_ms(barge.p90)} "
                f"| {_ms(barge.p95)} | {_ms(barge.maximum)} |"
            )
        return "\n".join(lines)


def _ms(v: float | None) -> str:
    return "-" if v is None else f"{v * 1000:.0f} ms"


def _fmt_dist(d: Distribution) -> str:
    if not d.n:
        return "no samples"
    return (
        f"n={d.n}  p50 {_ms(d.p50)}  p90 {_ms(d.p90)}  "
        f"p95 {_ms(d.p95)}  max {_ms(d.maximum)}"
    )


def load_traces(jsonl_path: str | Path) -> list[TurnTrace]:
    """Rebuild traces from a JSONL file written during a run."""
    traces: list[TurnTrace] = []
    for line in Path(jsonl_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        raw.pop("derived", None)
        moments = raw.pop("moments", [])
        trace = TurnTrace(**raw)
        trace.moments = [_moment(m) for m in moments]
        traces.append(trace)
    return traces


def _moment(m: dict[str, Any]):
    from bvagent.profiling.trace import Moment

    return Moment(**m)
