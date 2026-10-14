"""Tests for the latency arithmetic.

These guard the claims the report makes. A breakdown that silently
double-counts, or a TTFA that omits the endpointing wait, produces numbers that
look fine and are wrong — which is worse than a crash, because it gets
published.
"""

from __future__ import annotations

import pytest

from bvagent.profiling.report import Distribution, LatencyReport, percentile
from bvagent.profiling.trace import TurnTrace


def make_trace(**overrides) -> TurnTrace:
    """A turn with a plausible, internally consistent timeline."""
    base = dict(
        speech_end=100.0,
        transcript_at=100.30,
        user_stopped=100.90,
        llm_request=100.91,
        llm_first_token=101.25,
        tts_text_at=101.45,
        tts_first_audio=101.52,
        bot_speaking=101.55,
    )
    base.update(overrides)
    trace = TurnTrace(turn=1)
    for key, value in base.items():
        setattr(trace, key, value)
    return trace


class TestCriticalPath:
    def test_ttfa_spans_speech_end_to_first_audio(self):
        assert make_trace().ttfa_secs == pytest.approx(1.55)

    def test_ttfa_includes_the_endpointing_wait(self):
        # The caller sits through the endpointing silence, so it is part of
        # what they experience. Measuring from user_stopped instead would
        # flatter the result by the whole VAD window.
        trace = make_trace()
        assert trace.ttfa_secs > trace.ttfa_engineered_secs
        assert trace.ttfa_secs - trace.ttfa_engineered_secs == pytest.approx(
            trace.turn_end_wait_secs
        )

    def test_critical_path_stages_sum_to_ttfa(self):
        trace = make_trace()
        named = sum(v for v in trace.stages().values() if v is not None)
        assert named + trace.unattributed_secs == pytest.approx(trace.ttfa_secs)

    def test_stt_is_not_on_the_critical_path(self):
        # STT overlaps the turn-end wait. Including it in stages() would make
        # the parts sum to more than the measured whole.
        assert "stt" not in make_trace().stages()


class TestConcurrency:
    def test_stt_inside_the_window_costs_nothing(self):
        trace = make_trace(transcript_at=100.30, user_stopped=100.90)
        assert trace.stt_secs == pytest.approx(0.30)
        assert trace.stt_on_critical_path_secs == 0.0

    def test_stt_past_the_window_is_charged(self):
        # A slow decode that outlasts the endpointing wait does delay the turn.
        trace = make_trace(transcript_at=101.40, user_stopped=101.40)
        assert trace.stt_on_critical_path_secs == pytest.approx(0.0, abs=1e-9)
        slow = make_trace(transcript_at=101.50, user_stopped=101.20)
        assert slow.stt_on_critical_path_secs == pytest.approx(0.30)


class TestMissingData:
    def test_absent_moments_yield_none_not_zero(self):
        # A turn with no tool call must not report a 0ms tool call; zero is a
        # measurement, None is the absence of one.
        trace = make_trace()
        assert trace.tool_secs is None
        assert trace.barge_in_secs is None

    def test_out_of_order_frames_do_not_produce_negative_spans(self):
        trace = make_trace(bot_speaking=99.0)
        assert trace.ttfa_secs is None

    def test_incomplete_turn_is_not_complete(self):
        assert not make_trace(bot_speaking=None).complete


class TestBargeIn:
    def test_barge_in_measures_onset_to_silence(self):
        trace = make_trace()
        trace.barge_in_at = 103.0
        trace.barge_in_stopped = 103.22
        assert trace.barge_in_secs == pytest.approx(0.22)


class TestPercentiles:
    def test_interpolates_between_samples(self):
        assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)

    def test_single_sample(self):
        assert percentile([7], 95) == 7

    def test_empty(self):
        assert percentile([], 50) is None

    def test_distribution_ignores_missing(self):
        dist = Distribution.of("x", [1.0, None, 3.0])
        assert dist.n == 2
        assert dist.mean == pytest.approx(2.0)


class TestReport:
    def test_only_measured_turns_count(self):
        good = make_trace()
        bad = make_trace(bot_speaking=None)
        report = LatencyReport.build([good, bad], label="t")
        assert len(report.traces) == 2
        assert len(report.measured) == 1
        assert report.ttfa().n == 1

    def test_hidden_fraction_reports_when_stt_is_free(self):
        report = LatencyReport.build([make_trace(), make_trace()], label="t")
        assert report.stt_hidden_fraction() == 1.0

    def test_renders_without_samples(self):
        report = LatencyReport.build([], label="empty")
        assert "No turn produced" in report.to_text()

    def test_text_and_markdown_render(self):
        report = LatencyReport.build([make_trace()], label="t")
        text = report.to_text()
        assert "Time to first audio" in text
        assert "Turn-end wait" in text
        assert "| metric |" in report.to_markdown()


class TestTTFS:
    def test_ttfs_excludes_the_silence_vad_listened_through(self):
        # The turn strategy sizes its timer from decode-after-the-VAD-frame,
        # not decode-after-the-speech. Conflating them oversizes it by the
        # whole stop_secs window.
        trace = make_trace(transcript_at=100.70)
        trace.vad_stop_secs = 0.5
        assert trace.stt_secs == pytest.approx(0.70)
        assert trace.stt_ttfs_secs == pytest.approx(0.20)

    def test_ttfs_unknown_without_a_reported_stop_secs(self):
        assert make_trace().stt_ttfs_secs is None


class TestLLMQueue:
    def test_queue_span_is_named_not_unattributed(self):
        # A cold model load lands here. Before this stage existed it showed up
        # as 121s of "unattributed overhead" in a real run, which said nothing
        # about what had happened.
        trace = make_trace(user_stopped=100.90, llm_request=130.0)
        assert trace.llm_queue_secs == pytest.approx(29.1)
        assert "llm_queue" in trace.stages()

    def test_queue_is_normally_negligible(self):
        assert make_trace().llm_queue_secs == pytest.approx(0.01)


class TestSTTBinding:
    def test_stt_binding_when_turn_waits_on_the_transcript(self):
        # turn-end and STT land together: the turn closed as soon as the
        # transcript did, so stop_secs was not what set the wait.
        trace = make_trace(transcript_at=100.73, user_stopped=100.74)
        assert trace.stt_is_binding is True

    def test_not_binding_when_the_timer_is_longer(self):
        trace = make_trace(transcript_at=100.30, user_stopped=100.90)
        assert trace.stt_is_binding is False
