"""Tests for the Bangla-specific handling and the tool backend."""

from __future__ import annotations

import pytest

from bvagent.agent.backend import Backend, from_bengali_digits, to_bengali_digits
from bvagent.agent.tools import build_tools
from bvagent.config import TTSConfig
from bvagent.services.mms_tts import MMSTTSService


class TestBengaliNumerals:
    def test_round_trip(self):
        assert to_bengali_digits(4421) == "৪৪২১"
        assert from_bengali_digits("৪৪২১") == "4421"

    def test_mixed_input_normalizes(self):
        # Whisper and the LLM both produce either script, sometimes in the
        # same utterance, so lookups have to accept both.
        assert from_bengali_digits("৪৪2১") == "4421"

    def test_non_digits_survive(self):
        assert to_bengali_digits("BR-12") == "BR-১২"


class TestBackend:
    async def test_lookup_accepts_either_script(self):
        backend = Backend(latency_secs=0)
        ascii_result = await backend.get_loan_status("4421")
        bengali_result = await backend.get_loan_status("৪৪২১")
        assert ascii_result == bengali_result
        assert ascii_result["found"]

    async def test_amounts_come_back_in_bengali_numerals(self):
        # Results are read aloud; ASCII digits make the TTS switch phoneme
        # sets mid-sentence, which is audible.
        result = await Backend(latency_secs=0).get_loan_status("4421")
        assert result["installment_bdt"] == "৫০০০"
        assert not any(c.isdigit() and c.isascii() for c in result["due_date"])

    async def test_unknown_account_is_reported_not_invented(self):
        result = await Backend(latency_secs=0).get_loan_status("0000")
        assert result["found"] is False

    async def test_overdue_loan_flagged(self):
        result = await Backend(latency_secs=0).get_loan_status("7813")
        assert result["is_overdue"] is True

    async def test_slots_are_deterministic(self):
        # A benchmark replays the same call; a backend that answered
        # differently each run would change the reply length and therefore
        # the synthesis cost being measured.
        backend = Backend(latency_secs=0)
        first = await backend.find_available_slots("গুলশান")
        second = await backend.find_available_slots("গুলশান")
        assert first == second


class TestTools:
    def test_scenario_advertises_only_its_own_tools(self):
        # Unused schemas still cost prefill time on every turn.
        loan = {s.name for s in build_tools(Backend(), "loan").standard_tools}
        booking = {s.name for s in build_tools(Backend(), "booking").standard_tools}
        assert loan == {"get_loan_status", "record_payment_promise"}
        assert not loan & booking

    def test_unknown_scenario_rejected(self):
        with pytest.raises(ValueError):
            build_tools(Backend(), "nonsense")


class TestTTSChunking:
    def chunker(self, limit: int) -> MMSTTSService:
        service = MMSTTSService.__new__(MMSTTSService)
        service._config = TTSConfig(max_chars_per_chunk=limit)
        return service

    def test_short_text_is_one_pass(self):
        assert self.chunker(160)._chunks("আসসালামু আলাইকুম") == ["আসসালামু আলাইকুম"]

    def test_splits_at_the_danda(self):
        # Bangla ends sentences with ।, not a full stop.
        text = "আপনার কিস্তি দশ তারিখে। মোট বকেয়া পাঁচ হাজার টাকা।"
        chunks = self.chunker(30)._chunks(text)
        assert len(chunks) > 1
        assert chunks[0].endswith("।")

    def test_chunks_reassemble_to_the_original(self):
        text = "আপনার কিস্তি দশ তারিখে। মোট বকেয়া পাঁচ হাজার টাকা। ধন্যবাদ।"
        assert " ".join(self.chunker(30)._chunks(text)) == text

    def test_text_without_boundaries_is_not_split_mid_word(self):
        text = "একটিঅতিদীর্ঘশব্দযাভাঙাযাবেনা"
        assert self.chunker(5)._chunks(text) == [text]

    def test_empty_text_yields_nothing(self):
        assert self.chunker(160)._chunks("   ") == []
