"""Scripted Bangla caller turns for reproducible benchmarking.

The caller audio is synthesized with the same MMS voice the agent speaks with.
That is a deliberate trade and it has one important consequence: these runs
measure **pipeline latency**, not transcription accuracy. Synthetic speech is
cleaner than a phone line, so the WER here is optimistic and no accuracy claim
should be made from it. What it buys is the thing a latency benchmark needs —
byte-identical input on every run, so a change in the numbers is a change in
the system rather than in how someone happened to say a sentence.

For accuracy work, drop real recordings into ``data/audio`` and point the
harness at them; the transport reads WAVs either way.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

# A loan-collection call that exercises both tools and ends naturally.
LOAN_TURNS: list[tuple[str, str]] = [
    ("greeting", "আসসালামু আলাইকুম, জি বলুন।"),
    ("account", "আমার অ্যাকাউন্ট নম্বর চার চার দুই এক।"),
    ("when_due", "আচ্ছা, কত টাকা দিতে হবে আর কবের মধ্যে?"),
    ("promise", "আমি আগামী রবিবার টাকা জমা দিয়ে দেব।"),
    ("thanks", "ধন্যবাদ, রাখছি তাহলে।"),
]

# A branch-appointment call.
BOOKING_TURNS: list[tuple[str, str]] = [
    ("branch", "আমি গুলশান শাখায় আসতে চাই।"),
    ("choose", "প্রথম সময়টাই আমার জন্য ঠিক আছে।"),
    ("name", "আমার নাম রহিম উদ্দিন।"),
    ("thanks", "ধন্যবাদ আপনাকে।"),
]

# Spoken over the bot mid-reply, to measure how fast it stops.
BARGE_IN_TURN: tuple[str, str] = (
    "barge_in",
    "এক মিনিট, আমি একটু পরে বলছি।",
)

SCENARIO_TURNS = {"loan": LOAN_TURNS, "booking": BOOKING_TURNS}


@dataclass
class CorpusEntry:
    name: str
    text: str
    path: Path
    audio: np.ndarray


class Corpus:
    """Synthesizes and caches the caller's side of a scripted call."""

    def __init__(self, directory: str | Path = "data/audio", sample_rate: int = 16000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self._tts = None

    def _filename(self, name: str, text: str) -> Path:
        # Hash the text so edited lines regenerate instead of silently serving
        # yesterday's audio under today's label.
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        return self.directory / f"caller_{name}_{digest}.wav"

    def _load_tts(self):
        if self._tts is None:
            import torch
            from transformers import AutoTokenizer, VitsModel

            logger.info("loading MMS-TTS to synthesize the caller corpus")
            tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
            model = VitsModel.from_pretrained("facebook/mms-tts-ben").eval()
            self._tts = (tokenizer, model, torch)
        return self._tts

    def get(self, name: str, text: str) -> CorpusEntry:
        import soundfile as sf

        path = self._filename(name, text)
        if path.exists():
            audio, rate = sf.read(str(path), dtype="float32")
            if rate == self.sample_rate:
                return CorpusEntry(name, text, path, audio)

        tokenizer, model, torch = self._load_tts()
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            waveform = model(**inputs).waveform
        audio = waveform.squeeze().detach().float().cpu().numpy()

        native = int(model.config.sampling_rate)
        if native != self.sample_rate:
            dst_len = max(1, int(round(audio.size / native * self.sample_rate)))
            audio = np.interp(
                np.linspace(0, audio.size - 1, dst_len),
                np.linspace(0, audio.size - 1, audio.size),
                audio,
            ).astype(np.float32)

        sf.write(str(path), audio, self.sample_rate)
        logger.info(f"synthesized {path.name} ({audio.size / self.sample_rate:.2f}s)")
        return CorpusEntry(name, text, path, audio)

    def scenario(self, scenario: str) -> list[CorpusEntry]:
        turns = SCENARIO_TURNS.get(scenario)
        if turns is None:
            raise ValueError(f"unknown scenario {scenario!r}")
        return [self.get(name, text) for name, text in turns]

    def barge_in(self) -> CorpusEntry:
        return self.get(*BARGE_IN_TURN)
