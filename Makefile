VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: help setup test serve bench sweep clean

help:
	@echo "setup   install into .venv and prefetch models"
	@echo "test    run the test suite"
	@echo "serve   run the live agent on ws://localhost:8765"
	@echo "bench   replay a scripted call and print a latency report"
	@echo "sweep   compare configurations (DIM=vad|stt|stt_device|tts_chunk|thinking)"

setup:
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PY) -e .
	uv pip install --python $(PY) torch --torch-backend=auto
	$(PY) -m nltk.downloader punkt_tab
	$(PY) -c "from faster_whisper import WhisperModel; WhisperModel('large-v3')"
	$(PY) -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/mms-tts-ben')"

test:
	$(PY) -m pytest tests/ -q

serve:
	$(PY) -m bvagent.bot --scenario $(or $(SCENARIO),loan)

bench:
	$(PY) -m bvagent.bench.run --scenario $(or $(SCENARIO),loan) --calls $(or $(CALLS),1) --markdown

sweep:
	$(PY) -m bvagent.bench.sweep --dimension $(or $(DIM),vad) --calls $(or $(CALLS),1)

clean:
	rm -rf runs/ .pytest_cache **/__pycache__
