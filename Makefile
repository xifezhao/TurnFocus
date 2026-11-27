PYTHON ?= python3

.PHONY: help install-cpu install-gpu test smoke dry-run verify-archive checksums anon-zip

help:
	@echo "TurnFocus Artifact Makefile Targets:"
	@echo "  make install-cpu     Install minimal CPU dependencies + editable package"
	@echo "  make install-gpu     Install full CUDA/HuggingFace dependencies + editable package"
	@echo "  make test            Run all 13 mathematical & functional unit tests"
	@echo "  make smoke           Run <60s CPU smoke check (scripts/run_smoke_test.sh)"
	@echo "  make dry-run         Run zero-download CPU end-to-end pipeline dry-run + archive diff"
	@echo "  make verify-archive  Verify SHA256SUMS.txt and archived JSON/Markdown integrity"
	@echo "  make checksums       Regenerate SHA256SUMS.txt"
	@echo "  make anon-zip        Build stripped anonymous archive for double-blind review"

install-cpu:
	$(PYTHON) -m pip install -r requirements-cpu.txt
	$(PYTHON) -m pip install --no-deps -e .

install-gpu:
	$(PYTHON) -m pip install -r requirements-colab.txt
	$(PYTHON) -m pip install --no-deps -e .

test:
	$(PYTHON) -m pytest tests/ -q

smoke:
	PYTHON_BIN=$(PYTHON) bash scripts/run_smoke_test.sh

dry-run:
	$(PYTHON) scripts/run_colab_ae.py --dry-run --max-samples 6 --seeds 101,202 --output-dir /tmp/turnfocus_dry_run
	$(PYTHON) scripts/verify_archive.py --skip-checksums --compare-dir /tmp/turnfocus_dry_run

verify-archive:
	$(PYTHON) scripts/verify_archive.py

checksums:
	$(PYTHON) scripts/generate_checksums.py

anon-zip:
	bash scripts/export_anonymous_snapshot.sh
