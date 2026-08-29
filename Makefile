# Thin wrapper over uv, matching the convention used elsewhere in this workspace.


# Run from Git Bash on Windows.

PY := .venv/Scripts/python.exe

.PHONY: help install test lint fmt quick experiment gallery clean

help:            ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:         ## create the venv and install everything
	uv sync --extra dev

test:            ## run the test suite
	$(PY) -m pytest -q

lint:            ## ruff check + format check
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

fmt:             ## apply ruff formatting
	$(PY) -m ruff format src tests

quick:           ## ~3 min smoke run; numbers are NOT meaningful
	$(PY) -m pinnbp.cli experiment --config configs/quick.yaml --tag smoke

experiment:      ## the real run; regenerates docs/RESULTS.md (~30 min on this machine)
	$(PY) -m pinnbp.cli experiment --config configs/experiment.yaml

gallery:         ## plot one window under each artifact mechanism
	$(PY) -m pinnbp.cli gallery

clean:           ## drop the derived data cache
	rm -rf data/cache
