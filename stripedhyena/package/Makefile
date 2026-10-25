PY ?= python

.PHONY: help install test-cpu test-sim test-hw clean

help:
	@echo "install   pip install -e . --no-deps   (--no-deps protects the DLAMI torch stack)"
	@echo "test-cpu  CPU parity on a tiny random model (no Trainium)"
	@echo "test-sim  NKI cascade kernel via nki.simulate (no Trainium)"
	@echo "test-hw   on-device parity + generation (Trainium + torch-neuronx + weights)"

install:
	$(PY) -m pip install -e . --no-deps

test-cpu:
	$(PY) -m pytest tests -m cpu -q

test-sim:
	$(PY) -m pytest tests -m simulator -q

test-hw:
	$(PY) -m pytest tests -m hardware -q -s

clean:
	rm -rf build *.egg-info .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
