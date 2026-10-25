# Simulator tests need no Trainium; hardware tests do.
PY ?= python

.PHONY: help install test test-sim test-hw lint clean

help:
	@echo "install   pip install -e . --no-deps   (--no-deps protects the container stack)"
	@echo "test-sim  CPU-only kernel parity via nki.simulate  (no Trainium needed)"
	@echo "test-hw   on-device parity + a decode smoke test   (needs Trainium)"
	@echo "test      test-sim + test-hw"

install:
	$(PY) -m pip install -e . --no-deps

test-sim:
	$(PY) -m pytest tests -m simulator -q

test-hw:
	$(PY) -m pytest tests -m hardware -q

test: test-sim test-hw

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

test-container:
	$(PY) -m pytest tests -m container -q
