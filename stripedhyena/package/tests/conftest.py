import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


def pytest_collection_modifyitems(config, items):
    have_dev = os.path.exists("/dev/neuron0") and _have("torch_neuronx")
    have_w = os.path.isdir(os.environ.get("SHN_WEIGHTS", os.path.expanduser("~/weights/hessian-7b")))
    have_nki = _have("nki")
    have_torch = _have("torch")
    for it in items:
        if "hardware" in it.keywords and not have_dev:
            it.add_marker(pytest.mark.skip(reason="needs a Trainium device + torch_neuronx (PyTorch Native: SDK 2.33 DLAMI venv)"))
        elif "weights" in it.keywords and not have_w:
            it.add_marker(pytest.mark.skip(reason="checkpoint not found (set SHN_WEIGHTS)"))
        elif "simulator" in it.keywords and not have_nki:
            it.add_marker(pytest.mark.skip(reason="nki not installed (pip install nki)"))
        elif "cpu" in it.keywords and not have_torch:
            it.add_marker(pytest.mark.skip(reason="torch not installed"))
