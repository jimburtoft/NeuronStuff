import pytest

def pytest_collection_modifyitems(config, items):
    """Skip hardware tests unless a Neuron device is actually present."""
    have_hw = False
    try:
        import torch
        have_hw = hasattr(torch, "neuron") or "neuron" in str(getattr(torch, "__config__", ""))
        import os
        have_hw = have_hw or os.path.exists("/dev/neuron0")
    except Exception:
        pass
    have_container = False
    try:
        import torch_neuronx  # noqa: F401
        have_container = True
    except Exception:
        pass

    skip_dev = pytest.mark.skip(
        reason="no Trainium device (/dev/neuron0); simulator tests still run")
    skip_ctr = pytest.mark.skip(
        reason="torch_neuronx not importable (needs the AWS PyTorch Native container)")

    for item in items:
        # `container` = needs torch_neuronx but no device (e.g. patch round-trip tests)
        if "container" in item.keywords and not have_container:
            item.add_marker(skip_ctr)
        elif "hardware" in item.keywords and not have_hw:
            item.add_marker(skip_dev)
