# Integration Guide

## Quick Integration with Existing PIRL Project

### Step 1: Copy Files to Your PIRL Project

```bash
# Navigate to your existing PIRL project
cd /path/to/your/pirl_itsc2024

# Copy the Neuron-optimized agent
cp /path/to/pirl-neuron-optimization/rl_agent/PIRL_torch_neuron.py pirl_carla/rl_agent/

# Copy test files (optional)
cp /path/to/pirl-neuron-optimization/tests/test_pirl_minimal.py .
cp /path/to/pirl-neuron-optimization/tests/run_neuron_tests.sh .
```

### Step 2: Minimal Code Changes

In your existing training script (e.g., `training_pirl_Town2.py`):

```python
# Change this line:
# from rl_agent.PIRL_torch import PIRLagent, agentOptions, pinnOptions

# To this:
from rl_agent.PIRL_torch_neuron import PIRLagentNeuron, agentOptions, pinnOptions

# Change agent creation:
# agent = PIRLagent(model, actNum, agentOp, pinnOp)

# To this:
agent = PIRLagentNeuron(model, actNum, agentOp, pinnOp, trace_model=True)
```

### Step 3: Test Compatibility

```bash
python test_pirl_minimal.py
```

### Step 4: Production Deployment

For production inference, use batch processing:

```python
# Collect multiple states for batch processing
states_batch = []
for _ in range(512):  # Optimal batch size
    state = env.get_state()
    states_batch.append(state)

# Batch inference
states_array = np.array(states_batch)
q_values_batch = agent.get_qs_batch(states_array)

# Process results
for i, q_values in enumerate(q_values_batch):
    action = torch.argmax(q_values).item()
    # Use action for state i
```

## Performance Optimization Tips

1. **Use large batch sizes** (512, 1024, 2048) for maximum throughput
2. **Pre-warm the models** by running a few inference calls before production
3. **Monitor batch size distribution** in your application
4. **Consider hybrid deployment**: CPU for training, Neuron for inference

## Backward Compatibility

The Neuron-optimized agent is a drop-in replacement:
- ✅ Same API as original PIRLagent
- ✅ Works on CPU when Neuron unavailable
- ✅ No changes needed to existing training loops
- ✅ Compatible with existing checkpoints