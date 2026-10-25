"""Replace step_fir's `roll + in-place scatter` state update with `concat`. EAGER-ONLY.

WHY: step_fir is the ONLY component that fails to batch --
    step_fir  BS=1 1.804 -> BS=4 5.103 ms/call = 2.83x  (ideal 1.00x)
    step_iir  BS=1 0.882 -> BS=4 0.966 ms/call = 1.10x  (flat)
Both hold (B,D,k) state and reduce over the last dim. The difference is that step_fir
also performs LAYOUT work: torch.roll + in-place scatter. An earlier profile attributed
12,225 DMA_DIRECT2D instructions to `fir_state[..., -1] = u` (engine.py:447) at BS=1.

SCREEN (bench/ring_screen.py, sync-bracketed, eager):
                              BS=1 k=3   BS=1 k=128  BS=4 k=3   BS=4 k=128
    upstream roll+scatter       0.782      0.792      1.676      2.028
    concat                      0.681      0.765      0.808      1.848
    speedup                     1.15x      1.04x      2.07x      1.10x
At BS=4 the short-filter (k=3) update DOUBLES under roll+scatter but stays nearly FLAT
under concat. concat is specifically the batch-scaling fix.

⚠ WHY THIS CONTRADICTS AN EARLIER SCREEN, WHICH REJECTED CONCAT AT 24x WORSE:
That screen measured under torch.compile. A later measurement showed torch.compile is itself an
8-119x pessimization on this model and we now ship EAGER. Different regime, opposite
result. A rejected optimization must be re-screened when the execution mode changes.

A ring buffer was also screened and REJECTED: 118-2444 ms, i.e. 100-1200x worse.
Single-column writes are pathological here -- the same defect as the scatter it replaces.

MATH: roll-by-one-then-overwrite-last == drop-first-then-append. Exactly equivalent;
upstream already uses `torch.cat` in the `cache_size < filter_length - 1` branch.
"""
import torch

_orig = None


def install():
    global _orig
    from vortex.model.engine import HyenaInferenceEngine as H
    _orig = H.step_fir

    def step_fir(self, u, fir_state, weight, bias=None, gated_bias=False, flip_filter=False):
        weight = weight.squeeze()
        cache_size = fir_state.shape[-1]
        filter_length = weight.shape[-1]
        if flip_filter:
            weight = weight.flip(-1)[..., -cache_size - 1:].unsqueeze(0)
        else:
            weight = weight[..., : cache_size + 1].unsqueeze(0)

        in_dtype = u.dtype
        weight = weight if weight.dtype == torch.float32 else weight.to(torch.float32)
        uf = u if u.dtype == torch.float32 else u.to(torch.float32)
        sf = fir_state if fir_state.dtype == torch.float32 else fir_state.to(torch.float32)

        h0, h = weight[..., -1], weight[..., :-1]
        y = h0 * uf + torch.sum(sf * h, dim=-1)
        if bias is not None:
            b = bias if bias.dtype == torch.float32 else bias.to(torch.float32)
            y = y + (b * uf if gated_bias else b)

        # --- BS-GATED: concat wins at BS>1, LOSES at BS=1 ---
        # Measured (eager, paired, Welch):
        #   BS=1  160.4 -> 164.6 ms  +4.44 sigma  SLOWER (0.975x)
        #   BS=4  322.9 -> 311.6 ms -13.59 sigma  FASTER (1.036x)
        # At BS=1 the extra allocation costs more than the roll it removes; the win is
        # purely from batch scaling (screen: 1.15x at BS=1 k=3 vs 2.07x at BS=4 k=3).
        if cache_size < filter_length - 1:
            fir_state = torch.cat([fir_state, u[..., None]], dim=-1)   # upstream path
        elif fir_state.shape[0] > 1:
            fir_state = torch.cat([fir_state[..., 1:], u[..., None]], dim=-1)
        else:
            fir_state = torch.roll(fir_state, -1, dims=2)              # BS=1: upstream is faster
            fir_state[..., -1] = u

        return (y if y.dtype == in_dtype else y.to(in_dtype)), fir_state

    H.step_fir = step_fir


def uninstall():
    if _orig is not None:
        from vortex.model.engine import HyenaInferenceEngine as H
        H.step_fir = _orig
