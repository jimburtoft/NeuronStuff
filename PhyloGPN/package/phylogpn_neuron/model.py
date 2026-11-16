"""PhyloGPN (songlab/PhyloGPN) in plain PyTorch, for TorchNeuron on Trainium2.

Same math as the Hugging Face model (`modeling_phylogpn.py`):
- weights: the Hub checkpoint's reverse-complement weight symmetrization is applied once at load
  time, instead of on every forward (`torch.nn.utils.parametrize` in the Hub code); exact.
- graph: "same"-padded dilated convolutions with an uncropped residual, plus one crop of
  RECEPTIVE_FIELD // 2 positions per side before the output head. The central outputs are identical
  to the Hub model's valid convolutions with a per-block crop. Each block's zero-padding edge
  effect grows by its own trim, and the final crop removes exactly the sum.
- GELU is the explicit fp32 erf form (exact GELU), which TorchNeuron lowers to one native activation.
"""
import torch
import torch.nn.functional as F
from torch import nn

VOCAB = "ACGTN-"
PAD_ID = 5
UNK_ID = 4
RECEPTIVE_FIELD = 481
HALF = RECEPTIVE_FIELD // 2  # 240


def gelu(x):
    y = x.float()
    return (0.5 * y * (1.0 + torch.erf(y * 0.7071067811865476))).to(x.dtype)


class ChannelLayerNorm(nn.LayerNorm):
    """LayerNorm over the channel axis of a [B, C, L] tensor."""

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, outer=960, inner=480, kernel_size=5, dilation=1):
        super().__init__()
        self.layer_norm1 = ChannelLayerNorm(outer)
        self.conv1 = nn.Conv1d(outer, inner, 1)
        self.layer_norm2 = ChannelLayerNorm(inner)
        self.conv2 = nn.Conv1d(inner, inner, kernel_size, dilation=dilation)
        self.layer_norm3 = ChannelLayerNorm(inner)
        self.conv3 = nn.Conv1d(inner, outer, 1)
        self.dilation = dilation
        self.trim = (kernel_size - 1) * dilation // 2
        self.kernel_op = None  # set by PhyloGPN.use_kernel()

    def forward(self, x):
        if self.kernel_op is not None:
            return self.kernel_op(x, self.kw1, self.kw2, self.kw3, self.kvecs)
        h = self.conv1(gelu(self.layer_norm1(x)))
        h = gelu(self.layer_norm2(h))
        h = F.conv1d(h, self.conv2.weight, self.conv2.bias, padding=self.trim, dilation=self.dilation)
        h = self.conv3(gelu(self.layer_norm3(h)))
        return x + h


class PhyloGPN(nn.Module):
    """forward(ids [B, L], L >= 481) -> (embeddings [B, L-480, 960], log-rates [B, L-480, 4] in ACGT order)."""

    def __init__(self, outer=960, inner=480, kernel_size=5, stack_size=2, num_stacks=20):
        super().__init__()
        dilations = num_stacks * [kernel_size ** i for i in range(stack_size)]
        self.embedding = nn.Embedding(6, outer, padding_idx=PAD_ID)
        self.blocks = nn.Sequential(*[Block(outer, inner, kernel_size, d) for d in dilations])
        self.output_layer_norm = nn.LayerNorm(outer)
        self.output_conv = nn.Conv1d(outer, 4, 1)
        self.crop = sum(b.trim for b in self.blocks)
        assert self.crop == HALF

    def forward(self, ids):
        x = self.embedding(ids).transpose(1, 2)
        x = self.blocks(x)
        x = x[:, :, self.crop: x.shape[2] - self.crop]
        e = self.output_layer_norm(x.transpose(1, 2))
        logits = self.output_conv(gelu(e).transpose(1, 2)).transpose(1, 2)
        return e, logits

    @torch.no_grad()
    def center_(self):
        """Centre the embedding rows and every conv's output channels. Exact (each stream is only
        read through a LayerNorm, which ignores a per-position shift); required by the NKI kernel."""
        self.embedding.weight -= self.embedding.weight.mean(1, keepdim=True)
        for b in self.blocks:
            for c in (b.conv1, b.conv2, b.conv3):
                c.weight -= c.weight.mean(0, keepdim=True)
                c.bias -= c.bias.mean()
        self._centred = True
        return self

    def use_kernel(self, cfg=None):
        """Replace every block's forward by the fused NKI kernel (call after .to(device)).

        Inference only: the kernel has no backward, and calling backward through it raises a RuntimeError.
        Run under torch.no_grad(); for training, leave the kernel off (stock graph, full autograd)."""
        from . import kernel
        assert getattr(self, "_centred", False), "call center_() (in fp32, before casting) first"
        ops = kernel.register_torch_ops(cfg)
        for b in self.blocks:
            dev, dt = b.conv1.weight.device, b.conv1.weight.dtype
            cpu = Block(b.conv1.in_channels, b.conv1.out_channels, b.conv2.kernel_size[0], b.dilation)
            cpu.load_state_dict({k: v.detach().cpu().float() for k, v in b.state_dict().items()})
            w1, w2, w3, vecs = kernel.prepare_block_weights(cpu)
            for n, t in (("kw1", w1.to(dt)), ("kw2", w2.to(dt)), ("kw3", w3.to(dt)), ("kvecs", vecs.float())):
                b.register_buffer(n, t.to(dev), persistent=False)
            b.kernel_op = ops[b.dilation]
        return self
