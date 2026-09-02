"""Per-sample LoRA gradient extraction.

WHY THIS IS NOT JUST `loss.backward()`
--------------------------------------
Influence needs a *separate* gradient per training sample. Autograd gives the
sum over the batch, so the naive approach is batch-size-1 backward per sample —
correct but slow, and it still leaves a storage problem: one Qwen2.5-14B LoRA
gradient is 275,251,200 floats (~550MB fp16), so 10k samples is ~5.5PB.

Both problems are solved by the same structural fact. For a LoRA module
    h = W x + (alpha/r) * B A x
with A: (r, d_in), B: (d_out, r), write z = A x and let g = dL/d(B z). Then for
a single token
    grad_B = g z^T          (outer product, rank 1)
    grad_A = (B^T g) x^T    (outer product, rank 1)
and a sample's gradient is the sum of these over its supervised tokens. So we
never materialise a full gradient: capturing (x, g) per module per token is
enough to reconstruct or project it. This is the structure LoGra exploits
(Choe et al. 2024).

We therefore run ONE backward over a batch while capturing per-token (x, g) via
hooks, then split by sample. That gives true per-sample gradients from a batched
backward — no batch-size-1 loop.

SCALE NOTE: `alpha/r` scaling is applied so the captured gradients match what an
optimizer would see. Getting this wrong rescales every influence score by a
constant, which is invisible in rank correlations but NOT in the magnitude-based
analyses (Gini, top-k mass) that H2 depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class LoRAModuleRef:
    """One LoRA-adapted projection, with the tensors needed to rebuild grads."""

    name: str
    lora_A: nn.Module           # (r, d_in)
    lora_B: nn.Module           # (d_out, r)
    scaling: float              # alpha / r

    @property
    def r(self) -> int:
        return self.lora_A.weight.shape[0]

    @property
    def d_in(self) -> int:
        return self.lora_A.weight.shape[1]

    @property
    def d_out(self) -> int:
        return self.lora_B.weight.shape[0]

    @property
    def n_params(self) -> int:
        return self.lora_A.weight.numel() + self.lora_B.weight.numel()


def find_lora_modules(model, adapter_name: str = "default") -> list[LoRAModuleRef]:
    """Discover PEFT LoRA modules, in a deterministic order.

    Order matters: it fixes the layout of the concatenated gradient vector, and
    therefore whether two runs' projections are comparable at all. Sorted by
    module name so it cannot drift with model internals.
    """
    refs: list[LoRAModuleRef] = []
    for name, mod in model.named_modules():
        if not (hasattr(mod, "lora_A") and hasattr(mod, "lora_B")):
            continue
        try:
            a = mod.lora_A[adapter_name]
            b = mod.lora_B[adapter_name]
        except (KeyError, TypeError):
            continue
        scaling = float(getattr(mod, "scaling", {}).get(adapter_name, 1.0)) \
            if isinstance(getattr(mod, "scaling", None), dict) \
            else float(getattr(mod, "scaling", 1.0))
        refs.append(LoRAModuleRef(name=name, lora_A=a, lora_B=b, scaling=scaling))

    refs.sort(key=lambda x: x.name)
    if not refs:
        raise RuntimeError(
            "No LoRA modules found. Is the adapter loaded, and is adapter_name "
            f"correct (got {adapter_name!r})?"
        )
    return refs


@dataclass
class _Capture:
    """Per-module activation/grad buffers for one backward pass."""

    a_in: dict[str, torch.Tensor] = field(default_factory=dict)   # x into lora_A
    b_gout: dict[str, torch.Tensor] = field(default_factory=dict)  # dL/d(lora_B out)


class LoRAGradientCapture:
    """Context manager capturing per-token (x, g) for every LoRA module.

    Usage:
        with LoRAGradientCapture(refs) as cap:
            loss.backward()
        grads = cap.per_sample_grads(sample_index)

    The captured tensors are (batch, seq, dim); splitting by the batch axis is
    what makes per-sample gradients recoverable from one batched backward.
    """

    def __init__(self, refs: list[LoRAModuleRef], keep_positions=None):
        """
        keep_positions: optional 1-D LongTensor of token indices to retain.

        MEMORY, NOT AN OPTIMISATION DETAIL — at Qwen2.5-32B the hooks hold every
        module's (x, g) simultaneously:

            ~500-token finetuning sample ->   8.4 GB
            ~3000-token query prefix     ->  50.3 GB   <- OOMs on 2xH100
            ~250-token query span only   ->   4.2 GB

        Restricting to supervised positions is EXACT, not approximate: masked
        positions carry grad_output = 0, so they contribute nothing to
        grad_A / grad_B. For queries we supervise only the misaligned action
        span, which is short, so this is what makes 32B queries feasible at all.
        """
        self.refs = refs
        self.cap = _Capture()
        self._handles: list = []
        self.keep = keep_positions

    def __enter__(self) -> LoRAGradientCapture:
        for ref in self.refs:
            self._handles.append(
                ref.lora_A.register_forward_hook(self._make_fwd_hook(ref.name))
            )
            self._handles.append(
                ref.lora_B.register_full_backward_hook(self._make_bwd_hook(ref.name))
            )
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def _slice(self, t):
        """Keep only supervised positions along the sequence axis."""
        if self.keep is None or t.dim() < 2:
            return t
        return t.index_select(1, self.keep.to(t.device))

    def _make_fwd_hook(self, name: str):
        def hook(_mod, inputs, _out):
            # inputs[0] is x entering lora_A: (batch, seq, d_in)
            self.cap.a_in[name] = self._slice(inputs[0].detach())
        return hook

    def _make_bwd_hook(self, name: str):
        def hook(_mod, _grad_in, grad_out):
            # grad_out[0] is dL/d(lora_B output): (batch, seq, d_out)
            self.cap.b_gout[name] = self._slice(grad_out[0].detach())
        return hook

    def per_sample_grads(self, i: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Reconstruct (grad_A, grad_B) for batch element `i`, per module.

        Returned in the same units an optimizer would see, i.e. including the
        alpha/r scaling.
        """
        out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for ref in self.refs:
            x = self.cap.a_in[ref.name][i]           # (seq, d_in)
            g = self.cap.b_gout[ref.name][i]         # (seq, d_out)
            A = ref.lora_A.weight                    # (r, d_in)
            B = ref.lora_B.weight                    # (d_out, r)

            z = x @ A.T.to(x.dtype)                  # (seq, r)
            # grad_B = sum_t g_t z_t^T ; grad_A = sum_t (B^T g_t) x_t^T
            grad_B = g.T.to(torch.float32) @ z.to(torch.float32)          # (d_out, r)
            gb = g.to(torch.float32) @ B.to(torch.float32)                # (seq, r)
            grad_A = gb.T @ x.to(torch.float32)                           # (r, d_in)

            # NO extra alpha/r factor here. PEFT computes
            #     out = lora_B(lora_A(x)) * scaling
            # so the backward hook's grad_output is dL/d(lora_B out), which
            # ALREADY carries the scaling. Multiplying again inflated grad_B by
            # exactly `scaling` (caught by test_matches_autograd). `ref.scaling`
            # is kept for reference but must not be applied.
            out[ref.name] = (grad_A, grad_B)
        return out

    def flat_per_sample_grad(self, i: int) -> torch.Tensor:
        """Concatenate a sample's gradient into one vector (module order fixed).

        Only for tests and small models — at 14B this is ~275M floats. Real
        pipelines project instead (see projection.py).
        """
        parts: list[torch.Tensor] = []
        grads = self.per_sample_grads(i)
        for ref in self.refs:
            gA, gB = grads[ref.name]
            parts.append(gA.reshape(-1))
            parts.append(gB.reshape(-1))
        return torch.cat(parts)
