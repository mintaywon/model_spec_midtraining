"""Random projection of per-sample LoRA gradients.

WHY PROJECTION IS MANDATORY
---------------------------
One Qwen2.5-14B LoRA gradient is 275,251,200 floats = ~550MB in fp16. Storing
10k AFT samples raw would be ~5.5PB. Projecting to 2^15 = 32,768 dims gives
64KB/sample -> ~640MB for 10k. That is the difference between "impossible" and
"a file".

WHY NOT A DENSE PROJECTION MATRIX
---------------------------------
A dense (275M x 32k) Gaussian matrix is itself ~18TB. Instead we exploit the
same rank-1 structure gradients.py uses. For a LoRA module the gradient is

    grad_B = G^T Z   (d_out x r),   grad_A = (G B)^T X   (r x d_in)

so a Johnson-Lindenstrauss projection factorises per module:

    <P, grad_B> = <P_left^T P_right, G^T Z>   with small per-side sketches.

Concretely we sketch each side independently — left factor to `k_left`, right
factor to `k_right` — and take the outer product, giving k_left*k_right dims per
module per matrix. This is the LoGra construction (Choe et al. 2024). The
per-module seeds are derived from one master seed, so the projection is exactly
reproducible across runs, machines, and checkpoints.

REPRODUCIBILITY IS LOAD-BEARING: H1 compares influence profiles ACROSS
checkpoints. If two runs used different projections, the resulting scores are
not comparable and the rank correlation is meaningless. `seed` is therefore part
of the stored metadata, and `ProjectionSpec.fingerprint()` must match between
any two runs whose scores are compared.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from tda.influence.gradients import LoRAModuleRef


@dataclass(frozen=True)
class ProjectionSpec:
    """Everything needed to reproduce a projection exactly."""

    seed: int
    k_left: int = 16
    k_right: int = 16
    dtype: str = "float16"

    def dim_per_matrix(self) -> int:
        return self.k_left * self.k_right

    def total_dim(self, refs: list[LoRAModuleRef]) -> int:
        # Two matrices (A and B) per module.
        return 2 * len(refs) * self.dim_per_matrix()

    def fingerprint(self, refs: list[LoRAModuleRef]) -> str:
        """Stable id of (spec, module layout). Scores are comparable iff equal."""
        h = hashlib.sha256()
        h.update(f"{self.seed}|{self.k_left}|{self.k_right}|{self.dtype}".encode())
        for r in refs:
            h.update(f"|{r.name}:{r.r}x{r.d_in}:{r.d_out}x{r.r}".encode())
        return h.hexdigest()[:16]


def _sketch(name: str, kind: str, rows: int, cols: int, spec: ProjectionSpec,
            device, dtype) -> torch.Tensor:
    """Deterministic (cols x rows) Gaussian sketch for one module/side.

    Seeded from (master seed, module name, side) so it is identical on every
    machine and every run, without storing the matrix.
    """
    tag = f"{spec.seed}|{name}|{kind}".encode()
    seed = int.from_bytes(hashlib.sha256(tag).digest()[:8], "big") % (2**63)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    m = torch.randn(rows, cols, generator=gen, dtype=torch.float32)
    m /= rows ** 0.5          # preserve expected inner products
    return m.to(device=device, dtype=dtype)


class LoRAProjector:
    """Projects per-sample LoRA gradients to a fixed low-dimensional vector.

    Layout is deterministic: module order (sorted by name) x {A, B}. Two runs
    with the same ProjectionSpec and module layout produce directly comparable
    vectors — which is exactly what H1's cross-checkpoint comparison needs.
    """

    def __init__(self, refs: list[LoRAModuleRef], spec: ProjectionSpec,
                 device="cpu", compute_dtype=torch.float32):
        self.refs = refs
        self.spec = spec
        self.device = device
        self.compute_dtype = compute_dtype
        self._cache: dict[tuple[str, str], torch.Tensor] = {}

    def _get(self, name: str, kind: str, rows: int, cols: int, device) -> torch.Tensor:
        # Device is part of the cache key: a model sharded with
        # device_map="auto" puts different layers on different GPUs, so a
        # sketch cached on cuda:0 cannot multiply a gradient living on cuda:1.
        # The sketch VALUES are device-independent (seeded on CPU), so this only
        # duplicates memory, never changes results.
        key = (name, kind, str(device))
        if key not in self._cache:
            self._cache[key] = _sketch(name, kind, rows, cols, self.spec,
                                       device, self.compute_dtype)
        return self._cache[key]

    def project_matrix(self, name: str, kind: str, grad: torch.Tensor) -> torch.Tensor:
        """Two-sided sketch: (rows x cols) -> (k_left * k_right,).

        S_l^T @ grad @ S_r  keeps a JL-style unbiased estimate of inner products
        while never materialising a dense (numel x k) projection matrix.
        """
        rows, cols = grad.shape
        s_l = self._get(name, f"{kind}_l", rows, self.spec.k_left, grad.device)
        s_r = self._get(name, f"{kind}_r", cols, self.spec.k_right, grad.device)
        # detach: gradient tensors reconstructed from hooks can still carry an
        # autograd connection (they are built from weight tensors that require
        # grad). Without this, projecting 10k samples retains 10k graphs and
        # blows up memory long before it fails loudly.
        g = grad.detach().to(self.compute_dtype)
        return (s_l.T @ g @ s_r).reshape(-1).detach()

    def project_sample(
        self, grads: dict[str, tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """Project one sample's {module: (grad_A, grad_B)} into a flat vector."""
        parts: list[torch.Tensor] = []
        for ref in self.refs:                       # fixed order == fixed layout
            gA, gB = grads[ref.name]
            parts.append(self.project_matrix(ref.name, "A", gA))
            parts.append(self.project_matrix(ref.name, "B", gB))
        # Gather onto one device before concatenating: a sharded model yields
        # parts spread across GPUs.
        target = parts[0].device
        return torch.cat([p.to(target) for p in parts])

    @property
    def out_dim(self) -> int:
        return self.spec.total_dim(self.refs)
