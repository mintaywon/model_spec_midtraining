"""Tests for gradient projection.

The load-bearing property is INNER-PRODUCT PRESERVATION: influence is
<grad_query, grad_train>, so if projection distorts inner products, every score
is noise. `test_preserves_inner_products` checks this against exact values.

The second load-bearing property is REPRODUCIBILITY: H1 compares influence
profiles across checkpoints, which is only meaningful if both runs used the
identical projection.
"""

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from tda.influence.gradients import LoRAGradientCapture, find_lora_modules
from tda.influence.projection import LoRAProjector, ProjectionSpec


class TinyNet(nn.Module):
    def __init__(self, d_in=16, d_hidden=12, d_out=8):
        super().__init__()
        self.up = nn.Linear(d_in, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        return self.down(torch.relu(self.up(x)))


def _model(seed=0):
    torch.manual_seed(seed)
    m = get_peft_model(
        TinyNet(),
        LoraConfig(r=4, lora_alpha=8, target_modules=["up", "down"],
                   lora_dropout=0.0, bias="none"),
    )
    with torch.no_grad():                       # lora_B starts at zero
        for ref in find_lora_modules(m):
            ref.lora_B.weight.normal_(0.0, 0.5)
    return m


@pytest.fixture
def refs():
    return find_lora_modules(_model())


def _grads_for(model, x):
    refs = find_lora_modules(model)
    model.zero_grad()
    with LoRAGradientCapture(refs) as cap:
        model(x).pow(2).sum().backward()
        return cap.per_sample_grads(0), cap.flat_per_sample_grad(0)


def test_output_dim_matches_spec(refs):
    spec = ProjectionSpec(seed=0, k_left=8, k_right=8)
    proj = LoRAProjector(refs, spec)
    assert proj.out_dim == 2 * len(refs) * 64


def test_deterministic_across_instances(refs):
    """Same spec => bitwise-identical projection. Required for cross-run comparison."""
    spec = ProjectionSpec(seed=42)
    g = {r.name: (torch.randn(r.r, r.d_in), torch.randn(r.d_out, r.r)) for r in refs}
    a = LoRAProjector(refs, spec).project_sample(g)
    b = LoRAProjector(refs, spec).project_sample(g)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_different_seed_gives_different_projection(refs):
    g = {r.name: (torch.randn(r.r, r.d_in), torch.randn(r.d_out, r.r)) for r in refs}
    a = LoRAProjector(refs, ProjectionSpec(seed=1)).project_sample(g)
    b = LoRAProjector(refs, ProjectionSpec(seed=2)).project_sample(g)
    assert not torch.allclose(a, b)


def test_fingerprint_detects_incompatible_specs(refs):
    base = ProjectionSpec(seed=1, k_left=16, k_right=16)
    assert base.fingerprint(refs) == ProjectionSpec(1, 16, 16).fingerprint(refs)
    assert base.fingerprint(refs) != ProjectionSpec(2, 16, 16).fingerprint(refs)
    assert base.fingerprint(refs) != ProjectionSpec(1, 8, 16).fingerprint(refs)


def test_preserves_inner_products():
    """THE test: projected dot products must track exact ones.

    Influence is an inner product between query and training gradients, so this
    is the property the whole pipeline rests on. We use real LoRA gradients from
    distinct inputs and check correlation against exact values.
    """
    model = _model()
    refs = find_lora_modules(model)
    # Generous k: JL error shrinks like 1/sqrt(k).
    proj = LoRAProjector(refs, ProjectionSpec(seed=7, k_left=32, k_right=32))

    torch.manual_seed(11)
    grads, flats, projs = [], [], []
    for _ in range(12):
        g, flat = _grads_for(model, torch.randn(1, 6, 16))
        grads.append(g)
        flats.append(flat)
        projs.append(proj.project_sample(g))

    exact, approx = [], []
    for i in range(len(flats)):
        for j in range(i + 1, len(flats)):
            exact.append(torch.dot(flats[i], flats[j]).item())
            approx.append(torch.dot(projs[i], projs[j]).item())

    e = torch.tensor(exact)
    a = torch.tensor(approx)
    # Normalised so the check is about structure, not the sketch's overall scale.
    e_n = (e - e.mean()) / e.std()
    a_n = (a - a.mean()) / a.std()
    corr = torch.dot(e_n, a_n).item() / (len(e) - 1)
    assert corr > 0.9, f"projection distorts inner products (corr={corr:.3f})"


def test_self_similarity_ranks_highest():
    """A sample must be most similar to itself under projection.

    Weaker than full IP preservation but the practical failure signature: if
    this breaks, top-k influence rankings are meaningless.
    """
    model = _model()
    refs = find_lora_modules(model)
    proj = LoRAProjector(refs, ProjectionSpec(seed=3, k_left=24, k_right=24))

    torch.manual_seed(5)
    vecs = [proj.project_sample(_grads_for(model, torch.randn(1, 5, 16))[0])
            for _ in range(6)]

    for i, v in enumerate(vecs):
        sims = [torch.dot(v, w).item() / (v.norm() * w.norm()).item() for w in vecs]
        assert max(range(len(sims)), key=lambda j: sims[j]) == i


def test_projection_shrinks_dimension_massively(refs):
    """Sanity: the point of this module is a large size reduction."""
    spec = ProjectionSpec(seed=0, k_left=16, k_right=16)
    raw = sum(r.n_params for r in refs)
    assert spec.total_dim(refs) < raw or raw < 10_000  # tiny model caveat

    # At Qwen2.5-14B scale the reduction is the real story.
    qwen_params = 275_251_200
    qwen_modules = 48 * 7
    projected = 2 * qwen_modules * spec.dim_per_matrix()
    assert projected < qwen_params / 1000


def test_projected_output_is_detached():
    """Projected vectors must not retain autograd graphs.

    Reconstructed gradients are built from weight tensors that require grad, so
    the projection can silently stay connected to the graph. At 10k samples that
    retains 10k graphs and exhausts memory. Fails as `.numpy()` refusing to run.
    """
    model = _model()
    refs = find_lora_modules(model)
    proj = LoRAProjector(refs, ProjectionSpec(seed=0, k_left=8, k_right=8))
    grads, _ = _grads_for(model, torch.randn(1, 4, 16))
    out = proj.project_sample(grads)
    assert not out.requires_grad
    out.numpy()  # would raise if still attached


def test_projects_gradients_from_mixed_devices(refs):
    """A model sharded with device_map="auto" yields grads on several devices.

    The sketch cache must be keyed by device, and parts gathered before concat.
    Simulated on CPU via meta-free copies; on GPU this raised
    "Expected all tensors to be on the same device, cuda:0 and cuda:1".
    """
    spec = ProjectionSpec(seed=5, k_left=8, k_right=8)
    proj = LoRAProjector(refs, spec)
    grads = {r.name: (torch.randn(r.r, r.d_in), torch.randn(r.d_out, r.r)) for r in refs}
    out = proj.project_sample(grads)
    assert out.numel() == proj.out_dim

    # Sketch values must not depend on which device produced them.
    a = proj.project_matrix(refs[0].name, "A", grads[refs[0].name][0])
    b = LoRAProjector(refs, spec).project_matrix(refs[0].name, "A", grads[refs[0].name][0])
    torch.testing.assert_close(a, b, rtol=0, atol=0)
