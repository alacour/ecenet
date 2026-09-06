"""Tests for the block-diagonal-GEMM EquivariantLinear (``ecenet/equivariant.py``).

The layer computes y[e,o,m] = Σ_i x[e,i,m]·W[m,o,i] (bias on the m=0 cos slot
only) as one dense row-major GEMM against a per-m block-diagonal weight. These
tests pin it to the original einsum formulation:

  1. forward equivalence vs the einsum reference (several shapes, fp64);
  2. gradient equivalence — input, weight, and bias grads vs the reference;
  3. non-contiguous (m-major, einsum-layout) inputs give the same result;
  4. batched leading dims (B, n_e, F, n_ang) round-trip unchanged;
  5. (CUDA + Triton only) the per-mode Triton kernel path — forward and all
     grads vs the fp64 reference, IEEE and TF32 dots, awkward shapes, m-major
     inputs, leading dims, and the frozen-parameter (no dW) case.

Pure PyTorch on CPU (fp64) for 1-4; 5 is skipped without CUDA/Triton.
Run:  python tests/test_equivariant_linear.py
"""

import os
import sys  # noqa: E402 — repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch

from ecenet.equivariant import EquivariantLinear
from ecenet.equivariant_linear_kernel import _HAS_TRITON, el_triton_ok

torch.manual_seed(0)
DTYPE = torch.float64


def reference(A_cos, A_sin, weights, bias):
    """The original einsum formulation — the spec."""
    oc = torch.einsum('...id,doi->...od', A_cos, weights)
    os_ = torch.einsum('...id,doi->...od', A_sin, weights)
    oc = oc.clone()
    oc[..., 0] = oc[..., 0] + bias
    return oc, os_


def make_layer(Fi=12, Fo=9, m_max=2, seed=0):
    torch.manual_seed(seed)
    lin = EquivariantLinear(Fi, Fo, m_max + 1, m_max).to(DTYPE)
    lin.dense_gemm = True     # force the dense path (auto picks einsum on CPU)
    with torch.no_grad():                # non-trivial bias
        lin.bias.add_(torch.randn_like(lin.bias))
    return lin


def make_inputs(n_e=40, Fi=12, m_max=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    mk = lambda: torch.randn(n_e, Fi, m_max + 1, generator=g, dtype=DTYPE)  # noqa: E731
    return mk(), mk()


def test_forward_equivalence():
    worst = 0.0
    for Fi, Fo, m_max in ((12, 9, 2), (7, 7, 3), (1, 5, 0), (16, 3, 1)):
        lin = make_layer(Fi, Fo, m_max)
        A_cos, A_sin = make_inputs(Fi=Fi, m_max=m_max)
        oc, os_ = lin(A_cos, A_sin)
        rc, rs = reference(A_cos, A_sin, lin.weights, lin.bias)
        e = max((oc - rc).abs().max(), (os_ - rs).abs().max()).item()
        worst = max(worst, e)
        assert e < 1e-12, f"forward Fi={Fi},Fo={Fo},m={m_max}: {e:.2e}"
    print(f"  forward equivalence vs einsum reference (worst {worst:.1e})")


def test_gradient_equivalence():
    lin = make_layer()
    A_cos, A_sin = make_inputs(seed=2)
    goc = torch.randn_like(A_cos[:, :9])   # (n_e, Fo, n_ang)
    gos = torch.randn_like(A_cos[:, :9])

    def grads(fn):
        a = A_cos.detach().clone().requires_grad_(True)
        b = A_sin.detach().clone().requires_grad_(True)
        lin.zero_grad()
        oc, os_ = fn(a, b)
        (oc * goc + os_ * gos).sum().backward()
        return a.grad, b.grad, lin.weights.grad.clone(), lin.bias.grad.clone()

    new = grads(lambda a, b: lin(a, b))
    ref = grads(lambda a, b: reference(a, b, lin.weights, lin.bias))
    worst = 0.0
    for name, n, r in zip(("dA_cos", "dA_sin", "dW", "db"), new, ref):
        e = (n - r).abs().max().item()
        worst = max(worst, e)
        assert e < 1e-12, f"{name} mismatch: {e:.2e}"
    print(f"  gradient equivalence: input/weight/bias grads match (worst {worst:.1e})")


def test_noncontiguous_input():
    """m-major (einsum-layout) inputs — strides (F, 1, n_e·F) — must give the
    same result as contiguous ones (reshape copies as needed internally)."""
    lin = make_layer()
    A_cos, A_sin = make_inputs(seed=3)

    def as_m_major(t):
        n_e, F, na = t.shape
        out = torch.empty(na, n_e, F, dtype=t.dtype).permute(1, 2, 0)
        out.copy_(t)
        assert not out.is_contiguous()
        return out

    oc, os_ = lin(A_cos, A_sin)
    mc, ms = lin(as_m_major(A_cos), as_m_major(A_sin))
    e = max((oc - mc).abs().max(), (os_ - ms).abs().max()).item()
    assert e < 1e-15, f"m-major input mismatch: {e:.2e}"
    print(f"  non-contiguous (m-major) input matches contiguous ({e:.1e})")


def test_batched_leading_dims():
    lin = make_layer()
    A_cos, A_sin = make_inputs(n_e=24, seed=4)
    B_cos = A_cos.reshape(4, 6, 12, 3)
    B_sin = A_sin.reshape(4, 6, 12, 3)
    oc, os_ = lin(A_cos, A_sin)
    bc, bs = lin(B_cos, B_sin)
    assert bc.shape == (4, 6, 9, 3) and bs.shape == (4, 6, 9, 3)
    e = max((bc.reshape(24, 9, 3) - oc).abs().max(),
            (bs.reshape(24, 9, 3) - os_).abs().max()).item()
    assert e < 1e-15, f"batched mismatch: {e:.2e}"
    print(f"  batched leading dims match flat ({e:.1e})")


def test_dispatch():
    """Auto-dispatch: einsum on CPU / strict fp32 / fp64; dense only where the
    padded GEMM rides tensor cores (CUDA fp16/bf16, or fp32 with TF32 on).
    dense_gemm=True/False overrides both ways."""
    lin = EquivariantLinear(4, 4, 3, 2)
    x64 = torch.randn(2, 4, 3, dtype=torch.float64)
    assert not lin._use_dense(x64), "CPU/fp64 must take the einsum path"
    lin.dense_gemm = True
    assert lin._use_dense(x64), "dense_gemm=True must override"
    lin.dense_gemm = False
    assert not lin._use_dense(x64), "dense_gemm=False must override"
    lin.dense_gemm = None
    if torch.cuda.is_available():
        x32 = torch.randn(2, 4, 3, dtype=torch.float32, device='cuda')
        old = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            assert not lin._use_dense(x32), "CUDA fp32 without TF32 → einsum"
            torch.backends.cuda.matmul.allow_tf32 = True
            assert lin._use_dense(x32), "CUDA fp32 with TF32 → dense"
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old
        print("  dispatch: einsum on CPU/strict-fp32, dense on TF32; overrides work")
    else:
        print("  dispatch: einsum on CPU, overrides work (CUDA cases skipped)")


def _triton_available():
    return torch.cuda.is_available() and _HAS_TRITON


def _rel(a, b):
    """max |a-b| relative to max |b| — scale-free for random-normal inputs."""
    return ((a.double() - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def test_triton_vs_reference():
    """Kernel path (float32, CUDA) vs the fp64 einsum reference: forward,
    dA_cos/dA_sin/dW/db. IEEE dots must sit at fp32 round-off (K-term
    accumulation); TF32 dots at the 10-bit-mantissa level."""
    if not _triton_available():
        print("  triton vs reference: skipped (no CUDA/Triton)")
        return
    shapes = ((5, 12, 9, 2),        # E < one tile, odd widths
              (1000, 7, 7, 3),      # K, N < 16 (dot tiles masked)
              (300, 1, 5, 0),       # m_max = 0, single input channel
              (2049, 64, 96, 3),    # multi-tile K and N, E not a tile multiple
              (777, 130, 40, 1))    # K > BLOCK_K·4, N < BLOCK_N
    old = torch.backends.cuda.matmul.allow_tf32
    try:
        for tf32, tol in ((False, 1e-4), (True, 2e-2)):
            torch.backends.cuda.matmul.allow_tf32 = tf32
            worst = 0.0
            for n_e, Fi, Fo, m_max in shapes:
                lin = make_layer(Fi, Fo, m_max).float().cuda()
                lin.dense_gemm = None
                lin.triton_gemm = True
                A_cos, A_sin = make_inputs(n_e, Fi, m_max)
                a = A_cos.float().cuda().requires_grad_(True)
                b = A_sin.float().cuda().requires_grad_(True)
                oc, os_ = lin(a, b)
                goc = torch.randn_like(oc)
                gos = torch.randn_like(os_)
                lin.zero_grad()
                (oc * goc + os_ * gos).sum().backward()
                # fp64 reference with identical weights / upstream grads
                w64, b64 = lin.weights.detach().double(), lin.bias.detach().double()
                a64 = a.detach().double().requires_grad_(True)
                b64_ = b.detach().double().requires_grad_(True)
                w64.requires_grad_(True); b64.requires_grad_(True)
                rc, rs = reference(a64, b64_, w64, b64)
                (rc * goc.double() + rs * gos.double()).sum().backward()
                errs = {'fwd_cos': _rel(oc, rc), 'fwd_sin': _rel(os_, rs),
                        'dA_cos': _rel(a.grad, a64.grad), 'dA_sin': _rel(b.grad, b64_.grad),
                        'dW': _rel(lin.weights.grad, w64.grad),
                        'db': _rel(lin.bias.grad, b64.grad)}
                for k, e in errs.items():
                    worst = max(worst, e)
                    assert e < tol, (f"tf32={tf32} shape={(n_e, Fi, Fo, m_max)} "
                                     f"{k}: rel err {e:.2e} > {tol}")
            print(f"  triton vs fp64 reference, tf32={tf32}: fwd+grads worst rel {worst:.1e}")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def test_triton_layouts_and_frozen():
    """m-major inputs and leading dims go through the kernel unchanged; with
    frozen weights the backward still yields input grads (dW skipped)."""
    if not _triton_available():
        print("  triton layouts/frozen: skipped (no CUDA/Triton)")
        return
    lin = make_layer().float().cuda()
    lin.triton_gemm = True
    A_cos, A_sin = make_inputs(n_e=24, seed=5)
    a, b = A_cos.float().cuda(), A_sin.float().cuda()
    oc, os_ = lin(a, b)

    def as_m_major(t):
        n_e, F, na = t.shape
        out = torch.empty(na, n_e, F, dtype=t.dtype, device=t.device).permute(1, 2, 0)
        out.copy_(t)
        return out
    mc, ms = lin(as_m_major(a), as_m_major(b))
    e = max((oc - mc).abs().max(), (os_ - ms).abs().max()).item()
    assert e == 0.0, f"m-major input through the kernel differs: {e:.2e}"
    bc, bs = lin(a.reshape(4, 6, 12, 3), b.reshape(4, 6, 12, 3))
    assert bc.shape == (4, 6, 9, 3)
    e = max((bc.reshape(24, 9, 3) - oc).abs().max(), (bs.reshape(24, 9, 3) - os_).abs().max()).item()
    assert e == 0.0, f"leading dims through the kernel differ: {e:.2e}"

    lin.requires_grad_(False)
    a.requires_grad_(True)
    oc, os_ = lin(a, b)
    g, = torch.autograd.grad((oc.square() + os_.square()).sum(), a)
    assert g.shape == a.shape and torch.isfinite(g).all()
    assert lin.weights.grad is None
    print("  triton: m-major + leading-dim inputs bit-identical; frozen weights → dX only")


def test_triton_dispatch():
    lin = EquivariantLinear(4, 4, 3, 2)
    x = torch.randn(2, 4, 3)
    assert not lin._use_triton(x), "CPU must not take the Triton path"
    lin.triton_gemm = True
    assert lin._use_triton(x), "triton_gemm=True must override"
    lin.triton_gemm = None
    if _triton_available():
        xc = x.cuda()
        assert lin._use_triton(xc) == el_triton_ok(xc)
        assert el_triton_ok(xc) and not el_triton_ok(xc.double())
        print("  triton dispatch: auto on CUDA fp32, off on CPU/fp64, override works")
    else:
        print("  triton dispatch: off on CPU, override works (CUDA cases skipped)")


if __name__ == "__main__":
    print("EquivariantLinear block-diagonal-GEMM tests")
    test_forward_equivalence()
    test_gradient_equivalence()
    test_noncontiguous_input()
    test_batched_leading_dims()
    test_dispatch()
    test_triton_vs_reference()
    test_triton_layouts_and_frozen()
    test_triton_dispatch()
    print("All tests passed.")
