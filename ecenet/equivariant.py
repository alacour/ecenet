"""SO(2)-equivariant layers operating on bond-frame angular features.

Once ACE features are Wigner-rotated into a bond frame (see model.py /
ace_basis.py), each angular mode ``m`` transforms as ``e^{imφ}`` under rotation
about the bond axis. Features are carried as cos/sin Fourier pairs
``(A_cos, A_sin)`` of shape ``(n_edges, n_features, n_angular)`` with
``n_angular = m_max + 1``. This module provides the layer types that act on that
representation while preserving the SO(2) structure:

- ``EquivariantLinear``: per-mode channel mixing (block-diagonal across ``m``),
  the same weights applied to the cos and sin parts, bias only on the ``m=0``
  (invariant) channel.
- ``RealSpaceNonlinearity``: applies a pointwise nonlinearity equivariantly via
  iDFT → σ → DFT on a θ-grid, coupling modes while staying SO(2)-equivariant.
"""

import numpy as np
import torch
import torch.nn as nn

from ecenet.equivariant_linear_kernel import EquivariantLinearTriton, el_triton_ok
from ecenet.realspace_kernel import RealSpaceFused, is_fusible


class EquivariantLinear(nn.Module):
    """Block-diagonal linear layer preserving equivariance.

    Same weights for cos/sin parts. Bias only on m=0 (invariant).
    Angular channels: m = 0, 1, ..., m_max (index 0 is m=0).

    The map y[e,o,m] = Σ_i x[e,i,m]·W[m,o,i] never mixes angular modes, so it
    is evaluated as ONE dense row-major GEMM over the flattened (feature,
    angular) axis, against a (in·n_ang, out·n_ang) weight that is
    block-diagonal per m (assembled from the parameter each call — ~MB-scale,
    autograd flows through the assembly; the zero blocks cost n_ang× the
    strictly needed flops, which TF32/tensor cores absorb). The earlier
    einsum formulation lowered to a batched bmm over the angular axis, whose
    batch-contiguity requirement copied the full per-edge activation on every
    call and emitted an m-major output that downstream consumers (the fused
    nonlinearity, residual adds, the next einsum) paid to re-normalize —
    profiling showed that layout churn at ~29% of the whole force step. The
    dense form's input and output are both plain row-major views: no
    permutes, no copies, and the layout every consumer wants.

    The trade only pays where the padded GEMM is (near-)free: on CUDA tensor
    cores. On fp32 CUDA cores (TF32 off) and in float64 the 3x padding costs
    exactly 3x — measured slower than the einsum path despite the copies —
    so dispatch is automatic: dense for CUDA fp16/bf16, and for CUDA fp32
    with TF32 enabled; einsum otherwise. ``dense_gemm`` (True/False)
    overrides the automatic choice.

    Triton path (``equivariant_linear_kernel``): on CUDA fp32 with Triton
    importable, a per-mode tensor-core dot reads the interleaved layout
    directly — no copies (einsum) and no zero-block padding (dense). It is
    the first choice there; ``triton_gemm`` (True/False) overrides, and
    ``ECENET_EL_TRITON=0`` disables it process-wide for A/B timing.
    """

    def __init__(self, in_features, out_features, n_angular, m_max):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_angular = n_angular
        self.m_max = m_max
        self.dense_gemm = None       # None = auto (see _use_dense)
        self.triton_gemm = None      # None = auto (see _use_triton)

        # (n_angular, out_features, in_features)
        std = (2.0 / (in_features + out_features)) ** 0.5
        self.weights = nn.Parameter(torch.randn(n_angular, out_features, in_features) * std)

        self.bias = nn.Parameter(torch.zeros(out_features))

    def _use_dense(self, x):
        if self.dense_gemm is not None:
            return self.dense_gemm
        if not x.is_cuda:
            return False
        if x.dtype in (torch.float16, torch.bfloat16):
            return True
        return (x.dtype == torch.float32
                and torch.backends.cuda.matmul.allow_tf32)

    def _use_triton(self, x):
        if self.triton_gemm is not None:
            return self.triton_gemm
        return el_triton_ok(x, self.n_angular)

    def _dense_weight(self):
        """(in·n_ang, out·n_ang) block-diagonal-per-m weight, row-major flat
        index c = feature·n_ang + m on both sides."""
        na, Fo, Fi = self.weights.shape
        Wd = self.weights.new_zeros(Fi, na, Fo, na)
        m = torch.arange(na, device=self.weights.device)
        Wd[:, m, :, m] = self.weights.permute(0, 2, 1)   # (na, Fi, Fo) slot
        return Wd.reshape(Fi * na, Fo * na)

    def forward(self, A_cos, A_sin):
        if self._use_triton(A_cos):
            return EquivariantLinearTriton.apply(A_cos, A_sin, self.weights, self.bias)
        if not self._use_dense(A_cos):
            A_cos_out = torch.einsum('...id,doi->...od', A_cos, self.weights)
            A_sin_out = torch.einsum('...id,doi->...od', A_sin, self.weights)
            A_cos_out[..., 0] = A_cos_out[..., 0] + self.bias   # bias on m=0 only
            return A_cos_out, A_sin_out

        na, Fo = self.n_angular, self.out_features
        lead = A_cos.shape[:-2]
        Wd = self._dense_weight()
        # Bias enters the GEMM epilogue (addmm), on the m=0 slots of the cos
        # part only — sin has no invariant channel.
        bias_full = self.bias.new_zeros(Fo, na)
        bias_full[:, 0] = self.bias
        x_cos = A_cos.reshape(-1, self.in_features * na)
        x_sin = A_sin.reshape(-1, self.in_features * na)
        y_cos = torch.addmm(bias_full.reshape(-1), x_cos, Wd)
        y_sin = x_sin @ Wd
        return y_cos.view(*lead, Fo, na), y_sin.view(*lead, Fo, na)


class RealSpaceNonlinearity(nn.Module):
    """Nonlinear layer via real-space transform on the angular coordinate.

    Transforms Fourier coefficients (cos/sin parts for m=0,...,m_max) to
    function values on a uniform θ grid, applies a pointwise nonlinearity,
    and transforms back to Fourier space.

    This preserves equivariance because pointwise operations in angular
    space commute with rotation (θ → θ - φ).

    Args:
        n_features: number of feature channels
        m_max: maximum angular frequency
        n_grid: number of θ grid points (default: 4*m_max + 1)
        activation: pointwise nonlinearity ('silu', 'relu', 'tanh', 'gelu'),
                    or 'identity' — an exact no-op: synthesis→analysis is an
                    exact round-trip for bandlimited features (the grid
                    oversamples), so the block reduces to its linear part
    """

    def __init__(self, n_features, m_max, n_grid=None, activation='silu'):
        super().__init__()
        self.n_features = n_features
        self.m_max = m_max
        self.n_angular = m_max + 1

        # Grid size: oversample to reduce aliasing from the nonlinearity.
        if n_grid is None:
            n_grid = 2 * (2 * m_max) + 1
        n_grid = int(n_grid)
        self.n_grid = n_grid

        # DFT synthesis/analysis buffers — filled by _build_bases(), which always
        # computes them in float64 and casts down. See _build_bases for why.
        self.register_buffer('cos_synth', torch.empty(m_max + 1, n_grid))     # (n_angular, n_grid)
        self.register_buffer('sin_synth', torch.empty(m_max + 1, n_grid))
        self.register_buffer('cos_analysis', torch.empty(n_grid, m_max + 1))  # (n_grid, n_angular)
        self.register_buffer('sin_analysis', torch.empty(n_grid, m_max + 1))
        self._build_bases()

        # No pre-activation affine: this is pure σ(f(θ)). Earlier versions kept
        # fixed scale=1 / shift=0 buffers and applied them in forward, which was
        # an exact identity — two elementwise passes over the full (n_edges,
        # n_features, n_grid) grid tensor for nothing. Removed; checkpoints that
        # still carry the buffers load fine (see calculator.from_checkpoint).

        # Nonlinearity. 'identity' makes the whole block an exact no-op (the
        # round-trip through the θ grid is exact for bandlimited features), so
        # the enclosing layer reduces to its linear part — the linearized-
        # model ablation. The Triton fast path dispatches only for silu, so
        # every other choice (identity included) takes the generic paths.
        act_map = {'silu': nn.SiLU, 'relu': nn.ReLU, 'tanh': nn.Tanh,
                   'gelu': nn.GELU, 'identity': nn.Identity}
        if activation not in act_map:
            raise ValueError(f"activation must be one of {sorted(act_map)}, "
                             f"got {activation!r}")
        self.activation = act_map[activation]()

        # Opt-in fused path (recompute-in-backward; Triton on CUDA). Set via
        # ECENet.set_activation_fused. Off by default. See ecenet/realspace_kernel.
        self.fused = False

    def _build_bases(self):
        """Fill the DFT synthesis/analysis buffers, computing them in float64 and
        casting into whatever dtype the buffers currently hold.

        These are exact deterministic constants (a uniform-grid DFT pair), and the
        layer's equivariance rests on the analysis quadrature being faithful: the
        pointwise activation commutes with the bond-frame rotation θ → θ-φ only if
        synthesis→analysis round-trips a bandlimited function accurately.

        They used to be built at the *default* dtype — torch.linspace with no dtype
        is float32 — so a later .double() left the buffers float64-typed but
        float32-*accurate*. That capped the round-trip, and with it the model's
        rotational consistency, at ~1e-7 no matter the working precision. Hence:
        always compute in float64, then cast. Rebuilt on every dtype/device change
        (_apply) and after load_state_dict, so a checkpoint's stored copy can never
        reintroduce that rounding.
        """
        dev = self.cos_synth.device
        f64 = torch.float64
        # Uniform grid on [0, 2π)
        theta = torch.linspace(0, 2 * np.pi, self.n_grid + 1, dtype=f64, device=dev)[:-1]
        m = torch.arange(self.n_angular, dtype=f64, device=dev).unsqueeze(1)  # (n_ang, 1)

        # Synthesis: f(θ_k) = Σ_m A_cos[m]·cos(m·θ_k) + A_sin[m]·sin(m·θ_k)
        cos_synth = torch.cos(m * theta)
        sin_synth = torch.sin(m * theta)

        # Analysis: A_cos[m] = norm[m]·Σ_k f(θ_k)·cos(m·θ_k);  norm[0]=1/N, norm[m>0]=2/N
        norm = torch.full((self.n_angular,), 2.0 / self.n_grid, dtype=f64, device=dev)
        norm[0] = 1.0 / self.n_grid

        dt = self.cos_synth.dtype
        self.cos_synth.copy_(cos_synth.to(dt))
        self.sin_synth.copy_(sin_synth.to(dt))
        self.cos_analysis.copy_((cos_synth.T * norm.unsqueeze(0)).to(dt))
        self.sin_analysis.copy_((sin_synth.T * norm.unsqueeze(0)).to(dt))

    def _apply(self, *args, **kwargs):
        # .to(dtype)/.double()/.cuda() cast the buffers in place; recompute them at
        # full precision afterwards rather than inheriting the old dtype's rounding.
        out = super()._apply(*args, **kwargs)
        out._build_bases()
        return out

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._build_bases()  # constants — never take them from a checkpoint

    def forward(self, A_cos, A_sin):
        """
        Args:
            A_cos, A_sin: (n_edges, n_features, n_angular)

        Returns:
            A_cos_out, A_sin_out: (n_edges, n_features, n_angular)
        """
        # Fused path: recompute the grid tensor in the backward instead of saving
        # it (drops the ~3x (n_e,F,n_grid) transient from the saved-for-backward
        # set). Only when grad is on (the win is the backward) and the config is
        # the common one. Inference/forces — not double-backward (force-loss
        # training keeps the path below).
        if self.fused and is_fusible(self) and torch.is_grad_enabled():
            return RealSpaceFused.apply(
                A_cos, A_sin, self.cos_synth, self.sin_synth,
                self.cos_analysis, self.sin_analysis, self.activation)

        # Synthesis: Fourier coefficients → grid values
        f_grid = A_cos @ self.cos_synth + A_sin @ self.sin_synth

        # Apply nonlinearity
        f_grid = self.activation(f_grid)

        # Analysis: grid values → Fourier coefficients
        A_cos_out = f_grid @ self.cos_analysis
        A_sin_out = f_grid @ self.sin_analysis

        return A_cos_out, A_sin_out
