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


class EquivariantLinear(nn.Module):
    """Block-diagonal linear layer preserving equivariance.

    Same weights for cos/sin parts. Bias only on m=0 (invariant).
    Angular channels: m = 0, 1, ..., m_max (index 0 is m=0).
    """

    def __init__(self, in_features, out_features, n_angular, m_max, n_types=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_angular = n_angular
        self.m_max = m_max
        self.edge_type_linear = (n_types is not None)

        # (n_angular, out_features, in_features)
        std = (2.0 / (in_features + out_features)) ** 0.5
        self.weights = nn.Parameter(torch.randn(n_angular, out_features, in_features) * std)

        # Factorized per-type additive correction to weights; zero-init → shared baseline at start
        # W_corr[e] = weights_src[type_i[e]] + weights_tgt[type_j[e]]
        if self.edge_type_linear:
            self.weights_src = nn.Parameter(
                torch.zeros(n_types, n_angular, out_features, in_features)
            )
            self.weights_tgt = nn.Parameter(
                torch.zeros(n_types, n_angular, out_features, in_features)
            )

        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, A_cos, A_sin, type_i=None, type_j=None, type_idx=None):
        if self.edge_type_linear and (type_idx is not None or type_i is not None):
            n_t = self.weights_src.shape[0]

            # 2-pass type grouping (W_eff = weights + W_src[z_i] + W_tgt[z_j]):
            # sort edges by source type, then by target type. type_idx is the
            # precomputed (src_perm, src_sizes, tgt_perm, tgt_sizes) tuple from
            # precompute_type_idx_2pass; compute it inline if not supplied.
            if type_idx is not None:
                src_perm, src_sizes, tgt_perm, tgt_sizes = type_idx
            else:
                src_perm  = type_i.argsort(stable=True)
                src_sizes = type_i.bincount(minlength=n_t).tolist()
                tgt_perm  = type_j.argsort(stable=True)
                tgt_sizes = type_j.bincount(minlength=n_t).tolist()

            A_cos_out = A_cos.new_zeros(A_cos.shape[0], self.out_features, self.n_angular)
            A_sin_out = A_sin.new_zeros(A_sin.shape[0], self.out_features, self.n_angular)
            # Pass 1: (W + W_src[ti]) per source type
            offset = 0
            for ti, sz in enumerate(src_sizes):
                if sz > 0:
                    W_eff = self.weights + self.weights_src[ti]
                    idx   = src_perm[offset:offset + sz]
                    A_cos_out = A_cos_out.index_add(0, idx,
                        torch.einsum('eid,doi->eod', A_cos[idx], W_eff))
                    A_sin_out = A_sin_out.index_add(0, idx,
                        torch.einsum('eid,doi->eod', A_sin[idx], W_eff))
                offset += sz
            # Pass 2: add W_tgt[tj] per target type
            offset = 0
            for tj, sz in enumerate(tgt_sizes):
                if sz > 0:
                    idx = tgt_perm[offset:offset + sz]
                    A_cos_out = A_cos_out.index_add(0, idx,
                        torch.einsum('eid,doi->eod', A_cos[idx], self.weights_tgt[tj]))
                    A_sin_out = A_sin_out.index_add(0, idx,
                        torch.einsum('eid,doi->eod', A_sin[idx], self.weights_tgt[tj]))
                offset += sz
        else:
            # No type dependence: standard single matmul
            A_cos_out = torch.einsum('...id,doi->...od', A_cos, self.weights)
            A_sin_out = torch.einsum('...id,doi->...od', A_sin, self.weights)

        # Bias only on m=0 (index 0)
        A_cos_out[..., 0] = A_cos_out[..., 0] + self.bias

        return A_cos_out, A_sin_out


class RealSpaceNonlinearity(nn.Module):
    """Nonlinear layer via real-space transform on the angular coordinate.

    Transforms Fourier coefficients (cos/sin parts for m=0,...,m_max) to
    function values on a uniform θ grid, applies a pointwise nonlinearity,
    and transforms back to Fourier space.

    This preserves equivariance because pointwise operations in angular
    space commute with rotation (θ → θ - φ).

    Optionally mixes feature channels at each grid point via an MLP,
    which is also equivariant since it acts identically at every θ.

    When r_ij_embed_dim or edge_embed_dim > 0, the pre-activation scale and
    shift become data-dependent: predicted from invariants via a small MLP.
    This gives each edge its own operating point for the nonlinearity.

    When rms_norm=True, f(θ) is normalized to unit RMS per feature before
    the activation, keeping the signal in the activation's nonlinear regime
    regardless of amplitude growth during training. A learnable post_scale
    rescales the output.

    Args:
        n_features: number of feature channels
        m_max: maximum angular frequency
        n_grid: number of θ grid points (default: 4*m_max + 1)
        activation: pointwise nonlinearity ('silu', 'relu', 'tanh', 'gelu')
        mix_channels: if True, use an MLP that mixes features at each θ point
        hidden_dim: MLP hidden dimension (only used if mix_channels=True)
        r_ij_embed_dim: dimension of r_ij embedding (0 to disable)
        edge_embed_dim: dimension of edge type embedding (0 to disable)
        rms_norm: if True, normalize f(θ) to unit RMS before activation
    """

    def __init__(self, n_features, m_max, n_grid=None, activation='silu',
                 mix_channels=False, hidden_dim=None,
                 r_ij_embed_dim=0, edge_embed_dim=0, rms_norm=False,
                 n_types=None):
        super().__init__()
        self.n_features = n_features
        self.m_max = m_max
        self.n_angular = m_max + 1
        self.mix_channels = mix_channels
        self.r_ij_embed_dim = r_ij_embed_dim
        self.edge_embed_dim = edge_embed_dim
        self.rms_norm = rms_norm
        self.edge_type_nonlin = (n_types is not None)

        # Grid size: oversample to reduce aliasing from the nonlinearity.
        if n_grid is None:
            n_grid = 2 * (2 * m_max) + 1
        n_grid = int(n_grid)
        self.n_grid = n_grid

        # Uniform grid on [0, 2π)
        theta = torch.linspace(0, 2 * np.pi, n_grid + 1)[:-1]

        # Synthesis matrix: Fourier → grid
        # f(θ_k) = Σ_d [A_cos[m]*cos(m*θ_k) + A_sin[m]*sin(m*θ_k)]
        cos_synth = torch.stack([torch.cos(m * theta) for m in range(m_max + 1)], dim=0)
        sin_synth = torch.stack([torch.sin(m * theta) for m in range(m_max + 1)], dim=0)
        self.register_buffer('cos_synth', cos_synth)  # (n_angular, n_grid)
        self.register_buffer('sin_synth', sin_synth)  # (n_angular, n_grid)

        # Analysis matrix: grid → Fourier
        # A_cos[m] = norm[m] * Σ_k f(θ_k) * cos(m*θ_k)
        # norm[0] = 1/N, norm[m>0] = 2/N
        norm = torch.ones(m_max + 1) * (2.0 / n_grid)
        norm[0] = 1.0 / n_grid
        self.register_buffer('cos_analysis', cos_synth.T * norm.unsqueeze(0))  # (n_grid, n_angular)
        self.register_buffer('sin_analysis', sin_synth.T * norm.unsqueeze(0))  # (n_grid, n_angular)

        # Pre-activation scale and shift
        nonlin_features = n_features
        cond_dim = r_ij_embed_dim + edge_embed_dim
        self.data_dependent = cond_dim > 0
        if self.data_dependent:
            # Data-dependent: predict per-edge scale and shift from invariants
            cond_hidden = max(cond_dim, 32)
            self.cond_net = nn.Sequential(
                nn.Linear(cond_dim, cond_hidden),
                nn.SiLU(),
                nn.Linear(cond_hidden, 2 * n_features),
            )
            # Zero-init last layer → scale=1, shift=0 at init
            nn.init.zeros_(self.cond_net[-1].weight)
            nn.init.zeros_(self.cond_net[-1].bias)
        elif self.edge_type_nonlin:
            # Factorized: scale/shift = src[type_i] + tgt[type_j]; ones/zeros init → identity
            self.pre_scale_src = nn.Parameter(torch.zeros(n_types, nonlin_features, 1))
            self.pre_scale_tgt = nn.Parameter(torch.zeros(n_types, nonlin_features, 1))
            self.pre_shift_src = nn.Parameter(torch.zeros(n_types, nonlin_features, 1))
            self.pre_shift_tgt = nn.Parameter(torch.zeros(n_types, nonlin_features, 1))
            # Shared baseline so (src=0, tgt=0) → scale=1, shift=0
            self.pre_scale_base = nn.Parameter(torch.ones(nonlin_features, 1))
            self.pre_shift_base = nn.Parameter(torch.zeros(nonlin_features, 1))
        else:
            # No learnable affine on this nonlinearity: pure σ(f(θ)). Fixed
            # buffers (scale=1, shift=0) keep the pre-activation step an identity.
            self.register_buffer('pre_scale', torch.ones(nonlin_features, 1))
            self.register_buffer('pre_shift', torch.zeros(nonlin_features, 1))

        # RMS normalization before activation
        if self.rms_norm:
            self.rms_eps = 1e-6
            self.post_scale = nn.Parameter(torch.ones(n_features, 1))

        # Nonlinearity
        act_map = {'silu': nn.SiLU, 'relu': nn.ReLU, 'tanh': nn.Tanh, 'gelu': nn.GELU}
        if mix_channels:
            if hidden_dim is None:
                hidden_dim = n_features
            self.mlp = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                act_map[activation](),
                nn.Linear(hidden_dim, n_features),
            )
            nn.init.zeros_(self.mlp[-1].bias)
        else:
            self.activation = act_map[activation]()

    def forward(self, A_cos, A_sin, r_ij_embed=None, edge_embed=None,
                type_i=None, type_j=None):
        """
        Args:
            A_cos, A_sin: (n_edges, n_features, n_angular)
            r_ij_embed: (n_edges, r_ij_embed_dim) or None
            edge_embed: (n_edges, edge_embed_dim) or None
            type_i, type_j: (n_edges,) int tensors of source/target atom types, or None

        Returns:
            A_cos_out, A_sin_out: (n_edges, n_features, n_angular)
        """
        # Synthesis: Fourier coefficients → grid values
        f_grid = A_cos @ self.cos_synth + A_sin @ self.sin_synth

        # Pre-activation affine transform
        if self.data_dependent:
            # Predict per-edge scale and shift from invariants
            cond_parts = []
            if self.r_ij_embed_dim > 0 and r_ij_embed is not None:
                cond_parts.append(r_ij_embed)
            if self.edge_embed_dim > 0 and edge_embed is not None:
                cond_parts.append(edge_embed)
            cond_input = torch.cat(cond_parts, dim=-1)  # (n_edges, cond_dim)
            cond_out = self.cond_net(cond_input)  # (n_edges, 2 * n_features)
            scale_delta, shift = cond_out.split(self.n_features, dim=-1)
            # scale = 1 + delta so it starts near identity
            scale = (1.0 + scale_delta).unsqueeze(-1)  # (n_edges, n_features, 1)
            shift = shift.unsqueeze(-1)                 # (n_edges, n_features, 1)
            f_grid = scale * f_grid + shift
        elif self.edge_type_nonlin:
            scale = (self.pre_scale_base
                     + self.pre_scale_src[type_i]
                     + self.pre_scale_tgt[type_j])   # (n_edges, n_features, 1)
            shift = (self.pre_shift_base
                     + self.pre_shift_src[type_i]
                     + self.pre_shift_tgt[type_j])
            f_grid = scale * f_grid + shift
        else:
            f_grid = self.pre_scale * f_grid + self.pre_shift

        # RMS normalization: normalize per-feature to unit RMS across grid
        if self.rms_norm:
            rms = torch.sqrt(torch.mean(f_grid ** 2, dim=-1, keepdim=True) + self.rms_eps)
            f_grid = f_grid / rms

        # Apply nonlinearity
        if self.mix_channels:
            # Reshape to (n_edges * n_grid, n_features) for channel-mixing MLP
            shape = f_grid.shape
            f_flat = f_grid.permute(0, 2, 1).reshape(-1, self.n_features)
            f_flat = self.mlp(f_flat)
            f_grid = f_flat.reshape(shape[0], self.n_grid, self.n_features).permute(0, 2, 1)
        else:
            f_grid = self.activation(f_grid)

        # Rescale after activation
        if self.rms_norm:
            f_grid = self.post_scale * f_grid

        # Analysis: grid values → Fourier coefficients
        A_cos_out = f_grid @ self.cos_analysis
        A_sin_out = f_grid @ self.sin_analysis

        return A_cos_out, A_sin_out
