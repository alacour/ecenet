"""ecenet/model.py — ECENet: equivariant Cartesian-edge interatomic potential.

Pipeline:
  1. ACE atomic basis:
       A[i, t, n, s] = Σ_k R_n(r_ik) Y_s(r̂_ik) δ(type_k=t)
       shape: (n_atoms, n_types, n_max, n_sph)

  2. Joint contraction per central atom type:
       A_emb[i, c, s] = Σ_{t,n} A[i, t, n, s] * W[types[i], t, n, c]
       shape: (n_atoms, embed_dim, n_sph)
       W: (n_types, n_types, n_max, embed_dim)

  3. Gather for edge endpoints + Wigner rotation into bond frame:
       stack [A_emb[edge_i], A_emb[edge_j]] → rotate by D(r̂_ij)
       shape: (n_edges, 2*embed_dim, n_sph)

  4. Reshape to A_cos / A_sin:
       shape: (n_edges, 2*embed_dim, n_angular)  where n_angular = l_max + 1

  5. Equivariant layers × n_layers (EquivariantLinear → nonlinearity → residual)

  6. Contract to invariants:
       m=0: A_cos[:, :, 0]
       m>0: A_cos[:,:,m]² + A_sin[:,:,m]²
       Optional outer product with radial basis f_d(r_ij) of rank n_max_d.

  7. Output MLP([invariants, r_ij_scaled]) → per-edge scalar → sum over edges
     + per-type atomic energy baseline
"""

import torch
import torch.nn as nn

from ecenet.ace_basis import ACEBasisAnalytic
from ecenet.equivariant import EquivariantLinear, RealSpaceNonlinearity
from ecenet.radial import find_edges, get_cutoff_fn, radial_basis
from ecenet.spherical import build_D1_from_rhat, build_D_block, spherical_harmonics_float64, wigner_rotate

# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class ECENet(nn.Module):
    """ECENet — SO(3)-equivariant interatomic potential using per-edge SO(2) features.

    Args:
        n_types:        number of atom types
        r_cut_edge:     edge formation cutoff (Å)
        r_cut_neighbor: neighbour-list cutoff for the ACE basis (Å)
        l_max:          max angular momentum of the spherical-harmonic / ACE basis
        n_max:          radial basis functions per (type, l)
        embed_dim:      embedding dim after the joint (n_types, n_max) contraction
        n_layers:       equivariant layers per stage
        n_mp:           number of stages; one equivariant message-passing layer is
                        inserted between consecutive stages (n_mp-1 MP layers, no
                        trailing MP). n_mp=1 (default) is the plain model with no
                        message passing. n_mp=K is equivalent to the old
                        (n_mp_steps=K-1, n_final_layers=n_layers) layout.
        n_dist_basis:   radial basis size for the MP distance weighting
        n_max_d:        if set, outer-product the invariants with f_d(r_ij) of this rank
        m_max:          max angular mode |m| kept after the equivariant layers
                        (default: l_max); lower it to cut cost at large l_max
        cutoff_type:    'cosine' or 'poly'
        activation:     pointwise activation in the realspace nonlinearity ('silu', 'tanh', ...)
        n_grid:         θ-grid points for the realspace nonlinearity (default: 4*m_max+1)
        output_hidden_dims: hidden widths of the readout MLP (default: [64])
        analytic_ace_basis: use ACEBasisAnalytic (recommended for force training)
    """

    def __init__(
        self,
        n_types: int,
        r_cut_edge: float = 5.0,
        r_cut_neighbor: float = 4.0,
        l_max: int = 3,
        n_max: int = 4,
        embed_dim: int = 16,
        n_layers: int = 2,
        n_mp: int = 1,
        n_dist_basis: int = 8,
        n_max_d: int = None,
        cutoff_type: str = 'cosine',
        activation: str = 'silu',
        use_nonlinearity: bool = True,
        n_grid: int = None,
        analytic_ace_basis: bool = True,
        output_hidden_dims: list = None,
        n_dist_embed: int = 0,
        m_max: int = None,
        edge_type_nonlin: bool = False,
        edge_type_linear: bool = False,
        edge_type_output: bool = False,
    ):
        super().__init__()
        self.n_types = n_types
        self.r_cut_edge = r_cut_edge
        self.r_cut_neighbor = r_cut_neighbor
        l_max = int(l_max)
        self.l_max = l_max
        self.n_max = n_max
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_max_d = n_max_d
        self.cutoff_type = cutoff_type
        self.activation = activation
        self.use_nonlinearity = use_nonlinearity
        self.n_grid = n_grid
        self.analytic_ace_basis = analytic_ace_basis
        self.n_sph = (l_max + 1) ** 2
        self.m_max = int(m_max) if m_max is not None else l_max
        self.n_angular = self.m_max + 1   # m = 0..m_max (layers only use up to m_max)
        self.n_dist_embed = n_dist_embed
        self.edge_type_nonlin = edge_type_nonlin
        self.edge_type_linear = edge_type_linear
        self.edge_type_output = edge_type_output

        # ── Joint (n_types, n_max) → embed_dim contraction per central atom type ──
        # W[type_i, t, n, c]: for central atom of type type_i, contract
        # neighbor type t and radial channel n into embed channel c.
        self.W = nn.Parameter(
            torch.randn(n_types, n_types, n_max, embed_dim)
            / (n_types * n_max) ** 0.5
        )

        # ── SH → A_cos/A_sin reshape ──────────────────────────────────────
        # m_max controls output angular modes; ACE basis always uses full l_max.
        self.sph_to_angular = SphToAngular(embed_dim, l_max, m_max=self.m_max)
        # n_features_per_m = 2 * embed_dim * (l_max+1): one channel per (side, embed, l)
        self.n_features_per_m = 2 * embed_dim * (l_max + 1)

        # ── Distance-conditioned edge transform (optional) ────────────────
        # Applied to A_cos/A_sin after the Wigner rotation and reshape.
        # Learns a per-channel distance-dependent scale:
        #   f[e]        = radial_basis(r_ij[e])       # (n_edges, n_dist_embed)
        #   scale[e, c] = Σ_n f[e,n] * V_dist[n, c]  # (n_edges, C)
        #   A_cos_out   = A_cos * (1 + scale)         # broadcast over m; V_dist=0 → identity
        # V_dist init to zero → identity at init.
        if n_dist_embed > 0:
            self.V_dist = nn.Parameter(
                torch.zeros(n_dist_embed, self.n_features_per_m)
            )

        # ── Equivariant layers: Linear → RealSpaceNonlinearity → residual ────
        _n_types_for_layers = n_types if edge_type_nonlin else None
        _n_types_linear = n_types if edge_type_linear else None
        # Message passing: the model is `n_mp` stages of `n_layers` equivariant
        # layers each, with one equivariant MP layer *between* consecutive stages
        # (n_mp-1 MP layers total, no trailing MP). n_mp == 1 is the plain model:
        # a flat list of n_layers equivariant layers and no MP. n_mp >= 2 groups
        # the layers into stages and adds the interleaved MP layers.
        self.n_mp = n_mp
        self.layers = nn.ModuleList([
            ECENetLayer(self.n_features_per_m, self.m_max, activation=activation,
                        use_nonlinearity=use_nonlinearity, n_grid=n_grid,
                        n_types=_n_types_for_layers, n_types_linear=_n_types_linear)
            for _ in range(n_mp * n_layers)
        ])
        # n_mp >= 2: regroup the flat layers into `n_mp` stages and build the
        # `n_mp - 1` MP layers that sit between them.
        if n_mp > 1:
            flat = list(self.layers)
            self.layers = nn.ModuleList([
                nn.ModuleList(flat[g * n_layers:(g + 1) * n_layers])
                for g in range(n_mp)
            ])
            self.mp_layers = nn.ModuleList([
                ECENetMPLayer(
                    self.n_features_per_m, self.l_max, self.embed_dim,
                    n_types=n_types, n_dist_basis=n_dist_basis,
                    r_cut=self.r_cut_edge, cutoff_type=self.cutoff_type,
                    m_max=self.m_max,
                )
                for _ in range(n_mp - 1)
            ])

        # ── Output MLP ──────────────────────────────────────────────────────
        # inv → MLP → n_max_d, then dot with rij_basis (see _apply_output).
        hidden_dims = output_hidden_dims or [64]
        in_dim = self.n_features_per_m
        n_output_out = n_max_d if n_max_d is not None else 1
        mlp_dims = [in_dim] + list(hidden_dims) + [n_output_out]
        act = {'silu': nn.SiLU, 'tanh': nn.Tanh, 'relu': nn.ReLU,
               'gelu': nn.GELU}.get(activation, nn.SiLU)
        self.output_net = TypeDepMLP(mlp_dims, activation=act(),
                                     n_types=n_types if edge_type_output else None)

        # ── Per-type atomic energy baseline ──────────────────────────────
        self.atomic_energy = nn.Parameter(torch.zeros(n_types))


    # ── Helpers ────────────────────────────────────────────────────────────

    def _compute_ace_basis(self, pos_batch, nb_src, nb_dst, types, shift_vecs_nb=None):
        """Compute ACE atomic basis: (B, N, n_types, n_max, n_sph)."""
        if self.analytic_ace_basis:
            cutoff_type_id = 0 if self.cutoff_type == 'cosine' else 1
            return ACEBasisAnalytic.apply(
                pos_batch, nb_src, nb_dst, types,
                self.r_cut_neighbor, self.n_max, self.l_max,
                self.n_types, cutoff_type_id, shift_vecs_nb)

        B, N, _ = pos_batch.shape
        n_nb = nb_src.shape[0]
        device, dtype = pos_batch.device, pos_batch.dtype

        if n_nb == 0:
            return torch.zeros(B, N, self.n_types, self.n_max, self.n_sph,
                               device=device, dtype=dtype)

        diff_ik = pos_batch[:, nb_dst] - pos_batch[:, nb_src]
        if shift_vecs_nb is not None:
            diff_ik = diff_ik + shift_vecs_nb.to(dtype=dtype)[None]
        r_ik = torch.sqrt((diff_ik ** 2).sum(-1) + 1e-30)
        r_hat_ik = diff_ik / r_ik.unsqueeze(-1)

        f_R = radial_basis(r_ik.reshape(-1), self.r_cut_neighbor, self.n_max,
                           cutoff_type=self.cutoff_type).reshape(B, n_nb, self.n_max)
        Y = spherical_harmonics_float64(self.l_max, r_hat_ik.reshape(-1, 3),
                                        normalize=False).reshape(B, n_nb, self.n_sph)
        contributions = f_R.unsqueeze(-1) * Y.unsqueeze(-2)  # (B, n_nb, n_max, n_sph)

        neighbor_types = types[nb_dst]
        flat_idx = nb_src * self.n_types + neighbor_types
        flat_idx_exp = flat_idx[None, :, None, None].expand(B, n_nb, self.n_max, self.n_sph)
        A_flat = torch.zeros(B, N * self.n_types, self.n_max, self.n_sph,
                             device=device, dtype=dtype)
        A_flat = A_flat.scatter_add(1, flat_idx_exp, contributions)
        return A_flat.reshape(B, N, self.n_types, self.n_max, self.n_sph)

    def _embed(self, A, types):
        """Joint (n_types, n_max) → embed_dim contraction per central atom type.

        Args:
            A:     (n_atoms, n_types, n_max, n_sph)
            types: (n_atoms,) central atom type indices

        Returns:
            A_emb: (n_atoms, embed_dim, n_sph)
        """
        W_i = self.W[types]  # (n_atoms, n_types, n_max, embed_dim)
        return torch.einsum('itns,itnc->ics', A, W_i)

    def _apply_dist_embed(self, A_cos, A_sin, dist_ij):
        """Distance-conditioned per-channel scale on A_cos/A_sin."""
        f = radial_basis(dist_ij, self.r_cut_edge, self.n_dist_embed,
                         cutoff_type=self.cutoff_type)              # (n_edges, n_dist_embed)
        scale = (f @ self.V_dist).unsqueeze(-1)                    # (n_edges, C, 1)
        A_cos = A_cos * (1 + scale)
        A_sin = A_sin * (1 + scale)
        return A_cos, A_sin

    def _contract(self, A_cos, A_sin):
        """Extract m=0 invariants: (n_edges, n_features_per_m, n_angular) → (n_edges, n_features_per_m)."""
        return A_cos[:, :, 0]

    def _apply_output(self, invariants, dist_ij, type_i=None, type_j=None,
                      type_idx=None):
        """output_net(inv) → per-edge energies.

        n_max_d=None: the readout emits a single number per edge, multiplied by
        the cutoff envelope f(r) so the per-edge energy still decays smoothly to
        0 at r_cut_edge (continuous energy/forces) without an explicit radial
        basis — i.e. energy_edge = MLP(inv) · f(r_ij). The n_max_d>=1 path
        instead dots the MLP output with the (cutoff-enveloped) radial basis."""
        out_kw = dict(type_i=type_i, type_j=type_j, type_idx=type_idx)
        if self.n_max_d is not None:
            rij_basis = radial_basis(dist_ij, self.r_cut_edge, self.n_max_d,
                                     cutoff_type=self.cutoff_type)
            return (self.output_net(invariants, **out_kw) * rij_basis).sum(-1)
        env = get_cutoff_fn(self.cutoff_type)(dist_ij, self.r_cut_edge)   # (n_e,) smooth → 0 at r_cut
        return self.output_net(invariants, **out_kw).squeeze(-1) * env


    def _pack_sph(self, A_cos, A_sin):
        """Pack (n_edges, n_ch, n_angular) back to (n_edges, n_ch, n_sph).

        Inverse of SphToAngular: scatters A_cos (m≥0) and A_sin (m<0) back
        to their SH indices using the precomputed index buffers.
        """
        n_e = A_cos.shape[0]
        cos_idx = self.sph_to_angular.cos_idx          # (n_ch, n_angular)
        sin_idx = self.sph_to_angular.sin_idx
        cos_valid = self.sph_to_angular.cos_valid
        sin_valid = self.sph_to_angular.sin_valid
        h = torch.zeros(n_e, self.n_features_per_m, self.n_sph,
                        device=A_cos.device, dtype=A_cos.dtype)
        h = h.scatter_add(2, cos_idx[None].expand(n_e, -1, -1), A_cos * cos_valid)
        h = h.scatter_add(2, sin_idx[None].expand(n_e, -1, -1), A_sin * sin_valid)
        return h  # (n_edges, n_ch, n_sph)

    def _aggregate_lr_embeddings(self, A_cos, A_sin, r_hat, edge_j, n_atoms):
        """Aggregate edge features to per-atom (l0, l1) equivariant embeddings
        (exposed via return_embeddings; e.g. for downstream long-range terms).

        Avoids the full Wigner T rotation by:
          1. Pack A_cos/A_sin → full SH in bond frame  (E, n_ch, n_sph)
          2. Sum over the l'-expansion axis first       (E, 2*embed_dim, n_sph)
          3. l=0: D^0=1, rotation-invariant — take directly
          4. l=1: apply D^1_T (3×3) to get global frame — much cheaper than full D^l_max
          5. Scatter-sum to atoms

        Returns:
            l0: (n_atoms, 2*embed_dim)     per-atom invariant scalar embeddings
            l1: (n_atoms, 2*embed_dim, 3)  per-atom equivariant vector embeddings
        """
        device, dtype = A_cos.device, A_cos.dtype
        n_e = A_cos.shape[0]
        n_base = 2 * self.embed_dim

        h = self._pack_sph(A_cos, A_sin)                            # (E, n_ch, n_sph)

        # Sum over l'-expansion axis first (rotation is linear, sum commutes with D^T)
        h_sum = (h.view(n_e, n_base, self.l_max + 1, self.n_sph)
                  .sum(dim=2))                                       # (E, 2*embed_dim, n_sph)

        # l=0: D^0 = 1, no rotation needed
        h_l0 = h_sum[:, :, 0]                                       # (E, 2*embed_dim)

        # l=1: apply D^1_T (3×3) — unrotate bond-frame l=1 to global frame
        # forward rotation: A_rot = A @ D  →  unrotate: h_global = h_bond @ D^T
        # einsum: h_global[e,c,n] = Σ_m h_bond[e,c,m] * D[e,n,m]
        D1 = build_D1_from_rhat(r_hat)                              # (E, 3, 3)
        h_l1 = torch.einsum('ecm,enm->ecn', h_sum[:, :, 1:4], D1)  # (E, 2*embed_dim, 3)

        idx_j0 = edge_j[:, None].expand_as(h_l0)
        idx_j1 = edge_j[:, None, None].expand_as(h_l1)

        l0 = torch.zeros(n_atoms, n_base, device=device, dtype=dtype
                         ).scatter_add(0, idx_j0, h_l0)
        l1 = torch.zeros(n_atoms, n_base, 3, device=device, dtype=dtype
                         ).scatter_add(0, idx_j1, h_l1)
        return l0, l1

    def _aggregate_node_sph(self, A_cos, A_sin, r_hat, edge_j, n_atoms, D_block=None):
        """Aggregate edge features to per-atom spherical embeddings (global frame).

        Full-l counterpart of _aggregate_lr_embeddings: instead of extracting only
        l=0 (l0) and l=1 (l1), it unrotates *every* l with the Wigner D-block and
        scatters to atoms, yielding the per-node spherical tensor (exposed via
        return_node_sph).

          1. Pack A_cos/A_sin → full SH in the bond frame   (E, n_ch, n_sph)
          2. Sum over the l-expansion axis                  (E, 2*embed_dim, n_sph)
          3. Unrotate to the global frame: h_global = h_bond @ D^T (forward is A @ D)
          4. Scatter-sum to atoms

        Returns:
            node_sph: (n_atoms, 2*embed_dim, n_sph) global-frame node embeddings
        """
        device, dtype = A_cos.device, A_cos.dtype
        n_e = A_cos.shape[0]
        n_base = 2 * self.embed_dim

        h = self._pack_sph(A_cos, A_sin)                            # (E, n_ch, n_sph)
        h_sum = (h.view(n_e, n_base, self.l_max + 1, self.n_sph)
                  .sum(dim=2))                                       # (E, 2*embed_dim, n_sph)

        if D_block is None:
            D_block = build_D_block(r_hat, self.l_max)
        h_global = torch.bmm(h_sum, D_block.transpose(-1, -2).contiguous())  # (E, n_base, n_sph)

        idx = edge_j[:, None, None].expand_as(h_global)
        node_sph = torch.zeros(n_atoms, n_base, self.n_sph, device=device, dtype=dtype
                               ).scatter_add(0, idx, h_global)
        return node_sph

    def _run_equivariant_layers(self, A_cos, A_sin, **kwargs):
        """Run the equivariant layers, interleaving a message-passing layer
        between consecutive stages when n_mp >= 2 (n_mp-1 MP layers, no trailing MP)."""
        type_i   = kwargs.get('type_i')
        type_j   = kwargs.get('type_j')
        type_idx = kwargs.get('type_idx')
        if self.n_mp == 1:
            # Plain model: a flat list of equivariant layers, no message passing.
            for layer in self.layers:
                A_cos, A_sin = layer(A_cos, A_sin, type_i=type_i, type_j=type_j,
                                     type_idx=type_idx)
            return A_cos, A_sin
        # Message-passing path: stage, MP, stage, MP, ..., stage  (MP only between stages).
        r_hat   = kwargs.get('r_hat')
        edge_i  = kwargs.get('edge_i')
        edge_j  = kwargs.get('edge_j')
        dist_ij = kwargs.get('dist_ij')
        n_atoms = kwargs.get('n_atoms')
        D_block = kwargs.get('D_block')
        for gi, stage in enumerate(self.layers):
            for layer in stage:
                A_cos, A_sin = layer(A_cos, A_sin, type_i=type_i, type_j=type_j,
                                     type_idx=type_idx)
            if gi < len(self.mp_layers):          # no MP after the final stage
                A_cos, A_sin = self.mp_layers[gi](
                    A_cos, A_sin, r_hat, dist_ij, edge_i, edge_j,
                    n_atoms, type_i, type_j,
                    D_block=D_block)
        return A_cos, A_sin

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, positions: torch.Tensor, types: torch.Tensor,
                return_embeddings: bool = False, return_node_sph: bool = False):
        """Compute total energy, and optionally per-atom embeddings.

        Args:
            positions:         (n_atoms, 3)
            types:             (n_atoms,) int tensor of atom-type indices
            return_embeddings: if True, also return per-atom (l0, l1) equivariant
                               embeddings for downstream use (e.g. long-range terms)
            return_node_sph:   if True, also return the per-atom node_sph features

        Returns:
            energy                          if neither flag is set
            (energy, l0, l1)               if return_embeddings is True
              l0: (N, 2*embed_dim)
              l1: (N, 2*embed_dim, 3)
            (energy, node_sph)             if return_node_sph is True
              node_sph: (N, 2*embed_dim, n_sph)
        """
        device, dtype = positions.device, positions.dtype

        # ── Edges ─────────────────────────────────────────────────────────
        edge_i_undir, edge_j_undir = find_edges(positions, self.r_cut_edge)
        if len(edge_i_undir) == 0:
            energy = torch.zeros(1, device=device, dtype=dtype).squeeze()
            N = len(types)
            if return_node_sph:
                node_sph = torch.zeros(N, 2 * self.embed_dim, self.n_sph,
                                       device=device, dtype=dtype)
                return energy, node_sph
            if return_embeddings:
                l0 = torch.zeros(N, 2 * self.embed_dim, device=device, dtype=dtype)
                l1 = torch.zeros(N, 2 * self.embed_dim, 3, device=device, dtype=dtype)
                return energy, l0, l1
            return energy

        edge_i = torch.cat([edge_i_undir, edge_j_undir])
        edge_j = torch.cat([edge_j_undir, edge_i_undir])

        diff_ij = positions[edge_j] - positions[edge_i]
        dist_ij = torch.sqrt((diff_ij ** 2).sum(-1) + 1e-30)
        r_hat   = diff_ij / dist_ij.unsqueeze(-1)

        # ── Neighbor list ─────────────────────────────────────────────────
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        dist_mat = torch.sqrt((diff ** 2).sum(-1) + 1e-30)
        nb_mask = (dist_mat < self.r_cut_neighbor) & (dist_mat > 1e-10)
        nb_src, nb_dst = nb_mask.nonzero(as_tuple=True)

        # ── Step 1: ACE atomic basis ───────────────────────────────────────
        pos_batch = positions.unsqueeze(0)   # (1, N, 3)
        A_batch = self._compute_ace_basis(pos_batch, nb_src, nb_dst, types)
        A = A_batch.squeeze(0)               # (N, n_types, n_max, n_sph)

        # ── Step 2: Joint contraction → (N, embed_dim, n_sph) ────────────
        A_emb = self._embed(A, types)

        # ── Step 3: Gather + Wigner rotation ──────────────────────────────
        type_i = types[edge_i]
        type_j = types[edge_j]
        A_src = A_emb[edge_i]   # (n_edges, embed_dim, n_sph)
        A_tgt = A_emb[edge_j]
        A_both = torch.cat([A_src, A_tgt], dim=1)   # (n_edges, 2*embed_dim, n_sph)
        D_block = build_D_block(r_hat, self.l_max)
        A_rot = wigner_rotate(A_both, D_block)  # (n_edges, 2*embed_dim, n_sph)

        # ── Step 4: Reshape to A_cos / A_sin ──────────────────────────────
        A_cos, A_sin = self.sph_to_angular(A_rot)

        # ── Step 4b: Distance-conditioned edge transform (optional) ───────
        if self.n_dist_embed > 0:
            A_cos, A_sin = self._apply_dist_embed(A_cos, A_sin, dist_ij)

        # ── Step 5: Equivariant layers ────────────────────────────────────
        ti, tj = types[edge_i], types[edge_j]
        type_idx = (precompute_type_idx_2pass(ti, tj, self.n_types)
                    if (self.edge_type_linear or self.edge_type_output) else None)
        A_cos, A_sin = self._run_equivariant_layers(
            A_cos, A_sin,
            r_hat=r_hat, edge_i=edge_i, edge_j=edge_j,
            dist_ij=dist_ij, n_atoms=len(types),
            type_i=ti, type_j=tj, type_idx=type_idx, D_block=D_block)

        # ── Step 6+7: m=0 invariants → output_net → dot(rij_basis) ──────────
        invariants = self._contract(A_cos, A_sin)   # (n_edges, n_features_per_m)
        per_edge_energy = self._apply_output(invariants, dist_ij, type_i, type_j, type_idx=type_idx)
        energy = per_edge_energy.sum() + self.atomic_energy[types].sum()

        if return_node_sph:
            node_sph = self._aggregate_node_sph(
                A_cos, A_sin, r_hat, edge_j, len(types), D_block)
            return energy, node_sph
        if return_embeddings:
            l0, l1 = self._aggregate_lr_embeddings(
                A_cos, A_sin, r_hat, edge_j, len(types))
            return energy, l0, l1
        return energy

    def forward_pbc(self, positions: torch.Tensor, types: torch.Tensor,
                    edge_i: torch.Tensor, edge_j: torch.Tensor,
                    shift_vecs_edge: torch.Tensor,
                    nb_src: torch.Tensor, nb_dst: torch.Tensor,
                    shift_vecs_nb: torch.Tensor,
                    return_embeddings: bool = False, return_node_sph: bool = False):
        """Compute total energy with periodic boundary conditions.

        Args:
            positions:         (N, 3) atom positions in Cartesian Å (wrapped to unit cell)
            types:             (N,) int tensor of atom-type indices
            edge_i, edge_j:    (n_edges,) directed edge indices (both i→j and j→i)
            shift_vecs_edge:   (n_edges, 3) Cartesian PBC shift vectors for edges
            nb_src, nb_dst:    (n_nb,) directed neighbor pair indices
            shift_vecs_nb:     (n_nb, 3) Cartesian PBC shift vectors for neighbors
            return_embeddings: if True, also return per-atom (l0, l1) embeddings

        Returns:
            energy                  if return_embeddings is False
            (energy, l0, l1)        if return_embeddings is True
              l0: (N, 2*embed_dim)
              l1: (N, 2*embed_dim, 3)
        """
        device, dtype = positions.device, positions.dtype
        n_edges = len(edge_i)

        if n_edges == 0:
            energy = torch.zeros(1, device=device, dtype=dtype).squeeze()
            N = len(types)
            if return_node_sph:
                return energy, torch.zeros(N, 2 * self.embed_dim, self.n_sph,
                                           device=device, dtype=dtype)
            if return_embeddings:
                return (energy,
                        torch.zeros(N, 2 * self.embed_dim, device=device, dtype=dtype),
                        torch.zeros(N, 2 * self.embed_dim, 3, device=device, dtype=dtype))
            return energy

        # ── Edges with PBC shifts ──────────────────────────────────────────
        diff_ij = (positions[edge_j] - positions[edge_i]
                   + shift_vecs_edge.to(dtype=dtype))
        dist_ij = torch.sqrt((diff_ij ** 2).sum(-1) + 1e-30)
        r_hat   = diff_ij / dist_ij.unsqueeze(-1)

        # ── Step 1: ACE atomic basis with PBC neighbor shifts ─────────────
        pos_batch = positions.unsqueeze(0)   # (1, N, 3)
        A_batch = self._compute_ace_basis(pos_batch, nb_src, nb_dst, types,
                                          shift_vecs_nb=shift_vecs_nb)
        A = A_batch.squeeze(0)               # (N, n_types, n_max, n_sph)

        # ── Steps 2–7: identical to forward() ─────────────────────────────
        A_emb = self._embed(A, types)

        A_src  = A_emb[edge_i]
        A_tgt  = A_emb[edge_j]
        A_both = torch.cat([A_src, A_tgt], dim=1)
        D_block = build_D_block(r_hat, self.l_max)
        A_rot  = wigner_rotate(A_both, D_block)

        A_cos, A_sin = self.sph_to_angular(A_rot)

        if self.n_dist_embed > 0:
            A_cos, A_sin = self._apply_dist_embed(A_cos, A_sin, dist_ij)

        ti, tj = types[edge_i], types[edge_j]
        type_idx = (precompute_type_idx_2pass(ti, tj, self.n_types)
                    if (self.edge_type_linear or self.edge_type_output) else None)
        A_cos, A_sin = self._run_equivariant_layers(
            A_cos, A_sin,
            r_hat=r_hat, edge_i=edge_i, edge_j=edge_j,
            dist_ij=dist_ij, n_atoms=len(types),
            type_i=ti, type_j=tj, type_idx=type_idx, D_block=D_block)

        invariants = self._contract(A_cos, A_sin)
        per_edge_energy = self._apply_output(invariants, dist_ij, ti, tj, type_idx=type_idx)
        energy = per_edge_energy.sum() + self.atomic_energy[types].sum()

        if return_node_sph:
            node_sph = self._aggregate_node_sph(
                A_cos, A_sin, r_hat, edge_j, len(types), D_block)
            return energy, node_sph
        if return_embeddings:
            l0, l1 = self._aggregate_lr_embeddings(
                A_cos, A_sin, r_hat, edge_j, len(types))
            return energy, l0, l1
        return energy

    def forward_batch_multi(self, positions_list, types_list,
                            return_embeddings=False, return_node_sph=False):
        """Batch forward for variable-size, variable-composition structures.

        Topology is built per-structure in a cheap Python loop; the expensive
        ops (Wigner rotation, equivariant layers, output MLP) run once on the
        full flat edge set.

        Args:
            positions_list:  list of B tensors, each (N_b, 3)
            types_list:      list of B tensors, each (N_b,) of type indices
            return_embeddings: if True, also return per-atom (l0_list, l1_list) embeddings

        Returns:
            energies                          if return_embeddings is False
            (energies, l0_list, l1_list)      if return_embeddings is True
              l0_list: list of B (N_b, 2*embed_dim) tensors
              l1_list: list of B (N_b, 2*embed_dim, 3) tensors
        """
        B = len(positions_list)
        device = positions_list[0].device
        dtype  = positions_list[0].dtype

        A_src_list, A_tgt_list = [], []
        r_hat_list, dist_ij_list = [], []
        type_i_list, type_j_list = [], []
        edge_i_list, edge_j_list = [], []   # flat atom indices with offsets (for MP)
        struct_ids = []
        atomic_e_list = []
        atom_offset = 0
        atom_counts = []   # N_b per structure, for slicing embeddings

        for b, (pos, types) in enumerate(zip(positions_list, types_list)):
            N_b = pos.shape[0]
            diff = pos.unsqueeze(0) - pos.unsqueeze(1)              # (N_b, N_b, 3)
            dist_mat = torch.sqrt((diff ** 2).sum(-1) + 1e-30)      # (N_b, N_b)

            ei, ej = ((dist_mat < self.r_cut_edge) & (dist_mat > 1e-10)).nonzero(as_tuple=True)

            atomic_e_list.append(self.atomic_energy[types].sum())
            if len(ei) == 0:
                atom_offset += N_b
                continue

            nb_src, nb_dst = ((dist_mat < self.r_cut_neighbor) & (dist_mat > 1e-10)).nonzero(as_tuple=True)

            diff_ij = pos[ej] - pos[ei]
            dist_ij = torch.sqrt((diff_ij ** 2).sum(-1) + 1e-30)
            r_hat   = diff_ij / dist_ij.unsqueeze(-1)

            A = self._compute_ace_basis(pos.unsqueeze(0), nb_src, nb_dst, types).squeeze(0)
            A_emb = self._embed(A, types)   # (N_b, embed_dim, n_sph)

            A_src_list.append(A_emb[ei])
            A_tgt_list.append(A_emb[ej])
            r_hat_list.append(r_hat)
            dist_ij_list.append(dist_ij)
            type_i_list.append(types[ei])
            type_j_list.append(types[ej])
            edge_i_list.append(ei + atom_offset)
            edge_j_list.append(ej + atom_offset)
            struct_ids.append(torch.full((len(ei),), b, dtype=torch.long, device=device))
            atom_offset += N_b
            atom_counts.append(N_b)

        energies = torch.stack(atomic_e_list)   # (B,)

        total_edges = sum(len(x) for x in r_hat_list)
        if total_edges == 0:
            n_ch = 2 * self.embed_dim
            if return_node_sph:
                node_sph_list = [torch.zeros(c, n_ch, self.n_sph, dtype=dtype, device=device)
                                 for c in atom_counts]
                return energies, node_sph_list
            if return_embeddings:
                l0_list = [torch.zeros(c, n_ch, dtype=dtype, device=device)
                           for c in atom_counts]
                l1_list = [torch.zeros(c, n_ch, 3, dtype=dtype, device=device)
                           for c in atom_counts]
                return energies, l0_list, l1_list
            return energies

        # Merge flat edge arrays
        A_src      = torch.cat(A_src_list)
        A_tgt      = torch.cat(A_tgt_list)
        r_hat      = torch.cat(r_hat_list)
        dist_ij    = torch.cat(dist_ij_list)
        type_i     = torch.cat(type_i_list)
        type_j     = torch.cat(type_j_list)
        edge_i_flat = torch.cat(edge_i_list)
        edge_j_flat = torch.cat(edge_j_list)
        struct_idx  = torch.cat(struct_ids)

        A_both  = torch.cat([A_src, A_tgt], dim=1)
        D_block = build_D_block(r_hat, self.l_max)
        A_rot  = wigner_rotate(A_both, D_block)

        A_cos, A_sin = self.sph_to_angular(A_rot)

        if self.n_dist_embed > 0:
            A_cos, A_sin = self._apply_dist_embed(A_cos, A_sin, dist_ij)

        type_idx = (precompute_type_idx_2pass(type_i, type_j, self.n_types)
                    if (self.edge_type_linear or self.edge_type_output) else None)
        A_cos, A_sin = self._run_equivariant_layers(
            A_cos, A_sin,
            r_hat=r_hat, edge_i=edge_i_flat, edge_j=edge_j_flat,
            dist_ij=dist_ij, n_atoms=atom_offset,
            type_i=type_i, type_j=type_j, type_idx=type_idx, D_block=D_block)

        invariants = self._contract(A_cos, A_sin)
        per_edge_energy = self._apply_output(invariants, dist_ij, type_i, type_j,
                                              type_idx=type_idx)

        energies = energies + torch.zeros(B, dtype=dtype, device=device).scatter_add(
            0, struct_idx, per_edge_energy)

        if return_node_sph:
            node_sph_flat = self._aggregate_node_sph(
                A_cos, A_sin, r_hat, edge_j_flat, atom_offset, D_block)
            node_sph_list = []
            offset = 0
            for N_b in atom_counts:
                node_sph_list.append(node_sph_flat[offset:offset + N_b])
                offset += N_b
            return energies, node_sph_list
        if return_embeddings:
            l0_flat, l1_flat = self._aggregate_lr_embeddings(
                A_cos, A_sin, r_hat, edge_j_flat, atom_offset)
            # Slice per structure using atom offsets
            l0_list, l1_list = [], []
            offset = 0
            for N_b in atom_counts:
                l0_list.append(l0_flat[offset:offset + N_b])
                l1_list.append(l1_flat[offset:offset + N_b])
                offset += N_b
            return energies, l0_list, l1_list
        return energies

    def forward_batch(self, positions_list, types, topology=None,
                      return_embeddings=False, return_node_sph=False):
        """Compute energies for a batch of structures sharing the same atom types.

        Args:
            positions_list: list of B (N, 3) tensors
            types:          (N,) int tensor of atom-type indices (same for all structures)
            topology:       dict with precomputed 'edge_i', 'edge_j', 'nb_src', 'nb_dst'
                            (and optionally 'shift_vecs_edge', 'shift_vecs_nb' for PBC)
                            for the fixed-topology (same molecule) case, or None to
                            fall back to per-structure self.forward calls.

        Returns:
            energies: (B,) tensor
        """
        if not isinstance(topology, dict):
            # Variable-topology fallback: forward_batch_multi subsumes this case
            # (shared types is just every structure carrying the same type
            # vector). It builds topology per-structure and runs the expensive
            # ops once on the merged flat edge set — identical result.
            return self.forward_batch_multi(
                positions_list, [types] * len(positions_list),
                return_embeddings=return_embeddings,
                return_node_sph=return_node_sph)

        # ── Fixed topology: vectorized over B ─────────────────────────────
        B = len(positions_list)
        edge_i = topology['edge_i']
        edge_j = topology['edge_j']
        nb_src = topology['nb_src']
        nb_dst = topology['nb_dst']
        shift_vecs_edge = topology.get('shift_vecs_edge', None)
        shift_vecs_nb   = topology.get('shift_vecs_nb',   None)
        n_edges = edge_i.shape[0]

        pos_batch = torch.stack(positions_list)  # (B, N, 3)

        # ── Edges ────────────────────────────────────────────────────────
        diff_ij = pos_batch[:, edge_j] - pos_batch[:, edge_i]  # (B, n_edges, 3)
        if shift_vecs_edge is not None:
            diff_ij = diff_ij + shift_vecs_edge[None].to(dtype=pos_batch.dtype)
        dist_ij = torch.sqrt((diff_ij ** 2).sum(-1) + 1e-30)   # (B, n_edges)
        r_hat   = diff_ij / dist_ij.unsqueeze(-1)               # (B, n_edges, 3)

        # ── Step 1: ACE atomic basis (B, N, n_types, n_max, n_sph) ──────
        A_batch = self._compute_ace_basis(pos_batch, nb_src, nb_dst, types, shift_vecs_nb)

        # ── Step 2: Joint contraction → (B, N, embed_dim, n_sph) ────────
        W_i   = self.W[types]  # (N, n_types, n_max, embed_dim)
        A_emb = torch.einsum('bitns,itnc->bics', A_batch, W_i)

        # ── Step 3: Gather + Wigner rotation (flatten B*n_edges) ────────
        type_i = types[edge_i]   # (n_edges,)
        type_j = types[edge_j]
        A_src  = A_emb[:, edge_i]                                # (B, n_edges, embed_dim, n_sph)
        A_tgt  = A_emb[:, edge_j]
        A_both = torch.cat([A_src, A_tgt], dim=2)               # (B, n_edges, 2*embed_dim, n_sph)

        r_hat_flat  = r_hat.reshape(B * n_edges, 3)
        A_both_flat = A_both.reshape(B * n_edges, 2 * self.embed_dim, self.n_sph)
        D_block = build_D_block(r_hat_flat, self.l_max)
        A_rot_flat  = wigner_rotate(A_both_flat, D_block)

        # ── Step 4: Reshape to A_cos / A_sin ─────────────────────────────
        A_cos_flat, A_sin_flat = self.sph_to_angular(A_rot_flat)
        # shapes: (B*n_edges, n_features_per_m, n_angular)

        # ── Step 4b: Distance-conditioned edge transform (optional) ───────
        if self.n_dist_embed > 0:
            dist_flat = dist_ij.reshape(B * n_edges)
            A_cos_flat, A_sin_flat = self._apply_dist_embed(A_cos_flat, A_sin_flat, dist_flat)

        # ── Step 5: Equivariant layers ────────────────────────────────────
        # For batched MP: offset edge indices so scatter targets B*N atoms
        N = pos_batch.shape[1]
        offset = torch.arange(B, device=edge_i.device).repeat_interleave(n_edges) * N
        edge_i_flat = edge_i.repeat(B) + offset
        edge_j_flat = edge_j.repeat(B) + offset
        type_i_flat = type_i.repeat(B)
        type_j_flat = type_j.repeat(B)

        type_idx = (precompute_type_idx_2pass(type_i_flat, type_j_flat, self.n_types)
                    if (self.edge_type_linear or self.edge_type_output) else None)
        A_cos_flat, A_sin_flat = self._run_equivariant_layers(
            A_cos_flat, A_sin_flat,
            r_hat=r_hat_flat, edge_i=edge_i_flat, edge_j=edge_j_flat,
            dist_ij=dist_ij.reshape(B * n_edges), n_atoms=B * N,
            type_i=type_i_flat, type_j=type_j_flat, type_idx=type_idx, D_block=D_block)

        # ── Step 6+7: m=0 invariants → output_net → dot(rij_basis) ──────────
        invariants = self._contract(A_cos_flat, A_sin_flat)      # (B*n_edges, n_features_per_m)
        per_edge_energy = self._apply_output(invariants, dist_ij.reshape(B * n_edges),
                                             type_i_flat, type_j_flat, type_idx=type_idx)  # (B*n_edges,)
        energies = per_edge_energy.reshape(B, n_edges).sum(dim=1)        # (B,)
        energies = energies + self.atomic_energy[types].sum()

        if return_node_sph:
            node_sph_flat = self._aggregate_node_sph(
                A_cos_flat, A_sin_flat, r_hat_flat, edge_j_flat, B * N, D_block)
            node_sph_list = [node_sph_flat[b * N:(b + 1) * N] for b in range(B)]
            return energies, node_sph_list
        if return_embeddings:
            l0_flat, l1_flat = self._aggregate_lr_embeddings(
                A_cos_flat, A_sin_flat, r_hat_flat, edge_j_flat, B * N)
            l0_list = [l0_flat[b * N:(b + 1) * N] for b in range(B)]
            l1_list = [l1_flat[b * N:(b + 1) * N] for b in range(B)]
            return energies, l0_list, l1_list
        return energies


# ---------------------------------------------------------------------------
# Equivariant layer: Linear → RealSpaceNonlinearity → residual
# ---------------------------------------------------------------------------


class ECENetLayer(nn.Module):
    """One equivariant layer: EquivariantLinear → nonlinearity.

    linear(n_ch → n_ch) → nonlin(n_ch) → residual. The linear supports
    type-dependent weights (n_types_linear) or template mixing.

    Args:
        n_features:        number of feature channels (= n_features_per_m)
        m_max:             maximum angular frequency (= l_max)
        activation:        pointwise activation (used by the realspace nonlinearity)
        use_nonlinearity:  if False, skip nonlinearity entirely (linear-only layer)
    """

    def __init__(self, n_features: int, m_max: int, activation: str = 'silu',
                 use_nonlinearity: bool = True, n_grid: int = None,
                 n_types: int = None,
                 n_types_linear: int = None):
        super().__init__()
        n_angular = m_max + 1
        # nonlin_features: dimension at which the nonlinearity operates
        nonlin_features = n_features

        self.linear = EquivariantLinear(n_features, n_features, n_angular, m_max,
                                        n_types=n_types_linear)

        self.nonlin = None
        if use_nonlinearity:
            self.nonlin = RealSpaceNonlinearity(nonlin_features, m_max, n_grid=n_grid,
                                                activation=activation,
                                                n_types=n_types)
        self.use_nonlinearity = self.nonlin is not None
        self.edge_type_nonlin = (
            self.nonlin is not None
            and isinstance(self.nonlin, RealSpaceNonlinearity)
            and self.nonlin.edge_type_nonlin
        )
    def forward(self, A_cos, A_sin, type_i=None, type_j=None, type_idx=None):
        A_cos_in, A_sin_in = A_cos, A_sin

        A_cos, A_sin = self.linear(A_cos, A_sin, type_i=type_i, type_j=type_j,
                                   type_idx=type_idx)
        if self.nonlin is not None:
            if self.edge_type_nonlin:
                A_cos, A_sin = self.nonlin(A_cos, A_sin, type_i=type_i, type_j=type_j)
            else:
                A_cos, A_sin = self.nonlin(A_cos, A_sin)

        return A_cos + A_cos_in, A_sin + A_sin_in


# ---------------------------------------------------------------------------
# Message passing layer
# ---------------------------------------------------------------------------


class ECENetMPLayer(nn.Module):
    """Equivariant message passing for ECENet.

    For each atom j, aggregates full edge features from all incoming edges i→j,
    then adds the result to all outgoing edges j→k.

    Algorithm:
      1. Reshape A_cos/A_sin to (n_base, l_max+1, n_angular) — free view
      2. Per l: assemble m-ordered (2l+1) vector from cos/sin, apply D_l^T (unrotate to global)
         m>l components dropped here; no scatter to SH format needed
      3. Distance/type scalar weight over n_base channels
      4. Cat all l, scatter-sum to atoms → Delta (N, n_base, n_sph)
      5. Per l: apply D_l (rotate back), split m-ordered → cos/sin, add to A_cos/A_sin
    """

    def __init__(self, n_features_per_m: int, l_max: int, embed_dim: int,
                 n_types: int, n_dist_basis: int = 8, r_cut: float = 5.0,
                 cutoff_type: str = 'cosine', m_max: int = None):
        super().__init__()
        self.l_max     = l_max
        self.n_sph     = (l_max + 1) ** 2
        self.m_max     = m_max if m_max is not None else l_max
        self.n_angular = self.m_max + 1   # pack/unpack only m=0..m_max; rest are zero
        self.n_ch      = n_features_per_m
        self.n_base    = n_features_per_m // (l_max + 1)  # = 2*embed_dim; channels before l-expansion
        self.r_cut     = r_cut
        self.cutoff_type  = cutoff_type
        self.n_dist_basis = n_dist_basis

        # l_of_c[c] = c % (l_max+1) for ECENet channel layout
        l_of_c = torch.arange(n_features_per_m, dtype=torch.long) % (l_max + 1)

        # Pack/unpack indices: (n_ch, n_angular)
        pack_cos_idx   = torch.zeros(n_features_per_m, self.n_angular, dtype=torch.long)
        pack_sin_idx   = torch.zeros(n_features_per_m, self.n_angular, dtype=torch.long)
        pack_cos_valid = torch.zeros(n_features_per_m, self.n_angular, dtype=torch.bool)
        pack_sin_valid = torch.zeros(n_features_per_m, self.n_angular, dtype=torch.bool)
        for c in range(n_features_per_m):
            lp = int(l_of_c[c].item())
            for m in range(self.n_angular):
                if m <= lp:
                    pack_cos_idx[c, m]   = lp * lp + lp + m
                    pack_cos_valid[c, m] = True
                    if m > 0:
                        pack_sin_idx[c, m]   = lp * lp + lp - m
                        pack_sin_valid[c, m] = True
        self.register_buffer('pack_cos_idx',     pack_cos_idx)
        self.register_buffer('pack_sin_idx',     pack_sin_idx)
        self.register_buffer('pack_cos_valid_f', pack_cos_valid.float())
        self.register_buffer('pack_sin_valid_f', pack_sin_valid.float())

        # Per-channel distance/type weight over n_base = 2*embed_dim compact channels.
        # (l_max+1) l-channels share one weight; rotation commutes with the l-sum.)
        # w[e] is a per-pair linear map f_d (n_dist_basis) → (n_base); zero at init
        # → MP starts as a no-op. No bias (the cutoff envelope lives in f_d).
        self.W_msg = nn.Parameter(
            torch.zeros(n_types, n_types, n_dist_basis, self.n_base)
        )

    def _pack(self, A_cos, A_sin):
        """(n_edges, n_ch, n_angular) → (n_edges, n_ch, n_sph)."""
        n_e = A_cos.shape[0]
        idx_cos = self.pack_cos_idx[None].expand(n_e, -1, -1)
        idx_sin = self.pack_sin_idx[None].expand(n_e, -1, -1)
        h = (torch.zeros(n_e, self.n_ch, self.n_sph, device=A_cos.device, dtype=A_cos.dtype)
             .scatter_add(2, idx_cos, A_cos * self.pack_cos_valid_f)
             .scatter_add(2, idx_sin, A_sin * self.pack_sin_valid_f))
        return h

    def _unpack(self, h):
        """(n_edges, n_ch, n_sph) → (n_edges, n_ch, n_angular)."""
        n_e = h.shape[0]
        idx_cos = self.pack_cos_idx[None].expand(n_e, -1, -1)
        idx_sin = self.pack_sin_idx[None].expand(n_e, -1, -1)
        A_cos = torch.gather(h, 2, idx_cos) * self.pack_cos_valid_f
        A_sin = torch.gather(h, 2, idx_sin) * self.pack_sin_valid_f
        return A_cos, A_sin

    def forward(self, A_cos, A_sin, r_hat, dist_ij, edge_i, edge_j,
                n_atoms, type_i, type_j, D_block=None):
        n_e = A_cos.shape[0]
        device, dtype = A_cos.device, A_cos.dtype
        lp1 = self.l_max + 1

        # 1. Reshape to (n_e, n_base, l_max+1, n_angular) — free view
        Ac = A_cos.view(n_e, self.n_base, lp1, self.n_angular)
        As = A_sin.view(n_e, self.n_base, lp1, self.n_angular)

        if D_block is None:
            D_block = build_D_block(r_hat, self.l_max)

        # 2. Pack cos/sin → (n_e, n_base, n_sph); zeros where m > m_max
        h = torch.zeros(n_e, self.n_base, self.n_sph, device=device, dtype=dtype)
        for l in range(lp1):
            m_out = min(l, self.m_max)
            h[:, :, l*l + l : l*l + l + m_out + 1] = Ac[:, :, l, :m_out + 1]
            if m_out > 0:
                h[:, :, l*l + l - m_out : l*l + l] = As[:, :, l, 1:m_out + 1].flip(-1)

        # 3. Weight: w[e, c_base] = Σ_n f_n(r_ij) * W[t_i, t_j, n, c_base]
        f_d = radial_basis(dist_ij, self.r_cut, self.n_dist_basis, cutoff_type=self.cutoff_type)
        w = torch.einsum('en,enc->ec', f_d, self.W_msg[type_i, type_j])  # (n_e, n_base)

        # 4. Unrotate to global frame, weight, scatter-sum to atoms
        D_block_T = D_block.transpose(-1, -2).contiguous()
        h_global = torch.bmm(h, D_block_T) * w.unsqueeze(-1)
        idx   = edge_j[:, None, None].expand_as(h_global)
        Delta = torch.zeros(n_atoms, self.n_base, self.n_sph, device=device, dtype=dtype
                            ).scatter_add(0, idx, h_global)


        # 5. Rotate back into edge frame
        v_rot = torch.bmm(Delta[edge_i], D_block)              # (n_e, n_base, n_sph)

        # 6. Unpack → delta_cos/delta_sin, add to features
        delta_cos = torch.zeros(n_e, self.n_base, lp1, self.n_angular, device=device, dtype=dtype)
        delta_sin = torch.zeros(n_e, self.n_base, lp1, self.n_angular, device=device, dtype=dtype)
        for l in range(lp1):
            m_out = min(l, self.m_max)
            delta_cos[:, :, l, :m_out + 1] = v_rot[:, :, l*l + l : l*l + l + m_out + 1]
            if m_out > 0:
                delta_sin[:, :, l, 1:m_out + 1] = v_rot[:, :, l*l + l - m_out : l*l + l].flip(-1)

        return (A_cos + delta_cos.view(n_e, self.n_ch, self.n_angular),
                A_sin + delta_sin.view(n_e, self.n_ch, self.n_angular))


# ---------------------------------------------------------------------------
# SH → A_cos / A_sin reshape
# ---------------------------------------------------------------------------


class SphToAngular(nn.Module):
    """Convert rotated features (n_edges, 2*embed_dim, n_sph) to A_cos/A_sin.

    Reshapes n_sph = (l_max+1)² into an (l_max+1, 2*l_max+1) block indexed by
    (l, m), zero-padded where |m| > l, then merges the l axis into the channel
    dimension and separates m into cos (m>=0) and sin (m<0) components.

    Output shape: (n_edges, 2*embed_dim*(l_max+1), l_max+1)
      channel layout: [(side=0, embed=0, l=0), (side=0, embed=0, l=1), ...,
                       (side=0, embed=1, l=0), ..., (side=1, embed=embed_dim-1, l=l_max)]
      angular mode m = 0..l_max (the azimuthal frequency |m|).

    The triangular zero structure (|m| > l → 0) is preserved naturally.
    """

    def __init__(self, embed_dim: int, l_max: int, m_max: int = None):
        super().__init__()
        self.l_max = l_max
        m_max = m_max if m_max is not None else l_max
        self.m_max = m_max
        self.n_angular = m_max + 1          # m = 0..m_max (may be < l_max+1)
        self.n_ch = 2 * embed_dim * (l_max + 1)   # (side, embed, l) channels

        n_ch_base = 2 * embed_dim           # channels before l expansion

        # For each (embed_channel, l) and each m = 0..m_max, store the flat SH index
        # +m → index l²+l+m,  −m → index l²+l-m.
        # Channels with l < m have no valid component → index 0, masked to 0.
        # Only m=0..m_max are included; higher modes are discarded.
        cos_idx = torch.zeros(self.n_ch, self.n_angular, dtype=torch.long)
        sin_idx = torch.zeros(self.n_ch, self.n_angular, dtype=torch.long)
        cos_valid = torch.zeros(self.n_ch, self.n_angular)
        sin_valid = torch.zeros(self.n_ch, self.n_angular)

        c = 0
        for _ in range(n_ch_base):          # one entry per (side, embed_channel)
            for l in range(l_max + 1):
                base = l * l + l            # index of m=0 for this l
                for m in range(self.n_angular):
                    if m <= l:
                        cos_idx[c, m] = base + m    # +m component
                        cos_valid[c, m] = 1.0
                        if m > 0:
                            sin_idx[c, m] = base - m  # −m component
                            sin_valid[c, m] = 1.0
                c += 1

        self.register_buffer('cos_idx', cos_idx)
        self.register_buffer('sin_idx', sin_idx)
        self.register_buffer('cos_valid', cos_valid)
        self.register_buffer('sin_valid', sin_valid)

    def forward(self, A_rot):
        """
        Args:
            A_rot: (n_edges, 2*embed_dim, n_sph)
        Returns:
            A_cos, A_sin: (n_edges, 2*embed_dim*(l_max+1), l_max+1)
        """
        n_edges = A_rot.shape[0]
        # Repeat each embed channel l_max+1 times to align with (embed, l) layout
        A_exp = A_rot.repeat_interleave(self.l_max + 1, dim=1)  # (n_edges, n_ch, n_sph)
        # Gather cos (+m) and sin (−m) components
        A_cos = A_exp.gather(2, self.cos_idx[None].expand(n_edges, -1, -1)) * self.cos_valid
        A_sin = A_exp.gather(2, self.sin_idx[None].expand(n_edges, -1, -1)) * self.sin_valid
        return A_cos, A_sin


# ---------------------------------------------------------------------------
# Type-dependent output MLP
# ---------------------------------------------------------------------------


class TypeDepLinear(nn.Module):
    """Linear layer with factorized per-type additive weight correction.

    W[e] = weight + weights_src[type_i[e]] + weights_tgt[type_j[e]]

    Shared weight is randomly initialised; corrections are zero-initialised
    so training begins from the shared baseline.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 n_types: int = None):
        super().__init__()
        std = (2.0 / (in_features + out_features)) ** 0.5
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.edge_type = (n_types is not None)
        if self.edge_type:
            self.weights_src = nn.Parameter(torch.zeros(n_types, out_features, in_features))
            self.weights_tgt = nn.Parameter(torch.zeros(n_types, out_features, in_features))

    def forward(self, x, type_i=None, type_j=None, type_idx=None):
        out = x @ self.weight.T
        if self.edge_type and (type_idx is not None or type_i is not None):
            n_t = self.weights_src.shape[0]
            correction = x.new_zeros(x.shape[0], self.weight.shape[0])
            if type_idx is not None and len(type_idx) == 4:
                # 2-pass format
                src_perm, src_sizes, tgt_perm, tgt_sizes = type_idx
                offset = 0
                for ti, sz in enumerate(src_sizes):
                    if sz > 0:
                        idx = src_perm[offset:offset + sz]
                        correction = correction.index_add(0, idx, x[idx] @ self.weights_src[ti].T)
                    offset += sz
                offset = 0
                for tj, sz in enumerate(tgt_sizes):
                    if sz > 0:
                        idx = tgt_perm[offset:offset + sz]
                        correction = correction.index_add(0, idx, x[idx] @ self.weights_tgt[tj].T)
                    offset += sz
            else:
                if type_idx is not None:
                    pair_perm, pair_sizes = type_idx
                else:
                    pair_type  = type_i * n_t + type_j
                    pair_perm  = pair_type.argsort(stable=True)
                    pair_sizes = pair_type.bincount(minlength=n_t * n_t).tolist()
                offset = 0
                for pair, sz in enumerate(pair_sizes):
                    if sz > 0:
                        ti, tj = divmod(pair, n_t)
                        W_corr = self.weights_src[ti] + self.weights_tgt[tj]
                        idx = pair_perm[offset:offset + sz]
                        correction = correction.index_add(0, idx, x[idx] @ W_corr.T)
                    offset += sz
            out = out + correction
        if self.bias is not None:
            out = out + self.bias
        return out


class TypeDepMLP(nn.Module):
    """MLP whose linear layers support optional per-(type_i, type_j) weight corrections."""

    def __init__(self, dims: list, activation: nn.Module, n_types: int = None,
                 zero_init_last: bool = True):
        super().__init__()
        self.linears = nn.ModuleList([
            TypeDepLinear(dims[i], dims[i + 1], n_types=n_types)
            for i in range(len(dims) - 1)
        ])
        self.activation = activation
        if zero_init_last:
            nn.init.zeros_(self.linears[-1].bias)
            nn.init.normal_(self.linears[-1].weight, std=0.01)

    def forward(self, x, type_i=None, type_j=None, type_idx=None):
        for i, linear in enumerate(self.linears):
            x = linear(x, type_i=type_i, type_j=type_j, type_idx=type_idx)
            if i < len(self.linears) - 1:
                x = self.activation(x)
        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def precompute_type_idx_2pass(type_i, type_j, n_types):
    """Precompute src/tgt type groups for 2-pass equivariant linear.

    Returns (src_perm, src_sizes, tgt_perm, tgt_sizes) — 4-tuple.
    Pass 1: loop over src types (n_types groups, ~n_edges/n_types each).
    Pass 2: loop over tgt types (n_types groups, ~n_edges/n_types each).
    """
    src_perm  = type_i.argsort(stable=True)
    src_sizes = type_i.bincount(minlength=n_types).tolist()
    tgt_perm  = type_j.argsort(stable=True)
    tgt_sizes = type_j.bincount(minlength=n_types).tolist()
    return src_perm, src_sizes, tgt_perm, tgt_sizes
