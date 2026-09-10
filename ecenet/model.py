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

import functools
import warnings

import torch
import torch.nn as nn

from ecenet.ace_basis import ACEBasisAnalytic
from ecenet.edge_frame_kernel import edge_frame_fused, edge_frame_fused_single, pack_unrotate_fused
from ecenet.equivariant import EquivariantLinear, RealSpaceNonlinearity
from ecenet.film import ElementFiLM
from ecenet.radial import find_edges, get_cutoff_fn, radial_basis
from ecenet.spherical import build_D1_from_rhat, build_D_block, spherical_harmonics_float64, wigner_rotate

# les_readout modes whose l0 IS the latent charge itself (upstream's atomwise
# head bypassed via l0_is_charge). The single definition — consumers read the
# derived facts off the model via `les_flags` rather than re-spelling this.
_LES_EDGE_MODES = ('edge', 'edge_basis')
# les_alpha modes → width of the α slot in the packed l0 (1 scalar, or the
# full 3×3 tensor flattened) and number of per-edge head scalars it needs
# (a_e for the isotropic part; a_e + b_e for isotropic + bond-axial parts).
_LES_ALPHA_WIDTH = {None: 0, 'iso': 1, 'aniso': 9}
_LES_ALPHA_BLOCKS = {None: 0, 'iso': 1, 'aniso': 2}

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
        n_max_d:        if set, outer-product the invariants with f_d(r_ij) of this rank
        m_max:          max angular mode |m| kept after the equivariant layers
                        (default: l_max); lower it to cut cost at large l_max
        cutoff_type:    'cosine' or 'poly'
        activation:     pointwise activation in the realspace nonlinearity ('silu', 'tanh',
                        'relu', 'gelu'); 'identity' turns every RealSpaceNonlinearity — the
                        layer stack AND the MP trunk/receiver — into an exact no-op, so the
                        equivariant stack is purely linear (ablation; read-out MLPs keep
                        SiLU). Broader than use_nonlinearity=False, which skips the
                        nonlinearity in the main layer stack only (MP keeps its own).
        n_grid:         θ-grid points for the realspace nonlinearity (default: 4*m_max+1)
        output_hidden_dims: hidden widths of the readout MLP (default: [64])
        analytic_ace_basis: use ACEBasisAnalytic (recommended for force training)
        bottleneck_dim: if set, each equivariant layer becomes a low-rank block
                        (down → nonlin at this width → up, zero-init up so the
                        layer is identity at init); None → full-width layers
        mp_type:        how messages are aggregated at each receiver atom (n_mp
                        >= 2 only). Both styles share the same per-edge structure
                        — a fused message/score trunk and a receiver transform —
                        and differ only in the weight applied to each incoming
                        message:
                          'sum' (default): the raw signed score times the cutoff
                            envelope, summed (extensive in coordination).
                            Zero-init scores make the layer an exact no-op at
                            init.
                          'softmax': softmax over the receiver's incoming edges,
                            so the aggregate is a weighted *average* (intensive
                            in coordination). Zero-init scores make the
                            attention uniform at init.
        mp_dim:         bottleneck width of the fused message/score trunk and of
                        the receiver block (default: n_features_per_m // 4)
        mp_n_heads:     number of attention heads; the value channels
                        (n_base = 2*embed_dim) split evenly across them
                        (default 1). Ignored (with a warning) when n_mp=1.
        mp_l_attention: if True, each head emits one score PER degree l and
                        weights a receiver's in-edges independently per (head, l),
                        so a neighbour can matter for l=1 and not for l=2. The
                        score stays an invariant scalar applied uniformly across
                        that l's m-block, which is what keeps it equivariant —
                        the Wigner-D block is l-diagonal, so splitting across l is
                        legal, while splitting within an l / across m is not.
                        Widens the fused trunk to n_ch + n_heads*(l_max+1).
        mp_msg_envelope: if True (default), the aggregated message decays with
                        *absolute* distance. For mp_type='softmax' this
                        multiplies the softmax weight by f_cut(r) a second time,
                        undoing the normalizer's division of the absolute cutoff
                        (without it, a lone neighbour near r_cut still gets weight
                        ≈ 1). mp_type='sum' is already enveloped by construction,
                        so the flag is a no-op there — and cannot be turned off.
        element_film:   modulate the edge features once — right after they are
                        built and rotated into the bond frame, before the layer
                        stack — by an element(+distance)-conditioned FiLM gate
                        (see ecenet/film.py). Identity at init. ON by default;
                        element_film=False disables the gate.
        film_embed_dim: width of each element embedding in the FiLM gate (default 16)
        film_n_rbf:     radial-basis size φ(r) for the FiLM gate (0 → element-only,
                        so the gate depends on the element pair but not the bond
                        length)
        film_hidden:    FiLM gate MLP hidden width(s); None → [max(2*C, 32)]
        film_per_m:     if True the gate emits a scale per (channel, m) rather than
                        one per channel broadcast over m. Still equivariant: cos and
                        sin of a given (channel, m) share the scale. Structural-zero
                        slots (m > l of that channel) are masked to γ=1.
        film_shift:     if True the gate also predicts a shift β (full FiLM,
                        γ⊙x + β) as an extra head on the same MLP. β lands on the
                        m=0 slot of A_cos only — the invariant mode, the only place
                        an additive shift preserves equivariance.
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
        n_max_d: int = None,
        cutoff_type: str = 'cosine',
        activation: str = 'silu',
        use_nonlinearity: bool = True,
        n_grid: int = None,
        analytic_ace_basis: bool = True,
        output_hidden_dims: list = None,
        m_max: int = None,
        bottleneck_dim: int = None,
        mp_type: str = 'sum',
        mp_dim: int = None,
        mp_n_heads: int = 1,
        mp_msg_envelope: bool = True,
        mp_l_attention: bool = False,
        element_film: bool = True,
        film_embed_dim: int = 16,
        film_n_rbf: int = 0,
        film_hidden=None,
        film_per_m: bool = False,
        film_shift: bool = False,
        les_readout: str = 'sum',
        les_charge_scale: float = 1.0,
        les_dipole: bool = False,
        les_charges: bool = True,
        les_alpha: str | None = None,
    ):
        super().__init__()
        if mp_type == 'transformer':
            # Former name for 'softmax'. Accepted so scripts and checkpoints
            # written under the old name keep working (mp_type is saved in hparams).
            warnings.warn("mp_type='transformer' has been renamed to 'softmax'; "
                          "the old name still works but will not be documented.",
                          stacklevel=2)
            mp_type = 'softmax'
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
        self.bottleneck_dim = bottleneck_dim
        self.mp_type = mp_type
        self.n_sph = (l_max + 1) ** 2
        self.m_max = int(m_max) if m_max is not None else l_max
        self.n_angular = self.m_max + 1   # m = 0..m_max (layers only use up to m_max)

        # ── Joint (n_types, n_max) → embed_dim contraction per central atom type ──
        # W[type_i, t, n, c]: for central atom of type type_i, contract
        # neighbor type t and radial channel n into embed channel c.
        # (Initial Atomic embedding)
        self.W = nn.Parameter(
            torch.randn(n_types, n_types, n_max, embed_dim)
            / (n_types * n_max) ** 0.5
        )

        # ── SH → A_cos/A_sin reshape ──────────────────────────────────────
        # m_max controls output angular modes; ACE basis always uses full l_max.
        # Going from node to edge frame
        self.sph_to_angular = SphToAngular(embed_dim, l_max, m_max=self.m_max)
        # n_features_per_m = 2 * embed_dim * (l_max+1): one channel per (side, embed, l)
        self.n_features_per_m = 2 * embed_dim * (l_max + 1)

        # ── (l0, l1) read-out aggregation (return_embeddings) ─────────────
        # 'sum' (default): parameter-free scatter-sum of the final edge
        #   invariants — extensive in coordination, smooth only through the
        #   radial basis's own envelope.
        # 'softmax': attention-weighted aggregation mirroring the MP layers'
        #   softmax path — a zero-init linear score on each edge's invariant
        #   h_l0, segment-softmaxed over the receiver's in-edges with f_cut as
        #   a multiplicative log-bias, then the envelope multiplied back in
        #   (the normalizer divides the absolute f_cut out, exactly as in
        #   mp_msg_envelope). Weight is an invariant scalar shared by the l0
        #   and l1 messages, so equivariance is untouched. Zero-init score →
        #   uniform attention at init; on a lone in-edge the weight reduces to
        #   f_cut(r), giving the read-out absolute-distance decay.
        # 'edge': Allegro-LES-style per-edge charge decomposition — a linear
        #   scalar head on each edge's invariants, scatter-summed per atom, so
        #   l0 has width 1 and IS the latent charge (the LES wrapper must be
        #   called with l0_is_charge=True; upstream's atomwise head is
        #   bypassed). Standard init, deliberately NOT zero-init: the LES
        #   energy is quadratic in the charges, so q ≡ 0 is a gradient-free
        #   saddle a zero-init head could never leave. Smoothness at r_cut is
        #   inherited from the edge features' own radial envelope, as in
        #   Allegro-LES's EdgewiseReduce.
        # 'edge_basis' (the trainers' default when use_les=True — the
        #   bare-model default stays 'sum' so short-range models carry no
        #   unused charge-head parameters): 'edge' upgraded to mirror the energy readout end to
        #   end — an MLP with output_net's architecture (same input: the full
        #   n_features_per_m m=0 invariant set of _contract, not the l'-summed
        #   h_l0; same hidden widths and activation) emits n_max_d channels
        #   dotted with the (cutoff-enveloped) radial basis of the edge
        #   length, exactly mirroring _apply_output. The per-bond charge
        #   contribution gains an explicit learnable distance profile and
        #   vanishes exactly at r_cut (n_max_d=None falls back to a scalar ×
        #   f_cut, as in the energy readout). Built below, after the output-
        #   MLP config it mirrors. Last layer standard init, NOT output_net's
        #   near-zero init — see the saddle note under 'edge'.
        if les_readout not in ('sum', 'softmax', *_LES_EDGE_MODES):
            raise ValueError("les_readout must be 'sum', 'softmax', 'edge' or "
                             f"'edge_basis', got {les_readout!r}")
        self.les_readout = les_readout
        _edge_mode = les_readout in _LES_EDGE_MODES
        # les_dipole: the edge head also emits a per-edge scalar d_e whose
        # bond-dipole contribution d_e·r̂_e is scattered alongside the charge —
        # l0 is then PACKED as (n_atoms, 4): column 0 the latent charge,
        # columns 1:4 the latent atomic dipole (a true polar vector: an
        # invariant scalar times r̂, so parity-correct by construction — for a
        # planar/collinear neighbourhood the dipole is confined to exactly the
        # subspace mirror/axial symmetry allows). Charge and dipole share the
        # head trunk (the MP layers' fused-trunk pattern); the dipole rows of
        # the last layer are zero-init — u ≡ 0 at init is NOT a saddle (the
        # LES energy's qᵀf_qu·u cross-term drives it once charges exist), and
        # it means enabling the flag doesn't perturb the model at step 0.
        if les_dipole and not _edge_mode:
            raise ValueError(
                f"les_dipole=True requires les_readout='edge' or 'edge_basis' "
                f"(got {les_readout!r}): the dipole is emitted by the per-edge "
                "charge head, which the atomwise read-outs don't have.")
        # les_charges=False: dipoles-only ablation — the head emits ONLY the
        # dipole block and the q column of the packed l0 is hard zero, so
        # E_lr = ½uᵀf_uu·u alone (the f_qq and f_qu terms vanish identically
        # at q = 0 and the wrapper/calculators/tools need no changes). With
        # the charges gone the qᵀf_qu·u cross-term that un-saddles the
        # zero-init dipole slot is gone too — ∂E_lr/∂u ∝ u — so the dipole
        # block gets STANDARD init here (the 'edge' charge-head saddle note,
        # transplanted), with les_charge_scale now acting on u.
        if not les_charges and not les_dipole:
            raise ValueError(
                "les_charges=False requires les_dipole=True (dipoles-only "
                "read-out): with both off there is no latent multipole left "
                "for the LES energy.")
        self.les_dipole = bool(les_dipole)
        self.les_charges = bool(les_charges)
        # les_alpha: the edge head also emits a per-atom latent polarizability
        # α_i (Allegro-LES's construction; Kim, King, Park et al. 2026, arXiv
        # 2605.05746) for upstream's induced-dipole term — each atom's dipole
        # responds linearly to the field of the fixed multipoles, Δu_i = α_i·E_i,
        # lowering the energy by ½E_i·α_i·E_i (non-self-consistent: induced
        # dipoles don't see each other). Appended to the packed l0 after [q | u]:
        #   'iso':   one scalar a_e per edge, scatter-summed → α_i (1 column)
        #   'aniso': two scalars per edge, a_e·I + b_e·(r̂r̂ᵀ − I/3) — an
        #            isotropic part plus a traceless bond-axial part, symmetric
        #            and equivariant by construction (the Cartesian form of an
        #            l=0 + l=2 edge harmonic; a lone bond gives exactly the
        #            uniaxial α_∥/α_⊥ of a diatomic) — summed → α_i (9 columns,
        #            the flattened 3×3)
        # Head slots are zero-init: α ≡ 0 at init leaves an existing model
        # untouched at step 0, and is NOT a saddle (E_lr is linear in α with
        # gradient −½|E_i|², nonzero once charges exist). Unlike q, the sign of
        # α is physical (E_lr is linear in it), and les_charge_scale does not
        # touch it (a response coefficient, not a source). Physical α needs the
        # ε_e unscaling of the paper's Eqs. 23–24 (ε_e = 1 in vacuum).
        if les_alpha not in _LES_ALPHA_WIDTH:
            raise ValueError("les_alpha must be None, 'iso' or 'aniso', "
                             f"got {les_alpha!r}")
        if les_alpha is not None and not _edge_mode:
            raise ValueError(
                f"les_alpha={les_alpha!r} requires les_readout='edge' or "
                f"'edge_basis' (got {les_readout!r}): the polarizability is "
                "emitted by the per-edge charge head, which the atomwise "
                "read-outs don't have.")
        self.les_alpha = les_alpha
        self._n_alpha = _LES_ALPHA_WIDTH[les_alpha]
        self._l0_dim = ((1 + (3 if les_dipole else 0) + self._n_alpha)
                        if _edge_mode else 2 * embed_dim)
        # head slot counts (q, d_e, α scalars): each slot is one n_output_out
        # block of the 'edge_basis' MLP's last layer, or one 'edge' linear row
        self._les_head_blocks = (int(les_charges), int(les_dipole),
                                 _LES_ALPHA_BLOCKS[les_alpha])
        # les_charge_scale: fixed multiplier on the edge-mode latent charge
        # — the whole packed [q | u] when les_dipole is on, keeping the q–u
        # sign coupling intact (MACELES's output_scale; they ship 0.1). With standard head init,
        # q = s·q_raw starts small-but-nonzero — off the quadratic energy's
        # q=0 saddle, but with E_lr suppressed ~s² early, so the short-range
        # fit leads and the charges learn gently relative to it. Not a
        # parameter; recorded in hparams, so eval tools see scaled charges.
        if les_charge_scale <= 0:
            raise ValueError(f"les_charge_scale must be > 0, "
                             f"got {les_charge_scale!r}")
        if les_charge_scale != 1.0 and not _edge_mode:
            # Warn rather than silently ignore (repo convention): for
            # 'sum'/'softmax' the charge comes out of upstream's atomwise
            # head, so the model never sees it and cannot scale it.
            warnings.warn(
                f"les_charge_scale={les_charge_scale} is ignored: "
                f"les_readout={les_readout!r} maps l0 to charges inside the "
                "upstream LES head; only 'edge'/'edge_basis' emit the charge "
                "directly.", stacklevel=2)
        self.les_charge_scale = float(les_charge_scale)
        if les_readout == 'softmax':
            self.les_score = nn.Linear(2 * embed_dim, 1)
            nn.init.zeros_(self.les_score.weight)
            nn.init.zeros_(self.les_score.bias)
        elif les_readout == 'edge':
            n_q, n_u, n_a = self._les_head_blocks
            self.les_edge_charge = nn.Linear(2 * embed_dim, n_q + n_u + n_a,
                                             bias=False)
            with torch.no_grad():
                if les_dipole and les_charges:
                    self.les_edge_charge.weight[1].zero_()   # dipole slot
                if n_a:
                    self.les_edge_charge.weight[n_q + n_u:].zero_()   # α slots
            # les_charges=False: the single (standard-init) row IS the dipole
            # scalar — zero-init would sit on the uu-quadratic saddle (above)
        # (the 'edge_basis' head is built with the output MLP below)

        # ── Element(+distance)-conditioned FiLM gate (optional) ───────────────
        # A small MLP on [embed(type_i), embed(type_j), φ(r_ij)] → a scale γ on
        # A_cos/A_sin, applied once to the freshly built edge features. Identity
        # at init (the gate MLP's last layer is zero-init → γ=1, β=0).
        #   film_per_m=False (default): one scale γ_c per channel, broadcast over
        #     the angular modes m.
        #   film_per_m=True: a scale γ_{c,m} per (channel, m). Still equivariant —
        #     the bond frame's residual symmetry rotates (A_cos_m, A_sin_m) by mφ,
        #     which any scale commutes with as long as cos and sin of that mode
        #     share it (they do). The structural-zero slots (m > l_of_c[c]) are
        #     masked to γ=1: they can carry rotation-inconsistent values, so a
        #     per-mode scale must not touch them differentially (a per-channel γ
        #     scales them uniformly with the rest of the channel, which is safe).
        # film_shift=True adds a shift β (γ⊙x + β) as an extra head on the same gate
        # MLP. β lands on the m=0 slot of A_cos only: m=0 is the invariant mode, so
        # an invariant scalar added there is exactly equivariant, whereas a shift on
        # any m>0 mode is not (it does not commute with the e^{imφ} gauge rotation).
        # The m=0 *sin* slot is a structural zero (sin(0·φ)=0) and stays untouched.
        self.film_per_m = bool(film_per_m)
        self.film_shift = bool(film_shift)
        if not element_film:
            # Warn rather than silently ignore: a configured gate that is never
            # built looks like it was applied.
            _set = [n for n, v, d in (('film_embed_dim', film_embed_dim, 16),
                                      ('film_n_rbf', film_n_rbf, 0),
                                      ('film_hidden', film_hidden, None),
                                      ('film_per_m', film_per_m, False),
                                      ('film_shift', film_shift, False)) if v != d]
            if _set:
                warnings.warn(
                    f"{', '.join(_set)} ignored: they configure the FiLM gate, "
                    "which is off (element_film=False).", stacklevel=2)
        film_n_modes, film_mode_valid = 1, None
        if self.film_per_m:
            film_n_modes = self.n_angular
            l_of_c = torch.arange(self.n_features_per_m) % (l_max + 1)
            film_mode_valid = (torch.arange(film_n_modes)[None, :]
                               <= l_of_c[:, None]).to(torch.get_default_dtype())
        if self.film_shift:
            # one-hot selector for m=0, to add β without an in-place index write
            m0 = torch.zeros(1, 1, self.n_angular)
            m0[..., 0] = 1.0
            self.register_buffer('film_m0', m0)
        self.element_film = ElementFiLM(
            self.n_features_per_m, n_types,
            embed_dim=film_embed_dim, n_rbf=film_n_rbf, hidden=film_hidden,
            shift=self.film_shift, n_modes=film_n_modes,
            mode_valid=film_mode_valid) if element_film else None

        # ── Equivariant layers: Linear → RealSpaceNonlinearity → residual ────
        # Message passing: the model is `n_mp` stages of `n_layers` equivariant
        # layers each, with one equivariant MP layer *between* consecutive stages
        # (n_mp-1 MP layers total, no trailing MP). n_mp == 1 is the plain model:
        # a flat list of n_layers equivariant layers and no MP. n_mp >= 2 groups
        # the layers into stages and adds the interleaved MP layers.
        self.n_mp = n_mp
        self.layers = nn.ModuleList([
            ECENetLayer(self.n_features_per_m, self.m_max, activation=activation,
                        use_nonlinearity=use_nonlinearity, n_grid=n_grid,
                        bottleneck_dim=bottleneck_dim)
            for _ in range(n_mp * n_layers)
        ])
        # n_mp >= 2: regroup the flat layers into `n_mp` stages and build the
        # `n_mp - 1` MP layers that sit between them.
        if mp_type not in ('softmax', 'sum'):
            raise ValueError(
                f"Unknown mp_type '{mp_type}' (expected 'softmax' or 'sum'). "
                "The old distance/type-weighted 'edge' message passing has been removed.")
        # Warn rather than silently ignore: an MP-only knob left at a non-default
        # value with n_mp=1 does nothing, and a silent no-op looks like the
        # setting was applied.
        if mp_l_attention and n_mp == 1:
            warnings.warn(
                "mp_l_attention=True is ignored: message passing is off (n_mp=1).",
                stacklevel=2)
        if mp_n_heads != 1 and n_mp == 1:
            warnings.warn(
                f"mp_n_heads={mp_n_heads} is ignored: message passing is off "
                f"(n_mp=1).", stacklevel=2)
        # The only surprising case: 'sum' is enveloped by construction (a = s·f_cut),
        # so asking to turn the envelope OFF cannot be honoured.
        if not mp_msg_envelope and mp_type == 'sum' and n_mp > 1:
            warnings.warn(
                "mp_msg_envelope=False has no effect with mp_type='sum': its "
                "weight is s·f_cut, so the message is enveloped by construction.",
                stacklevel=2)
        if n_mp > 1:
            flat = list(self.layers)
            self.layers = nn.ModuleList([
                nn.ModuleList(flat[g * n_layers:(g + 1) * n_layers])
                for g in range(n_mp)
            ])
            self.mp_layers = nn.ModuleList([
                ECENetAttentionMPLayer(
                    self.n_features_per_m, self.l_max, self.embed_dim,
                    n_types=n_types,
                    r_cut=self.r_cut_edge, cutoff_type=self.cutoff_type,
                    m_max=self.m_max, mp_dim=mp_dim,
                    activation=activation, n_grid=n_grid, n_heads=mp_n_heads,
                    aggregation=mp_type, msg_envelope=mp_msg_envelope,
                    l_attention=mp_l_attention,
                )
                for _ in range(n_mp - 1)
            ])

        # ── Output MLP ──────────────────────────────────────────────────────
        # inv → MLP → n_max_d, then dot with rij_basis (see _apply_output).
        hidden_dims = output_hidden_dims or [64]
        in_dim = self.n_features_per_m
        n_output_out = n_max_d if n_max_d is not None else 1
        mlp_dims = [in_dim] + list(hidden_dims) + [n_output_out]
        # 'identity' linearizes only the equivariant stack (RealSpaceNonlinearity
        # becomes an exact no-op there, incl. the MP trunk/receiver); the scalar
        # read-out MLPs keep SiLU — an identity MLP would just collapse to one
        # linear map. Unknown strings already raised in RealSpaceNonlinearity.
        act = {'silu': nn.SiLU, 'tanh': nn.Tanh, 'relu': nn.ReLU,
               'gelu': nn.GELU, 'identity': nn.SiLU}[activation]
        self.output_net = OutputMLP(mlp_dims, activation=act())

        # 'edge_basis' charge head: same dims/activation as output_net (built
        # here so it mirrors the readout config exactly). zero_init_last=False:
        # near-zero charges would start at the quadratic LES energy's
        # gradient-free saddle (see the 'edge' note above). With les_dipole the
        # last layer widens to a second n_output_out block (dipole channels,
        # dotted with the same radial basis), zero-init per the note above.
        if les_readout == 'edge_basis':
            n_q, n_u, n_a = self._les_head_blocks
            q_dims = mlp_dims[:-1] + [n_output_out * (n_q + n_u + n_a)]
            self.les_edge_charge = OutputMLP(q_dims, activation=act(),
                                             zero_init_last=False)
            with torch.no_grad():
                last = self.les_edge_charge.linears[-1]
                if les_dipole and les_charges:
                    last.weight[n_output_out:2 * n_output_out].zero_()   # dipole
                    last.bias[n_output_out:2 * n_output_out].zero_()
                if n_a:                                       # α blocks
                    k = (n_q + n_u) * n_output_out
                    last.weight[k:].zero_()
                    last.bias[k:].zero_()
            # les_charges=False: the single (standard-init) block IS the
            # dipole — see the saddle note at the les_charges check above

        # ── Per-type atomic energy baseline ──────────────────────────────
        self.atomic_energy = nn.Parameter(torch.zeros(n_types))


    # ── Helpers ────────────────────────────────────────────────────────────

    @property
    def les_flags(self):
        """The l0 convention as ``LESLongRange.forward`` kwargs.

        ``{'l0_is_charge': ..., 'les_dipole': ..., 'les_alpha': ...}`` — the
        single source of truth for how this model's ``l0`` read-out is to be
        interpreted (edge modes: l0 IS the charge, packed [q | u | α] under
        ``les_dipole`` / ``les_alpha``; ``ecenet.les.unpack_l0`` splits it).
        Call sites do ``les_module(l0, pos, ..., **model.les_flags)`` instead
        of re-deriving the flags from hparams or ``les_readout`` literals.
        """
        return {'l0_is_charge': self.les_readout in _LES_EDGE_MODES,
                'les_dipole': self.les_dipole,
                'les_alpha': self.les_alpha}

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

    def _apply_element_film(self, A_cos, A_sin, type_i, type_j, dist_ij):
        """Element(+distance)-conditioned FiLM scale on A_cos/A_sin.

        A_cos and A_sin of a given (channel, m) share the scale, which is what
        keeps the e^{imφ} transformation intact → SO(2)/SO(3)-equivariant, whether
        the scale is per-channel or per-(channel, m) (``film_per_m``). With
        ``film_shift`` a shift β is added to the m=0 slot of A_cos only — the
        invariant mode. Identity at init.
        """
        gamma, beta = self._film_params(type_i, type_j, dist_ij)
        return self._film_apply(A_cos, A_sin, gamma, beta)

    def _film_params(self, type_i, type_j, dist_ij):
        """FiLM scale γ and (with film_shift) the m=0 shift β.

        γ is shaped to broadcast against (n_edges, C, n_angular): (n_edges, C, 1)
        per-channel, or (n_edges, C, n_angular) with film_per_m. β is (n_edges, C)
        — the m=0 slot only — or None.
        """
        r_basis = None
        if self.element_film.n_rbf:
            r_basis = radial_basis(dist_ij, self.r_cut_edge, self.element_film.n_rbf,
                                   cutoff_type=self.cutoff_type)      # (n_edges, n_rbf)
        gamma, beta = self.element_film(type_i, type_j, r_basis)
        if gamma.dim() == 2:
            gamma = gamma.unsqueeze(-1)          # broadcast over m
        return gamma, beta

    def _film_apply(self, A_cos, A_sin, gamma, beta):
        """γ⊙A (+ β on the m=0 slot of A_cos). β is None unless film_shift."""
        A_cos, A_sin = A_cos * gamma, A_sin * gamma
        if beta is not None:
            A_cos = A_cos + beta.unsqueeze(-1) * self.film_m0     # m=0 slot only
        return A_cos, A_sin

    def _contract(self, A_cos, A_sin):
        """Extract m=0 invariants: (n_edges, n_features_per_m, n_angular) → (n_edges, n_features_per_m)."""
        return A_cos[:, :, 0]

    def _apply_output(self, invariants, dist_ij):
        """output_net(inv) → per-edge energies.

        n_max_d=None: the readout emits a single number per edge, multiplied by
        the cutoff envelope f(r) so the per-edge energy still decays smoothly to
        0 at r_cut_edge (continuous energy/forces) without an explicit radial
        basis — i.e. energy_edge = MLP(inv) · f(r_ij). The n_max_d>=1 path
        instead dots the MLP output with the (cutoff-enveloped) radial basis."""
        if self.n_max_d is not None:
            rij_basis = radial_basis(dist_ij, self.r_cut_edge, self.n_max_d,
                                     cutoff_type=self.cutoff_type)
            return (self.output_net(invariants) * rij_basis).sum(-1)
        env = get_cutoff_fn(self.cutoff_type)(dist_ij, self.r_cut_edge)   # (n_e,) smooth → 0 at r_cut
        return self.output_net(invariants).squeeze(-1) * env


    def _run_equivariant_layers(self, A_cos, A_sin, **kwargs):
        """Run the equivariant layers, interleaving a message-passing layer
        between consecutive stages when n_mp >= 2 (n_mp-1 MP layers, no trailing MP)."""
        type_i   = kwargs.get('type_i')
        type_j   = kwargs.get('type_j')
        dist_ij  = kwargs.get('dist_ij')

        # Element(+distance)-conditioned FiLM scale on the edge features, applied
        # once before the layer stack (covers every forward path through here).
        if self.element_film is not None:
            A_cos, A_sin = self._apply_element_film(A_cos, A_sin, type_i, type_j, dist_ij)

        if self.n_mp == 1:
            # Plain model: a flat list of equivariant layers, no message passing.
            for layer in self.layers:
                A_cos, A_sin = layer(A_cos, A_sin)
            return A_cos, A_sin
        # Message-passing path: stage, MP, stage, MP, ..., stage  (MP only between stages).
        r_hat   = kwargs.get('r_hat')
        edge_i  = kwargs.get('edge_i')
        edge_j  = kwargs.get('edge_j')
        n_atoms = kwargs.get('n_atoms')
        D_block = kwargs.get('D_block')
        for gi, stage in enumerate(self.layers):
            for layer in stage:
                A_cos, A_sin = layer(A_cos, A_sin)
            if gi < len(self.mp_layers):          # no MP after the final stage
                A_cos, A_sin = self.mp_layers[gi](
                    A_cos, A_sin, r_hat, dist_ij, edge_i, edge_j,
                    n_atoms, type_i, type_j,
                    D_block=D_block)
        return A_cos, A_sin

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

    def _aggregate_lr_embeddings(self, A_cos, A_sin, r_hat, edge_j, n_atoms,
                                 with_l1=True, dist_ij=None):
        """Aggregate edge features to per-atom (l0, l1) equivariant embeddings
        (exposed via return_embeddings; e.g. for downstream long-range terms).

        Avoids the full Wigner T rotation by:
          1. Pack A_cos/A_sin → full SH in bond frame  (E, n_ch, n_sph)
          2. Sum over the l'-expansion axis first       (E, 2*embed_dim, n_sph)
          3. l=0: D^0=1, rotation-invariant — take directly
          4. l=1: apply D^1_T (3×3) to get global frame — much cheaper than full D^l_max
          5. Scatter to atoms — plain sum (les_readout='sum'), or the
             attention weights of les_readout='softmax' (see __init__), which
             need dist_ij for the f_cut log-bias + envelope.

        Returns:
            l0: (n_atoms, 2*embed_dim)     per-atom invariant scalar embeddings
                — edge modes: (n_atoms, 1), the latent charge itself; with
                les_dipole, packed (n_atoms, 4) = [q | u_x u_y u_z]; with
                les_alpha, a further 1 ('iso') or 9 ('aniso', flattened 3×3)
                columns of latent polarizability (see __init__)
            l1: (n_atoms, 2*embed_dim, 3)  per-atom equivariant vector embeddings

        with_l1=False skips the l=1 Wigner rotation + scatter and returns
        (l0, None) — l0 is rotation-invariant (D^0=1) and needs no frame change,
        so when only the invariant is wanted (e.g. LES latent charges) the l=1
        work is pure waste.
        """
        device, dtype = A_cos.device, A_cos.dtype
        n_e = A_cos.shape[0]
        n_base = 2 * self.embed_dim

        # The l'-summed h_l0 feeds every read-out except 'edge_basis' (whose
        # head runs on the full m=0 set directly); the packed h_sum also feeds
        # l1. Skip the pack when neither is needed (edge_basis + l0_only).
        h_sum = None
        if with_l1 or self.les_readout != 'edge_basis':
            h = self._pack_sph(A_cos, A_sin)                        # (E, n_ch, n_sph)

            # Sum over l'-expansion axis first (rotation is linear, sum commutes with D^T)
            h_sum = (h.view(n_e, n_base, self.l_max + 1, self.n_sph)
                      .sum(dim=2))                                   # (E, 2*embed_dim, n_sph)

            # l=0: D^0 = 1, no rotation needed
            h_l0 = h_sum[:, :, 0]                                   # (E, 2*embed_dim)

        a = None
        if self.les_readout in ('edge', 'edge_basis'):
            # Per-edge charge decomposition (Allegro-LES style): a scalar per
            # edge from its invariants, summed at the receiver. The result is
            # already the latent charge — width 1, no downstream head. This
            # applies to l0 only; l1 keeps the plain unweighted sum.
            # 'edge': a linear head on the l'-summed invariants h_l0.
            # 'edge_basis' mirrors the energy readout end to end: its MLP head
            # runs on the full m=0 invariant set (the same input _contract
            # feeds output_net) and its n_max_d channels are dotted with the
            # enveloped radial basis of the edge length, so each bond's
            # contribution carries a learnable distance profile and vanishes
            # exactly at r_cut.
            # With les_dipole the head emits two blocks sharing the trunk:
            # the charge block and a dipole-scalar block d_e, each reduced
            # the same way; the per-edge bond dipole is d_e·r̂_e, and h_l0
            # is packed (E, 4) = [q | d·r̂] (see __init__).
            if self.les_readout == 'edge_basis':
                if dist_ij is None:
                    raise ValueError("les_readout='edge_basis' needs dist_ij "
                                     "in _aggregate_lr_embeddings")
                out = self.les_edge_charge(A_cos[:, :, 0])   # (E, K | 2K)
                if self.n_max_d is not None:
                    rb = radial_basis(dist_ij, self.r_cut_edge, self.n_max_d,
                                      cutoff_type=self.cutoff_type)
                    out = (out.view(n_e, -1, self.n_max_d)
                           * rb[:, None, :]).sum(-1)           # (E, 1 | 2)
                else:
                    env = get_cutoff_fn(self.cutoff_type)(dist_ij,
                                                          self.r_cut_edge)
                    out = out * env[:, None]
            else:
                out = self.les_edge_charge(h_l0)               # (E, 1 | 2)
            # Pack the per-edge contributions as [q | d·r̂ | α] (see __init__).
            # dipoles-only (les_charges=False): the q column stays hard zero
            # so the packed layout — and every consumer of it — is unchanged.
            k = 0
            if self.les_charges:
                cols = [out[:, :1]]
                k = 1
            else:
                cols = [torch.zeros_like(out[:, :1])]
            if self.les_dipole:
                cols.append(out[:, k:k + 1] * r_hat)
                k += 1
            if self.les_alpha == 'iso':
                cols.append(out[:, k:k + 1])
                k += 1
            elif self.les_alpha == 'aniso':
                # a_e·I + b_e·(r̂r̂ᵀ − I/3): isotropic + traceless bond-axial
                eye = torch.eye(3, device=device, dtype=dtype)
                tr0 = r_hat[:, :, None] * r_hat[:, None, :] - eye / 3.0
                alpha_e = (out[:, k, None, None] * eye
                           + out[:, k + 1, None, None] * tr0)   # (E, 3, 3)
                cols.append(alpha_e.reshape(n_e, 9))
                k += 2
            h_l0 = cols[0] if len(cols) == 1 else torch.cat(cols, dim=1)
        elif self.les_readout == 'softmax':
            # Same shape as the MP layers' softmax path (one score slot):
            #   a_e = exp(s_e)·f_cut_e / (Σ_{e'→j} exp(s_e')·f_cut_e' + eps) · f_cut_e
            # The trailing f_cut is the envelope multiplied back in — the
            # normalizer divides the absolute f_cut out, so without it a lone
            # neighbour near r_cut would keep weight ≈ 1. The weight is an
            # invariant scalar, so applying it to both l0 and l1 messages
            # preserves equivariance.
            if dist_ij is None:
                raise ValueError("les_readout='softmax' needs dist_ij in "
                                 "_aggregate_lr_embeddings")
            s = self.les_score(h_l0).squeeze(-1)                     # (E,)
            f_cut = get_cutoff_fn(self.cutoff_type)(dist_ij, self.r_cut_edge)
            s_max = torch.full((n_atoms,), float('-inf'), device=device, dtype=dtype
                               ).scatter_reduce(0, edge_j, s.detach(), reduce='amax',
                                                include_self=True)
            num = torch.exp(s - s_max[edge_j]) * f_cut               # (E,)
            denom = torch.zeros(n_atoms, device=device, dtype=dtype
                                ).scatter_add(0, edge_j, num)
            a = num / (denom[edge_j] + 1e-6) * f_cut                 # (E,)
            h_l0 = a[:, None] * h_l0

        l0 = torch.zeros(n_atoms, h_l0.shape[1], device=device, dtype=dtype
                         ).scatter_add(0, edge_j[:, None].expand_as(h_l0), h_l0)
        # scale after the scatter — scatter_add is linear, and atoms are
        # ~30-60× fewer than edges (exact same result, applies to l0 only:
        # l1 keeps the plain unweighted sum)
        if (self.les_charge_scale != 1.0
                and self.les_readout in _LES_EDGE_MODES):
            if self._n_alpha:
                # the α slot is a response coefficient, not a source: unscaled
                n_qu = l0.shape[1] - self._n_alpha
                l0 = torch.cat([l0[:, :n_qu] * self.les_charge_scale,
                                l0[:, n_qu:]], dim=1)
            else:
                l0 = l0 * self.les_charge_scale

        if not with_l1:
            return l0, None

        # l=1: apply D^1_T (3×3) — unrotate bond-frame l=1 to global frame
        # forward rotation: A_rot = A @ D  →  unrotate: h_global = h_bond @ D^T
        # einsum: h_global[e,c,n] = Σ_m h_bond[e,c,m] * D[e,n,m]
        D1 = build_D1_from_rhat(r_hat)                              # (E, 3, 3)
        h_l1 = torch.einsum('ecm,enm->ecn', h_sum[:, :, 1:4], D1)  # (E, 2*embed_dim, 3)
        if a is not None:
            h_l1 = a[:, None, None] * h_l1
        l1 = torch.zeros(n_atoms, n_base, 3, device=device, dtype=dtype
                         ).scatter_add(0, edge_j[:, None, None].expand_as(h_l1), h_l1)
        return l0, l1

    # ── Fused-kernel toggles (opt-in) ──────────────────────────────────────

    def set_edge_frame_fused(self, enabled: bool = True, e2n: bool = True):
        """Toggle the fused gather→rotate→reshape path (edge_frame_kernel) on
        forward steps 3-4 AND the MP layers' rotate-back+unpack (their steps
        5-6, single-source variant) where the layer has that pattern. The fused
        autograd.Function re-gathers in the backward instead of saving the
        (n_edges, R, n_sph) intermediates, and its backward is composed of
        differentiable ops — so unlike set_activation_fused it is safe for
        double-backward force-loss training. Numerically identical to the
        unfused ops (see tests/test_edge_frame_kernel.py). Returns self.
        """
        self._edge_frame_fused = enabled
        for layer in getattr(self, 'mp_layers', []):
            layer.edge_frame_fused = enabled
            layer.edge_frame_fused_e2n = e2n
        return self

    def _edge_frame(self, A_emb, edge_i, edge_j, r_hat):
        """Steps 3-4: gather endpoint features, rotate into the bond frame,
        reshape to (A_cos, A_sin). Returns (A_cos, A_sin, D_block); D_block is
        built here so callers can reuse it (MP layers, node aggregation)."""
        D_block = build_D_block(r_hat, self.l_max)
        if getattr(self, '_edge_frame_fused', False):
            A_cos, A_sin = edge_frame_fused(A_emb, edge_i, edge_j, D_block,
                                            self.sph_to_angular)
        else:
            A_both = torch.cat([A_emb[edge_i], A_emb[edge_j]], dim=1)
            A_rot = wigner_rotate(A_both, D_block)
            A_cos, A_sin = self.sph_to_angular(A_rot)
        return A_cos, A_sin, D_block

    def set_activation_fused(self, enabled: bool = True):
        """Toggle the fused recompute-in-backward path on all RealSpaceNonlinearity
        modules (equivariant layers + MP). Drops the (n_e,F,n_grid) grid transient
        from the saved-for-backward set — a memory win for forces/MD, numerically
        equivalent (Triton on CUDA+silu, else PyTorch recompute). No-op for
        non-fusible configs. Inference-oriented (single backward); leave off for
        double-backward force-loss training. Returns self.
        """
        for mod in self.modules():
            if hasattr(mod, 'fused') and hasattr(mod, 'cos_synth'):
                mod.fused = enabled
        return self

    # ── Forward ────────────────────────────────────────────────────────────

    def _edgeless_result(self, types, device, dtype, return_embeddings, l0_only):
        """Zero-edge result for forward / forward_pbc.

        Keeps the per-element constants: the with-edges paths add atomic_energy
        for every atom (edgeless ones included), and forward_batch_multi does
        the same for zero-edge structures — a bare 0 here would make the energy
        jump by Σ atomic_energy when the last edge crosses r_cut. Embeddings
        are zero rows (an edgeless atom scatter-sums no edge invariants).
        """
        energy = self.atomic_energy[types].sum()
        if not return_embeddings:
            return energy
        N = len(types)
        l0 = torch.zeros(N, self._l0_dim, device=device, dtype=dtype)
        if l0_only:
            return energy, l0
        l1 = torch.zeros(N, 2 * self.embed_dim, 3, device=device, dtype=dtype)
        return energy, l0, l1

    def forward(self, positions: torch.Tensor, types: torch.Tensor,
                return_embeddings: bool = False, l0_only: bool = False):
        """Compute total energy, and optionally per-atom embeddings.

        Args:
            positions:         (n_atoms, 3)
            types:             (n_atoms,) int tensor of atom-type indices
            return_embeddings: if True, also return per-atom (l0, l1) equivariant
                               embeddings for downstream use (e.g. long-range terms)
            l0_only:           with return_embeddings, return only the invariant l0
                               and skip the l=1 work (returns (energy, l0))

        Returns:
            energy: scalar tensor              if return_embeddings is False
            (energy, l0, l1)                   if return_embeddings is True
              l0: (N, 2*embed_dim)
              l1: (N, 2*embed_dim, 3)
            (energy, l0)                       if return_embeddings and l0_only
        """
        device, dtype = positions.device, positions.dtype

        # ── Edges ─────────────────────────────────────────────────────────
        edge_i_undir, edge_j_undir = find_edges(positions, self.r_cut_edge)
        if len(edge_i_undir) == 0:
            return self._edgeless_result(types, device, dtype,
                                         return_embeddings, l0_only)

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

        # ── Steps 3+4: Gather + Wigner rotation + reshape to A_cos / A_sin ─
        A_cos, A_sin, D_block = self._edge_frame(A_emb, edge_i, edge_j, r_hat)

        # ── Step 5: Equivariant layers ────────────────────────────────────
        ti, tj = types[edge_i], types[edge_j]
        A_cos, A_sin = self._run_equivariant_layers(
            A_cos, A_sin,
            r_hat=r_hat, edge_i=edge_i, edge_j=edge_j,
            dist_ij=dist_ij, n_atoms=len(types),
            type_i=ti, type_j=tj, D_block=D_block)

        # ── Step 6+7: m=0 invariants → output_net → dot(rij_basis) ──────────
        invariants = self._contract(A_cos, A_sin)   # (n_edges, n_features_per_m)
        per_edge_energy = self._apply_output(invariants, dist_ij)
        energy = per_edge_energy.sum() + self.atomic_energy[types].sum()

        if return_embeddings:
            l0, l1 = self._aggregate_lr_embeddings(
                A_cos, A_sin, r_hat, edge_j, len(types), with_l1=not l0_only,
                dist_ij=dist_ij)
            if l0_only:
                return energy, l0
            return energy, l0, l1
        return energy

    def forward_pbc(self, positions: torch.Tensor, types: torch.Tensor,
                    edge_i: torch.Tensor, edge_j: torch.Tensor,
                    shift_vecs_edge: torch.Tensor,
                    nb_src: torch.Tensor, nb_dst: torch.Tensor,
                    shift_vecs_nb: torch.Tensor,
                    return_embeddings: bool = False, l0_only: bool = False):
        """Compute total energy with periodic boundary conditions.

        Args:
            positions:         (N, 3) atom positions in Cartesian Å (wrapped to unit cell)
            types:             (N,) int tensor of atom-type indices
            edge_i, edge_j:    (n_edges,) directed edge indices (both i→j and j→i)
            shift_vecs_edge:   (n_edges, 3) Cartesian PBC shift vectors for edges
            nb_src, nb_dst:    (n_nb,) directed neighbor pair indices
            shift_vecs_nb:     (n_nb, 3) Cartesian PBC shift vectors for neighbors
            return_embeddings: if True, also return per-atom (l0, l1) embeddings
            l0_only:           with return_embeddings, skip the l=1 work

        Returns:
            energy — or (energy, l0[, l1]) as in forward().
        """
        device, dtype = positions.device, positions.dtype
        n_edges = len(edge_i)

        if n_edges == 0:
            return self._edgeless_result(types, device, dtype,
                                         return_embeddings, l0_only)

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

        A_cos, A_sin, D_block = self._edge_frame(A_emb, edge_i, edge_j, r_hat)

        ti, tj = types[edge_i], types[edge_j]
        A_cos, A_sin = self._run_equivariant_layers(
            A_cos, A_sin,
            r_hat=r_hat, edge_i=edge_i, edge_j=edge_j,
            dist_ij=dist_ij, n_atoms=len(types),
            type_i=ti, type_j=tj, D_block=D_block)

        invariants = self._contract(A_cos, A_sin)
        per_edge_energy = self._apply_output(invariants, dist_ij)
        energy = per_edge_energy.sum() + self.atomic_energy[types].sum()

        if return_embeddings:
            l0, l1 = self._aggregate_lr_embeddings(
                A_cos, A_sin, r_hat, edge_j, len(types), with_l1=not l0_only,
                dist_ij=dist_ij)
            if l0_only:
                return energy, l0
            return energy, l0, l1
        return energy

    @torch.no_grad()
    def _local_topology(self, pos):
        """Edge + neighbour indices of one structure (directed, LOCAL, no grad).

        The single source of the nonzero recipe: build_topology and the
        on-the-fly branch of forward_batch_multi both call it, so precomputed
        and per-step topologies cannot drift apart. no_grad keeps dist_mat's
        sub/sqrt from spawning dead nodes into the force double-backward graph
        (the energy's distances are recomputed from the positions downstream).
        """
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)              # (N, N, 3)
        dist_mat = torch.sqrt((diff ** 2).sum(-1) + 1e-30)      # (N, N)
        ei, ej = ((dist_mat < self.r_cut_edge) & (dist_mat > 1e-10)).nonzero(as_tuple=True)
        nb_src, nb_dst = ((dist_mat < self.r_cut_neighbor)
                          & (dist_mat > 1e-10)).nonzero(as_tuple=True)
        return ei, ej, nb_src, nb_dst

    @torch.no_grad()
    def build_topology(self, positions_list):
        """Precompute per-structure (edge_i, edge_j, nb_src, nb_dst) LOCAL indices,
        matching forward_batch_multi's nonzero output exactly.

        Pass the returned list back as forward_batch_multi(..., topology=...) to
        skip the per-step O(N²) dist_mat + per-structure nonzero syncs. Valid only
        for FIXED positions (e.g. an in-memory training set) — the indices go
        stale the moment an atom crosses a cutoff. Indices land on each
        structure's own device. (Distinct from forward_batch's topology dict,
        which is one topology shared by every structure of the same molecule.)
        """
        return [self._local_topology(pos) for pos in positions_list]

    def forward_batch_multi(self, positions_list, types_list,
                            return_embeddings=False, l0_only=False,
                            topology=None):
        """Batch forward for variable-size, variable-composition structures.

        Only topology (edge/neighbour indices) is built per-structure in a
        cheap Python loop; the expensive ops (ACE basis, embed, Wigner
        rotation, equivariant layers, output MLP) run once on the concatenated
        atom/edge set.

        Args:
            positions_list:  list of B tensors, each (N_b, 3)
            types_list:      list of B tensors, each (N_b,) of type indices
            return_embeddings: if True, also return per-atom (l0, l1) embeddings,
                               each as a list of B per-structure tensors
            l0_only:           with return_embeddings, skip the l=1 work
                               (returns (energies, l0_list))
            topology:          optional list of B per-structure LOCAL index
                               tuples from build_topology (fixed positions):
                               (edge_i, edge_j, nb_src, nb_dst) for
                               non-periodic structures, or the 6-tuple
                               (edge_i, edge_j, shift_e, nb_src, nb_dst,
                               shift_nb) with Cartesian PBC shift vectors for
                               periodic ones (the trainers' convention:
                               diff = pos[j] − pos[i] + shift). Mixed batches
                               are allowed; skips the per-structure dist_mat
                               + nonzero syncs

        Returns:
            energies: (B,) tensor — or (energies, l0_list[, l1_list])
        """
        B = len(positions_list)
        device = positions_list[0].device
        dtype  = positions_list[0].dtype

        edge_i_list, edge_j_list = [], []   # flat atom indices with offsets (for MP)
        nb_src_list, nb_dst_list = [], []   # flat ACE neighbour indices (offset)
        shift_e_list, shift_nb_list = [], []   # per-structure PBC shifts (or None)
        any_pbc = False
        struct_ids = []
        atomic_e_list = []
        atom_offset = 0

        # ── Per-structure topology only (cheap: distance matrix + nonzero) ──
        # The expensive ops (ACE basis, embed, Wigner rotation, layers) run once
        # on the concatenated atom/edge set below. Keeping only edge/neighbour
        # index construction in this Python loop means it launches no ACE /
        # spherical-harmonic kernels per structure — so a batch of many tiny
        # molecules no longer pays O(#structures) kernel-launch overhead (the
        # cause of the DDP straggler on the rank that draws the small-molecule
        # batches). Neighbour indices carry a running atom offset and never cross
        # structures, so the single batched basis is block-diagonal == per-frame.
        for b, (pos, types) in enumerate(zip(positions_list, types_list)):
            N_b = pos.shape[0]
            # Topology is non-differentiable: precomputed (see build_topology)
            # or built now by the same helper, whose no_grad keeps dead nodes
            # out of the force double-backward graph. atomic_e (param-grad)
            # stays OUTSIDE it.
            entry = (topology[b] if topology is not None
                     else self._local_topology(pos))
            if len(entry) == 6:      # periodic: shifts interleaved (trainer order)
                ei, ej, she, nb_src, nb_dst, shn = entry
                any_pbc = True
            else:
                ei, ej, nb_src, nb_dst = entry
                she = shn = None

            atomic_e_list.append(self.atomic_energy[types].sum())
            if len(ei) == 0:
                atom_offset += N_b
                continue

            edge_i_list.append(ei + atom_offset)
            edge_j_list.append(ej + atom_offset)
            nb_src_list.append(nb_src + atom_offset)
            nb_dst_list.append(nb_dst + atom_offset)
            shift_e_list.append(she)
            shift_nb_list.append(shn)
            struct_ids.append(torch.full((len(ei),), b, dtype=torch.long, device=device))
            atom_offset += N_b

        energies = torch.stack(atomic_e_list)   # (B,)

        total_edges = sum(len(x) for x in edge_i_list)
        if total_edges == 0:
            if return_embeddings:
                n_ch = 2 * self.embed_dim
                l0_list = [torch.zeros(p.shape[0], self._l0_dim, dtype=dtype, device=device)
                           for p in positions_list]
                if l0_only:
                    return energies, l0_list
                l1_list = [torch.zeros(p.shape[0], n_ch, 3, dtype=dtype, device=device)
                           for p in positions_list]
                return energies, l0_list, l1_list
            return energies

        # Merge flat edge / neighbour arrays
        edge_i_flat = torch.cat(edge_i_list)
        edge_j_flat = torch.cat(edge_j_list)
        nb_src_flat = torch.cat(nb_src_list)
        nb_dst_flat = torch.cat(nb_dst_list)
        struct_idx  = torch.cat(struct_ids)

        # PBC: concatenate the per-structure shift vectors (zeros for any
        # non-periodic structures in a mixed batch). Shifts enter in exactly
        # the two places forward_pbc uses them: the edge diff below and the
        # ACE basis; everything downstream sees only diff/r_hat/dist.
        shift_e_flat = shift_nb_flat = None
        if any_pbc:
            shift_e_flat = torch.cat([
                s if s is not None else torch.zeros(len(e), 3, dtype=dtype,
                                                    device=device)
                for s, e in zip(shift_e_list, edge_i_list)])
            shift_nb_flat = torch.cat([
                s if s is not None else torch.zeros(len(n), 3, dtype=dtype,
                                                    device=device)
                for s, n in zip(shift_nb_list, nb_src_list)])

        # ── ACE basis + embed once on the whole concatenated atom set ──
        # The basis of a central atom depends only on its own neighbour list
        # (scattered by nb_src), and indices never cross structures, so a single
        # batched call reproduces the per-structure result exactly while
        # collapsing O(#structures) ACE/embed launches to one. Atoms in
        # zero-edge structures appear in no neighbour list → zero basis, unused.
        pos_all   = torch.cat(positions_list, dim=0)        # (N_total, 3)
        types_all = torch.cat(types_list, dim=0)            # (N_total,)

        A = self._compute_ace_basis(pos_all.unsqueeze(0), nb_src_flat, nb_dst_flat,
                                    types_all,
                                    shift_vecs_nb=shift_nb_flat).squeeze(0)
        A_emb = self._embed(A, types_all)                   # (N_total, embed_dim, n_sph)

        # Per-edge geometry computed flat from the concatenated positions
        # (gradients flow to the per-frame leaves through the cat → forces are
        # unchanged); features gathered from the per-atom embed in _edge_frame.
        diff_ij = pos_all[edge_j_flat] - pos_all[edge_i_flat]
        if shift_e_flat is not None:
            diff_ij = diff_ij + shift_e_flat.to(dtype=dtype)
        dist_ij = torch.sqrt((diff_ij ** 2).sum(-1) + 1e-30)
        r_hat   = diff_ij / dist_ij.unsqueeze(-1)
        type_i  = types_all[edge_i_flat]
        type_j  = types_all[edge_j_flat]

        A_cos, A_sin, D_block = self._edge_frame(A_emb, edge_i_flat, edge_j_flat, r_hat)

        A_cos, A_sin = self._run_equivariant_layers(
            A_cos, A_sin,
            r_hat=r_hat, edge_i=edge_i_flat, edge_j=edge_j_flat,
            dist_ij=dist_ij, n_atoms=atom_offset,
            type_i=type_i, type_j=type_j, D_block=D_block)

        invariants = self._contract(A_cos, A_sin)
        per_edge_energy = self._apply_output(invariants, dist_ij)

        energies = energies + torch.zeros(B, dtype=dtype, device=device).scatter_add(
            0, struct_idx, per_edge_energy)

        if return_embeddings:
            l0_flat, l1_flat = self._aggregate_lr_embeddings(
                A_cos, A_sin, r_hat, edge_j_flat, atom_offset, with_l1=not l0_only,
                dist_ij=dist_ij)
            # edge_j_flat indexes the full [0, atom_offset) atom space — the
            # running offset spans every structure, including any zero-edge ones —
            # so slice back by true atom count from positions_list (NOT
            # atom_counts, which skips zero-edge structures); those structures
            # correctly get all-zero rows.
            l0_list = []
            l1_list = None if l0_only else []
            offset = 0
            for p in positions_list:
                N_b = p.shape[0]
                l0_list.append(l0_flat[offset:offset + N_b])
                if not l0_only:
                    l1_list.append(l1_flat[offset:offset + N_b])
                offset += N_b
            if l0_only:
                return energies, l0_list
            return energies, l0_list, l1_list
        return energies

    def forward_batch(self, positions_list, types, topology=None,
                      return_embeddings=False, l0_only=False):
        """Compute energies for a batch of structures sharing the same atom types.

        Args:
            positions_list: list of B (N, 3) tensors
            types:          (N,) int tensor of atom-type indices (same for all structures)
            topology:       dict with precomputed 'edge_i', 'edge_j', 'nb_src', 'nb_dst'
                            (and optionally 'shift_vecs_edge', 'shift_vecs_nb' for PBC)
                            for the fixed-topology (same molecule) case, or None to
                            fall back to per-structure self.forward calls.
            return_embeddings: if True, also return per-atom (l0, l1) embeddings,
                            each as a list of B per-structure tensors
            l0_only:        with return_embeddings, skip the l=1 work

        Returns:
            energies: (B,) tensor — or (energies, l0_list[, l1_list])
        """
        if not isinstance(topology, dict):
            # Variable-topology fallback: forward_batch_multi subsumes this case
            # (shared types is just every structure carrying the same type
            # vector). It builds topology per-structure and runs the expensive
            # ops once on the merged flat edge set — identical result. A
            # per-structure topology LIST is forwarded rather than silently
            # dropped; dict-format entries (scripts/train_ecenet's
            # precompute_topology) are normalized to the (ei, ej, nb_src,
            # nb_dst) tuples forward_batch_multi unpacks.
            if isinstance(topology, list):
                topology = [
                    (t['edge_i'], t['edge_j'], t['nb_src'], t['nb_dst'])
                    if isinstance(t, dict) else t
                    for t in topology]
            else:
                topology = None
            return self.forward_batch_multi(
                positions_list, [types] * len(positions_list),
                return_embeddings=return_embeddings, l0_only=l0_only,
                topology=topology)

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

        # ── Step 5: Equivariant layers ────────────────────────────────────
        # For batched MP: offset edge indices so scatter targets B*N atoms
        N = pos_batch.shape[1]
        offset = torch.arange(B, device=edge_i.device).repeat_interleave(n_edges) * N
        edge_i_flat = edge_i.repeat(B) + offset
        edge_j_flat = edge_j.repeat(B) + offset
        type_i_flat = type_i.repeat(B)
        type_j_flat = type_j.repeat(B)

        A_cos_flat, A_sin_flat = self._run_equivariant_layers(
            A_cos_flat, A_sin_flat,
            r_hat=r_hat_flat, edge_i=edge_i_flat, edge_j=edge_j_flat,
            dist_ij=dist_ij.reshape(B * n_edges), n_atoms=B * N,
            type_i=type_i_flat, type_j=type_j_flat, D_block=D_block)

        # ── Step 6+7: m=0 invariants → output_net → dot(rij_basis) ──────────
        invariants = self._contract(A_cos_flat, A_sin_flat)      # (B*n_edges, n_features_per_m)
        per_edge_energy = self._apply_output(invariants, dist_ij.reshape(B * n_edges))  # (B*n_edges,)
        energies = per_edge_energy.reshape(B, n_edges).sum(dim=1)        # (B,)
        energies = energies + self.atomic_energy[types].sum()

        if return_embeddings:
            l0_flat, l1_flat = self._aggregate_lr_embeddings(
                A_cos_flat, A_sin_flat, r_hat_flat, edge_j_flat, B * N,
                with_l1=not l0_only, dist_ij=dist_ij.reshape(B * n_edges))
            l0_list = [l0_flat[b * N:(b + 1) * N] for b in range(B)]
            if l0_only:
                return energies, l0_list
            l1_list = [l1_flat[b * N:(b + 1) * N] for b in range(B)]
            return energies, l0_list, l1_list
        return energies


# ---------------------------------------------------------------------------
# Equivariant layer: Linear → RealSpaceNonlinearity → residual
# ---------------------------------------------------------------------------


class ECENetLayer(nn.Module):
    """One equivariant layer: EquivariantLinear → nonlinearity.

    Without bottleneck: linear(n_ch → n_ch) → nonlin(n_ch) → residual.
    With bottleneck:    linear_down(n_ch → r) → nonlin(r) → linear_up(r → n_ch) → residual.

    The bottleneck is a low-rank update: the nonlinearity runs at the (smaller)
    bottleneck width r, and the up-projection is zero-init so the whole block is
    identity at init (the residual carries the input through unchanged).

    Args:
        n_features:        number of feature channels (= n_features_per_m)
        m_max:             maximum angular frequency (= l_max)
        activation:        pointwise activation (used by the realspace nonlinearity)
        use_nonlinearity:  if False, skip nonlinearity entirely (linear-only layer)
        bottleneck_dim:    if set, use the down → nonlin(r) → up bottleneck structure
                           (low-rank); None → full-width linear → nonlin
    """

    def __init__(self, n_features: int, m_max: int, activation: str = 'silu',
                 use_nonlinearity: bool = True, n_grid: int = None,
                 bottleneck_dim: int = None):
        super().__init__()
        n_angular = m_max + 1
        self.bottleneck_dim = bottleneck_dim
        # nonlin_features: dimension at which the nonlinearity operates — the
        # bottleneck width r when bottlenecking, else the full feature width.
        nonlin_features = bottleneck_dim if bottleneck_dim is not None else n_features

        if bottleneck_dim is not None:
            self.linear_down = EquivariantLinear(n_features, bottleneck_dim, n_angular, m_max)
            self.linear_up   = EquivariantLinear(bottleneck_dim, n_features, n_angular, m_max)
            # Zero-init the up-projection → bottleneck starts as identity via residual.
            nn.init.zeros_(self.linear_up.weights)
            nn.init.zeros_(self.linear_up.bias)
        else:
            self.linear = EquivariantLinear(n_features, n_features, n_angular, m_max)

        self.nonlin = None
        if use_nonlinearity:
            self.nonlin = RealSpaceNonlinearity(nonlin_features, m_max, n_grid=n_grid,
                                                activation=activation)
        self.use_nonlinearity = self.nonlin is not None

    def forward(self, A_cos, A_sin):
        A_cos_in, A_sin_in = A_cos, A_sin

        # (Down-)linear: project to the bottleneck width r, else stay full width.
        if self.bottleneck_dim is not None:
            A_cos, A_sin = self.linear_down(A_cos, A_sin)
        else:
            A_cos, A_sin = self.linear(A_cos, A_sin)
        if self.nonlin is not None:
            A_cos, A_sin = self.nonlin(A_cos, A_sin)
        # Up-projection back to the full feature width.
        if self.bottleneck_dim is not None:
            A_cos, A_sin = self.linear_up(A_cos, A_sin)

        return A_cos + A_cos_in, A_sin + A_sin_in


# ---------------------------------------------------------------------------
# Message passing layer
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _sph_pack_index(l_max, m_max, n_angular, n_sph, device):
    """Precompute gather indices + validity masks for the angular↔SH pack/unpack.

    The (l, m) → SH-slot mapping is data-independent (fixed by l_max/m_max), so
    the per-l slice-assign+flip loop is exactly a fixed gather.

    Returns long index + bool mask tensors on `device` (cached per shape/device):
      pack_*   (n_sph,)            — gather source in the flat (lp1*n_angular) grid
      unpack_* (lp1*n_angular,)    — gather source in the (n_sph) SH grid
    cos = m≥0 modes, sin = m<0 modes (mirrors the flip). Masks zero the slots the
    loop left untouched (|m|>m_out when m_max<l). Bit-identical to the loop.
    """
    lp1 = l_max + 1
    pack_c  = torch.zeros(n_sph, dtype=torch.long)
    pack_s  = torch.zeros(n_sph, dtype=torch.long)
    pack_cm = torch.zeros(n_sph, dtype=torch.bool)
    pack_sm = torch.zeros(n_sph, dtype=torch.bool)
    up_c  = torch.zeros(lp1 * n_angular, dtype=torch.long)
    up_s  = torch.zeros(lp1 * n_angular, dtype=torch.long)
    up_cm = torch.zeros(lp1 * n_angular, dtype=torch.bool)
    up_sm = torch.zeros(lp1 * n_angular, dtype=torch.bool)
    for l in range(lp1):
        m_out = min(l, m_max)
        for m in range(0, m_out + 1):          # cos: SH slot l²+l+m ← angular (l, m)
            s = l * l + l + m
            pack_c[s] = l * n_angular + m
            pack_cm[s] = True
            up_c[l * n_angular + m] = s
            up_cm[l * n_angular + m] = True
        for k in range(1, m_out + 1):          # sin: SH slot l²+l-k ← angular (l, k)
            s = l * l + l - k
            pack_s[s] = l * n_angular + k
            pack_sm[s] = True
            up_s[l * n_angular + k] = s
            up_sm[l * n_angular + k] = True
    dev = torch.device(device)
    return tuple(t.to(dev) for t in (pack_c, pack_cm, pack_s, pack_sm,
                                     up_c, up_cm, up_s, up_sm))


def _pack_angular_to_sph(A_cos, A_sin, n_base, l_max, m_max, n_angular, n_sph):
    """(n_e, n_base*lp1, n_angular) → SH (n_e, n_base, n_sph). Vectorized pack."""
    n_e = A_cos.shape[0]
    lp1 = l_max + 1
    pc, pcm, ps, psm, *_ = _sph_pack_index(l_max, m_max, n_angular, n_sph, A_cos.device)
    ac  = A_cos.reshape(n_e, n_base, lp1 * n_angular)
    asn = A_sin.reshape(n_e, n_base, lp1 * n_angular)
    ic  = pc.view(1, 1, n_sph).expand(n_e, n_base, n_sph)
    isn = ps.view(1, 1, n_sph).expand(n_e, n_base, n_sph)
    return (ac.gather(2, ic) * pcm.to(A_cos.dtype)
            + asn.gather(2, isn) * psm.to(A_cos.dtype))


def _unpack_sph_to_angular(v_rot, n_base, l_max, m_max, n_angular, n_sph):
    """SH (n_e, n_base, n_sph) → cos/sin (n_e, n_base, lp1, n_angular). Vectorized."""
    n_e = v_rot.shape[0]
    lp1 = l_max + 1
    _, _, _, _, uc, ucm, us, usm = _sph_pack_index(l_max, m_max, n_angular, n_sph,
                                                   v_rot.device)
    L = lp1 * n_angular
    ic  = uc.view(1, 1, L).expand(n_e, n_base, L)
    isn = us.view(1, 1, L).expand(n_e, n_base, L)
    d_cos = (v_rot.gather(2, ic)  * ucm.to(v_rot.dtype)).view(n_e, n_base, lp1, n_angular)
    d_sin = (v_rot.gather(2, isn) * usm.to(v_rot.dtype)).view(n_e, n_base, lp1, n_angular)
    return d_cos, d_sin


class ECENetAttentionMPLayer(nn.Module):
    """Attention-style message passing for ECENet.

    Per edge (i→j):
      * a low-rank residual *message* m_e (equivariant, edge frame), and
      * an invariant scalar *score* s_e (down → nonlin → m=0 → scalar).
    Messages are unrotated to the common global frame and aggregated at each
    receiver atom as a score-weighted combination of that atom's incoming edges.
    The result is rotated back into each edge's bond frame, passed through a
    low-rank residual *receiver* transform, and added to the features.

    ``aggregation`` selects how the per-edge weight is formed:

    ``'softmax'`` — softmax over the receiver's incoming edges, with the
    smooth cutoff envelope folded in as a multiplicative log-bias,

        a_e = exp(s_e)·f_cut(r_ij) / (Σ_{e'→j} exp(s_e')·f_cut(r_e') + eps),

    so a departing edge's weight vanishes continuously as it crosses r_cut_edge
    (it leaves numerator and normalizer together — no jump), and the +eps floor
    keeps a node's aggregate finite/continuous as its last edge leaves. Because
    the aggregation is a normalized weighted average it is *intensive* in
    coordination.

    ``'sum'`` — the raw signed score times the same envelope, summed:

        a_e = s_e·f_cut(r_ij).

    There is no normalizer, so the aggregate is *extensive* in coordination and
    the envelope is what carries the continuity at r_cut (it is no longer divided
    back out, so the message also decays with absolute distance). Weights are
    signed, so a neighbour can contribute negatively. The score read-out is
    zero-init, which makes s_e = 0 and hence the whole layer an exact no-op at
    initialisation — the softmax path cannot do this, since exp(0) = 1.

    With ``n_heads > 1`` the layer is multi-head: the score head emits ``n_heads``
    invariant scores per edge (one shared trunk, ``n_heads`` linear read-outs) and
    the message's value channels (``n_base``) are split into ``n_heads``
    contiguous groups — whole spherical channels, full ``n_sph`` each. Head h
    weights the receiver's in-edges independently and gates *its own* value slice,
    so a neighbour can matter for one feature subspace and be suppressed for
    another (the per-subspace routing a single scalar can't represent). All splits
    are along channels (never within ``n_sph`` / across ``m``, which would break
    rotation-invariance); ``n_base = 2·embed_dim`` must be divisible by ``n_heads``.

    Message and scores come out of ONE fused trunk: down → nonlin at ``mp_dim`` →
    up, where the up-projection emits ``n_ch + n_heads`` channels. The first
    ``n_ch`` are the message (added residually to the input), and the m=0
    components of the trailing ``n_heads`` channels are the per-head scores. Since
    score and message share a trunk there is no dedicated score head, which makes
    this cheaper than computing the two separately. The up-projection is zero-init,
    so at initialisation the message residual is 0 and every score is 0 — which for
    ``'sum'`` means the layer is an exact identity, and for ``'softmax'`` means
    attention starts uniform (exp(0) = 1) over each receiver's in-edges.

    Equivariance: the trunk and receiver are EquivariantLinear +
    RealSpaceNonlinearity in a bond frame; the per-head scores (m=0 channel) and
    the cutoff (a function of the invariant distance) make the per-head weights
    rotation-invariant (for the softmax, its per-node normalizer is a sum of
    invariant scalars); the cross-edge sum happens in the common global frame via
    the Wigner-D unrotate/rotate.

    The receiver is an ``ECENetLayer`` bottleneck block (down → nonlin at
    ``mp_dim`` → up, zero-init up so it is identity at init).
    """

    def __init__(self, n_features_per_m: int, l_max: int, embed_dim: int,
                 n_types: int, r_cut: float = 5.0,
                 cutoff_type: str = 'cosine', m_max: int = None,
                 mp_dim: int = None,
                 activation: str = 'silu', n_grid: int = None, n_heads: int = 1,
                 aggregation: str = 'softmax', msg_envelope: bool = True,
                 l_attention: bool = False):
        super().__init__()
        if aggregation not in ('softmax', 'sum'):
            raise ValueError(
                f"Unknown aggregation '{aggregation}' (expected 'softmax' or 'sum').")
        self.aggregation = aggregation
        # msg_envelope: multiply the softmax weight by f_cut(r_e) as well. The
        # weight already carries f_cut, but normalization divides the ABSOLUTE
        # value back out — only the relative f_cut across a receiver's in-edges
        # survives, so a lone near-cutoff neighbour still gets weight ~1. This
        # restores absolute-distance decay of the message. f_cut is invariant, so
        # equivariance is untouched. 'sum' has no normalizer and is therefore
        # already enveloped (a = s·f_cut); folding f_cut in twice would give f_cut²,
        # so the flag applies to the softmax path only.
        self.msg_envelope = bool(msg_envelope) and aggregation == 'softmax'
        self.l_max     = l_max
        self.n_sph     = (l_max + 1) ** 2
        self.m_max     = m_max if m_max is not None else l_max
        self.n_angular = self.m_max + 1
        self.n_ch      = n_features_per_m
        self.n_base    = n_features_per_m // (l_max + 1)
        # Multi-head attention: each head emits its own score → its own weights
        # over the receiver's in-edges → gates its own contiguous slice of the
        # value channels (whole spherical channels, full n_sph each — NEVER a
        # split within n_sph / across m, which would break rotation-invariance).
        # So n_base = 2·embed_dim must be divisible by n_heads (the value split).
        if self.n_base % n_heads != 0:
            raise ValueError(
                f"{aggregation} MP: n_base (=2·embed_dim={self.n_base}) must be "
                f"divisible by n_heads ({n_heads}) to split the value across heads.")
        self.n_heads = n_heads
        # Per-l attention: with l_attention, each head emits one score PER degree l
        # (l=0..l_max) and weights the receiver's in-edges INDEPENDENTLY per (head, l)
        # — so a neighbour can be weighted differently for l=1 than for l=2. The
        # score is still an invariant scalar per l, applied uniformly across that
        # l's m-block (l_of_s expands per-(head,l) → per-(head,m)), so equivariance
        # holds: splitting across l is legal because the Wigner-D block is
        # l-diagonal; splitting *within* an l / across m is NOT. Without it, one
        # score per head weights all l uniformly (the default).
        self.l_attention = bool(l_attention)
        self.n_scores_per_head = (l_max + 1) if self.l_attention else 1
        self.n_scores = n_heads * self.n_scores_per_head
        # Map each spherical index s=(l,m) → its per-head score slot: its degree l
        # when l_attention (l-major order, contiguous [l², (l+1)²) blocks), else 0
        # (one shared score per head). Non-persistent: derived from l_max only.
        if self.l_attention:
            l_of_s = torch.cat([torch.full((2 * l + 1,), l, dtype=torch.long)
                                for l in range(l_max + 1)])
        else:
            l_of_s = torch.zeros(self.n_sph, dtype=torch.long)
        self.register_buffer('l_of_s', l_of_s, persistent=False)
        # +eps floor on the per-node softmax normalizer: keeps it finite (no 0/0 →
        # NaN when a node's last edge reaches r_cut and every num → 0 together).
        self.softmax_eps = 1e-6
        n_ch = n_features_per_m
        mp_dim = mp_dim if mp_dim is not None else max(n_ch // 4, 1)

        # Receiver: low-rank residual block (down → nonlin(mp_dim) → up).
        self.receiver = ECENetLayer(n_ch, self.m_max, activation=activation,
                                    n_grid=n_grid, bottleneck_dim=mp_dim)
        # Fused message + score trunk: ONE low-rank trunk (down → nonlin → up)
        # whose up-projection emits n_ch message channels PLUS n_scores score
        # channels; the m=0 invariants of the extra channels are the per-head
        # scores. This replaces a separate message block *and* a separate score
        # head, so it is cheaper than computing the two independently. up is
        # zero-init → message residual = 0 and scores = 0 at init.
        self.msg_down   = EquivariantLinear(n_ch, mp_dim, self.n_angular, self.m_max)
        self.msg_nonlin = RealSpaceNonlinearity(mp_dim, self.m_max, n_grid=n_grid,
                                                activation=activation)
        self.msg_up     = EquivariantLinear(mp_dim, n_ch + self.n_scores,
                                            self.n_angular, self.m_max)
        nn.init.zeros_(self.msg_up.weights)
        nn.init.zeros_(self.msg_up.bias)

        # Smooth cutoff envelope for the per-edge weight (→ 0 at r_cut_edge).
        self.r_cut = r_cut
        self.cutoff_fn = get_cutoff_fn(cutoff_type)

    def _pack(self, A_cos, A_sin):
        """(n_e, n_ch, n_angular) → SH (n_e, n_base, n_sph)."""
        return _pack_angular_to_sph(A_cos, A_sin, self.n_base, self.l_max,
                                    self.m_max, self.n_angular, self.n_sph)

    def _unpack(self, v_rot, n_e):
        """SH (n_e, n_base, n_sph) → (n_e, n_ch, n_angular)."""
        d_cos, d_sin = _unpack_sph_to_angular(
            v_rot, self.n_base, self.l_max, self.m_max, self.n_angular, self.n_sph)
        return (d_cos.reshape(n_e, self.n_ch, self.n_angular),
                d_sin.reshape(n_e, self.n_ch, self.n_angular))

    def forward(self, A_cos, A_sin, r_hat, dist_ij, edge_i, edge_j,
                n_atoms, type_i, type_j, D_block=None):
        n_e = A_cos.shape[0]
        device, dtype = A_cos.device, A_cos.dtype
        if D_block is None:
            D_block = build_D_block(r_hat, self.l_max)

        # 1+2. Fused trunk: down → nonlin → up(n_ch + n_scores). The first n_ch
        #      channels are the message (added residually, so the block is a
        #      low-rank update); the m=0 components of the trailing n_scores
        #      channels are the per-head scores (invariant, hence equivariance-safe).
        u_cos, u_sin = self.msg_down(A_cos, A_sin)
        u_cos, u_sin = self.msg_nonlin(u_cos, u_sin)
        u_cos, u_sin = self.msg_up(u_cos, u_sin)          # (n_e, n_ch+n_scores, n_angular)
        m_cos = u_cos[:, :self.n_ch] + A_cos
        m_sin = u_sin[:, :self.n_ch] + A_sin
        s = u_cos[:, self.n_ch:self.n_ch + self.n_scores, 0]   # (n_e, n_heads)

        # 3. Pack message → SH, unrotate to the common global frame.
        # Fused variant collapses the two into one op (the packed h never
        # materializes); the weighting below acts on h_global either way — the
        # gate is applied in the node frame, so 'sum'/'softmax'/l_attention all
        # compose with the fusion unchanged.
        if getattr(self, 'edge_frame_fused_e2n', False):
            h_global = pack_unrotate_fused(m_cos, m_sin, D_block,
                                           self.l_max, self.m_max)
        else:
            h = self._pack(m_cos, m_sin)
            h_global = torch.bmm(h, D_block.transpose(-1, -2))  # transposed view, no copy

        # 4. Per-edge weights, formed INDEPENDENTLY per head. Either way the smooth
        #    cutoff f_cut enters multiplicatively so a departing edge's weight
        #    vanishes continuously as it crosses r_cut, and every factor is an
        #    invariant scalar, so SO(3)-equivariance is preserved.
        # K = one slot per head, or per (head, l) with l_attention.
        H, K = self.n_heads, self.n_scores
        f_cut = self.cutoff_fn(dist_ij, self.r_cut)          # (n_e,) smooth → 0 at r_cut
        if self.aggregation == 'sum':
            #    a_e^k = s_e^k·f_cut_e — a plain signed weighted sum. No normalizer,
            #    so the aggregate is extensive in coordination and decays with
            #    absolute distance (f_cut is not divided back out).
            a = s * f_cut[:, None]                           # (n_e, H)
        else:
            #    Segment-softmax over the edges arriving at each receiver atom j,
            #    with the cutoff as a multiplicative log-bias on exp(s):
            #        a_e^k = exp(s_e^k)·f_cut_e / (Σ_{e'→j} exp(s_e'^k)·f_cut_e' + eps)
            #    A normalized weighted average → the aggregation is intensive in
            #    coordination. The +eps floor keeps it finite and continuous as a
            #    node's last edge leaves (all f_cut → 0 ⇒ Delta → 0, no jump).
            #    Per-(receiver, head) max-subtraction for numerical stability
            #    (detached, so this stays an exact softmax — invariant to the
            #    constant per-node shift). Receivers with no incoming edge keep
            #    -inf but are never gathered (every edge has a receiver in edge_j).
            ej_k = edge_j[:, None].expand(-1, K)             # (n_e, K)
            s_max = torch.full((n_atoms, K), float('-inf'), device=device, dtype=dtype
                               ).scatter_reduce(0, ej_k, s.detach(), reduce='amax',
                                                include_self=True)
            num = torch.exp(s - s_max[edge_j]) * f_cut[:, None]   # (n_e, K)
            denom = torch.zeros(n_atoms, K, device=device, dtype=dtype).scatter_add(0, ej_k, num)
            a = num / (denom[edge_j] + self.softmax_eps)     # (n_e, K) per-slot weights
            if self.msg_envelope:
                #    The normalizer just divided the absolute f_cut back out, so
                #    put it back: contribution = a_e · f_cut_e · m_e. Folded into
                #    the weight (cheaper than scaling the full-width message), so
                #    the aggregate decays with absolute distance, not just relative.
                a = a * f_cut[:, None]

        # 5. Weighted aggregation to receiver atoms. The value channels (n_base)
        #    split into H contiguous groups (full n_sph each); head h's weight gates
        #    head h's value slice, uniformly across all m (equivariant). Heads then
        #    concat back to n_base.
        #    The per-(head, l) weight is expanded to per-(head, spherical index) via
        #    l_of_s — uniform across each l's m-block, which is what keeps it
        #    equivariant. With l_attention off, l_of_s is all-zero, so this is the
        #    single per-head weight broadcast over every m (identical to before).
        hb = self.n_base // H
        a_full = a.reshape(n_e, H, self.n_scores_per_head)[:, :, self.l_of_s]  # (n_e, H, n_sph)
        contrib = h_global.reshape(n_e, H, hb, self.n_sph) * a_full[:, :, None, :]
        idx = edge_j[:, None, None, None].expand_as(contrib)
        Delta = torch.zeros(n_atoms, H, hb, self.n_sph, device=device, dtype=dtype
                            ).scatter_add(0, idx, contrib).reshape(n_atoms, self.n_base, self.n_sph)

        # 6. Gather to edges (source atom), rotate back to the edge frame.
        if getattr(self, 'edge_frame_fused', False):
            d_cos, d_sin = edge_frame_fused_single(
                Delta, edge_i, D_block, self.l_max, self.m_max)
        else:
            v_rot = torch.bmm(Delta[edge_i], D_block)        # (n_e, n_base, n_sph)
            d_cos, d_sin = self._unpack(v_rot, n_e)

        # 7. Receiver: low-rank residual (equivariant, edge frame).
        r_cos, r_sin = self.receiver(d_cos, d_sin)

        return A_cos + r_cos, A_sin + r_sin


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

        # Fold the (l_max+1)× channel-repeat into the SH gather so the forward
        # never materialises A_exp (a tensor (l_max+1)× the size of A_rot, plus
        # its grad buffer in backward). Expanded channel c reads base channel
        # c // (l_max+1); index into A_rot flattened over (base_channel, n_sph).
        # Non-persistent (deterministic from hparams), so existing checkpoints
        # keep loading under from_checkpoint's strict key check — a deliberate
        # divergence from dev, where these are persistent.
        n_sph = (l_max + 1) ** 2
        ch_src = torch.arange(self.n_ch) // (l_max + 1)          # (n_ch,)
        cos_flat = ch_src[:, None] * n_sph + cos_idx             # (n_ch, n_angular)
        sin_flat = ch_src[:, None] * n_sph + sin_idx
        self.register_buffer('cos_flat_idx', cos_flat.reshape(-1), persistent=False)
        self.register_buffer('sin_flat_idx', sin_flat.reshape(-1), persistent=False)

    def forward(self, A_rot):
        """
        Args:
            A_rot: (n_edges, 2*embed_dim, n_sph)
        Returns:
            A_cos, A_sin: (n_edges, 2*embed_dim*(l_max+1), l_max+1)
        """
        n_edges = A_rot.shape[0]
        # Gather +m (cos) and −m (sin) components straight from A_rot. The flat
        # indices already encode the (embed, l) channel repeat, so there is no
        # (l_max+1)×-larger A_exp intermediate (and no grad buffer for it).
        A_flat = A_rot.reshape(n_edges, -1)        # (n_edges, n_ch_base * n_sph)
        A_cos = (A_flat.index_select(1, self.cos_flat_idx)
                 .view(n_edges, self.n_ch, self.n_angular)) * self.cos_valid
        A_sin = (A_flat.index_select(1, self.sin_flat_idx)
                 .view(n_edges, self.n_ch, self.n_angular)) * self.sin_valid
        return A_cos, A_sin


# ---------------------------------------------------------------------------
# Output MLP
# ---------------------------------------------------------------------------


class OutputMLP(nn.Module):
    """Plain MLP readout over the per-edge invariants.

    Weights use fan-avg Gaussian init; the last layer is near-zero initialised
    so per-edge energies start close to 0 and the atomic-energy baseline
    dominates early training.
    """

    def __init__(self, dims: list, activation: nn.Module, zero_init_last: bool = True):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        for lin in self.linears:
            std = (2.0 / (lin.in_features + lin.out_features)) ** 0.5
            nn.init.normal_(lin.weight, std=std)
            nn.init.zeros_(lin.bias)
        self.activation = activation
        if zero_init_last:
            nn.init.normal_(self.linears[-1].weight, std=0.01)
            nn.init.zeros_(self.linears[-1].bias)

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if i < len(self.linears) - 1:
                x = self.activation(x)
        return x
