"""Tests for the LES integration: the model's ``return_embeddings`` hook and
the optional ``ecenet.les`` wrapper around the upstream ``les`` package.

Model-side (no optional dependency needed): the per-atom l0 read-out is
rotation-invariant and identical under ``l0_only``; l1 transforms as a vector;
the energy is unchanged by the flags; the batched paths match per-structure
forwards, including a zero-edge structure mid-batch (which exercises the
embedding slicing); forward_pbc with zero shifts matches forward.

Wrapper-side: without the upstream ``les`` package, ``import ecenet.les`` still
works and constructing `LESLongRange` raises an ImportError carrying the
install hint; with it installed, a smoke forward returns a finite energy.

With les_alpha ('iso'/'aniso'): the packed [q | u | α] layout, α's
equivariance and uniaxial single-bond form, and the wrapper's induced-dipole
energy against upstream's loop and finite differences.

Run:  python tests/test_les.py
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch

import ecenet
from ecenet import ECENet
from ecenet.les import _LES_PIN, LESLongRange

try:
    import les  # noqa: F401
    HAVE_LES = True
except ImportError:
    HAVE_LES = False

torch.manual_seed(0)
DTYPE = torch.float64
N_TYPES = 4
TOL = 1e-8

COMMON = dict(
    n_types=N_TYPES, r_cut_edge=5.0, r_cut_neighbor=4.0,
    l_max=2, n_max=3, embed_dim=8, n_layers=2, n_max_d=4,
)


def make_model(seed=0, **kwargs):
    torch.manual_seed(seed)
    m = ECENet(**COMMON, n_mp=2, bottleneck_dim=6, **kwargs).double()
    # Activate the layer stack (zero-init up-projections make it ~identity at
    # init) so the invariance tests see non-trivial features.
    for lyr in [x for stage in m.layers for x in stage]:
        with torch.no_grad():
            lyr.linear_up.weights.normal_(std=0.2)
    return m


def random_structure(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g, dtype=DTYPE) * 1.8
    types = torch.randint(0, N_TYPES, (n,), generator=g)
    return pos, types


def rand_rotation(seed=1):
    g = torch.Generator().manual_seed(seed)
    Q, R = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=DTYPE))
    Q = Q * torch.sign(torch.diag(R))
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# ── Model hook: return_embeddings / l0_only ─────────────────────────────────

def test_energy_unchanged_and_l0_only_consistent():
    m = make_model()
    pos, types = random_structure()
    e_plain = m(pos, types)
    e_full, l0_full, l1 = m(pos, types, return_embeddings=True)
    e_l0, l0_only = m(pos, types, return_embeddings=True, l0_only=True)
    assert (e_plain - e_full).abs() == 0.0 and (e_plain - e_l0).abs() == 0.0
    assert (l0_full - l0_only).abs().max() == 0.0, "l0_only changed l0"
    assert l0_full.shape == (len(types), 2 * m.embed_dim)
    assert l1.shape == (len(types), 2 * m.embed_dim, 3)
    print(f"  energy unchanged by flags; l0_only == full l0; "
          f"l0 {tuple(l0_full.shape)}, l1 {tuple(l1.shape)}")


def test_l0_rotation_invariant_l1_equivariant():
    m = make_model()
    pos, types = random_structure()
    Q = rand_rotation()
    _, l0_a, l1_a = m(pos, types, return_embeddings=True)
    _, l0_b, l1_b = m(pos @ Q.T, types, return_embeddings=True)
    d0 = (l0_a - l0_b).abs().max()
    assert d0 < TOL, f"l0 not rotation-invariant: {d0:.3e}"
    # l1 is in the real SH basis (m=-1,0,+1) = (y,z,x); mapped to Cartesian
    # (x,y,z) it must transform as a vector: l1(Qr) = Q l1(r).
    cart = [2, 0, 1]
    v_a = l1_a[:, :, cart]
    v_b = l1_b[:, :, cart]
    d1 = (v_b - torch.einsum('ncj,ij->nci', v_a, Q)).abs().max()
    # The pipeline's float64 SO(3) noise floor is ~3e-8 for l1 (vs ~4e-9 for
    # l0); a wrong basis mapping fails at O(1) (measured 0.68), so 1e-6 keeps
    # six orders of margin to a real break.
    assert d1 < 1e-6, f"l1 not vector-equivariant: {d1:.3e}"
    print(f"  l0 invariant ({d0:.1e}), l1 vector-equivariant ({d1:.1e})")


def test_batch_multi_matches_loop_with_zero_edge_structure():
    # Middle structure is a single atom (zero edges): its l0 must be zero rows
    # and must NOT shift the slices of the structures after it.
    m = make_model()
    structs = [random_structure(n, seed=s) for n, s in [(5, 1), (1, 2), (7, 3)]]
    pos_list = [p for p, _ in structs]
    types_list = [t for _, t in structs]
    energies, l0_list = m.forward_batch_multi(
        pos_list, types_list, return_embeddings=True, l0_only=True)
    assert len(l0_list) == 3
    for b, (pos, types) in enumerate(structs):
        e_ref, l0_ref = m(pos, types, return_embeddings=True, l0_only=True)
        de = (energies[b] - e_ref).abs()
        dl = (l0_list[b] - l0_ref).abs().max()
        assert de < TOL and dl < TOL, f"structure {b}: dE={de:.3e}, dl0={dl:.3e}"
    assert l0_list[1].abs().max() == 0.0, "zero-edge structure must have zero l0"
    print("  forward_batch_multi == per-structure forward (incl. zero-edge mid-batch)")


def test_forward_pbc_zero_shift_matches_forward():
    m = make_model()
    pos, types = random_structure()
    with torch.no_grad():
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(-1) + 1e-30)
        ei, ej = ((dist < m.r_cut_edge) & (dist > 1e-10)).nonzero(as_tuple=True)
        ns, nd = ((dist < m.r_cut_neighbor) & (dist > 1e-10)).nonzero(as_tuple=True)
    zed = torch.zeros(len(ei), 3, dtype=DTYPE)
    znb = torch.zeros(len(ns), 3, dtype=DTYPE)
    e_a, l0_a = m(pos, types, return_embeddings=True, l0_only=True)
    e_b, l0_b = m.forward_pbc(pos, types, ei, ej, zed, ns, nd, znb,
                              return_embeddings=True, l0_only=True)
    de, dl = (e_a - e_b).abs(), (l0_a - l0_b).abs().max()
    assert de < TOL and dl < TOL, f"pbc mismatch: dE={de:.3e}, dl0={dl:.3e}"
    print(f"  forward_pbc(zero shifts) == forward: dE={de:.1e}, dl0={dl:.1e}")


# ── Softmax (l0,l1) read-out (les_readout='softmax') ────────────────────────

def make_softmax_model(seed=0, score_std=0.0):
    m = make_model(seed=seed, les_readout='softmax')
    if score_std > 0:
        with torch.no_grad():
            m.les_score.weight.normal_(std=score_std)
            m.les_score.bias.normal_(std=score_std)
    return m


def test_softmax_readout_so3():
    # With a non-trivial (randomized) score head, the weighted read-out must
    # keep l0 invariant and l1 vector-equivariant: the weight is an invariant
    # scalar shared by both.
    m = make_softmax_model(score_std=0.5)
    pos, types = random_structure()
    Q = rand_rotation()
    _, l0_a, l1_a = m(pos, types, return_embeddings=True)
    _, l0_b, l1_b = m(pos @ Q.T, types, return_embeddings=True)
    d0 = (l0_a - l0_b).abs().max()
    assert d0 < TOL, f"softmax read-out broke l0 invariance: {d0:.3e}"
    cart = [2, 0, 1]
    d1 = (l1_b[:, :, cart]
          - torch.einsum('ncj,ij->nci', l1_a[:, :, cart], Q)).abs().max()
    assert d1 < 1e-6, f"softmax read-out broke l1 equivariance: {d1:.3e}"
    print(f"  softmax read-out: l0 invariant ({d0:.1e}), l1 equivariant ({d1:.1e})")


def test_softmax_readout_dimer_weight():
    # On a dimer each atom has exactly one in-edge, so the softmax weight has
    # a closed form: a = f_cut² / (f_cut + eps) — i.e. ≈ f_cut, the envelope.
    # les_readout is read at aggregation time, so flipping the attribute on ONE
    # model compares both paths with identical weights.
    from ecenet.radial import get_cutoff_fn
    m = make_softmax_model()          # zero-init score → s = 0 exactly
    types = torch.tensor([0, 1])
    f = get_cutoff_fn(m.cutoff_type)
    for r in (1.5, 3.0, 4.5):
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, r]], dtype=DTYPE)
        m.les_readout = 'sum'
        _, l0_sum = m(pos, types, return_embeddings=True, l0_only=True)
        m.les_readout = 'softmax'
        _, l0_soft = m(pos, types, return_embeddings=True, l0_only=True)
        f_cut = f(torch.tensor([r], dtype=DTYPE), m.r_cut_edge)
        expected = l0_sum * (f_cut ** 2 / (f_cut + 1e-6))
        d = (l0_soft - expected).abs().max()
        assert d < TOL, f"dimer weight mismatch at r={r}: {d:.3e}"
        ratio = f_cut.item() ** 2 / (f_cut.item() + 1e-6)
        print(f"  dimer r={r}: softmax/sum = {ratio:.4f} (≈ f_cut={f_cut.item():.4f})")


def test_softmax_readout_variants_consistent():
    # forward == forward_pbc(zero shifts) == forward_batch_multi, softmax
    # read-out with a randomized score, incl. a zero-edge structure mid-batch.
    m = make_softmax_model(score_std=0.5)
    structs = [random_structure(n, seed=s) for n, s in [(5, 1), (1, 2), (7, 3)]]
    pos_list = [p for p, _ in structs]
    types_list = [t for _, t in structs]
    energies, l0_list = m.forward_batch_multi(
        pos_list, types_list, return_embeddings=True, l0_only=True)
    for b, (pos, types) in enumerate(structs):
        _, l0_ref = m(pos, types, return_embeddings=True, l0_only=True)
        dl = (l0_list[b] - l0_ref).abs().max()
        assert dl < TOL, f"structure {b}: dl0={dl:.3e}"
    assert l0_list[1].abs().max() == 0.0, "zero-edge structure must have zero l0"

    pos, types = structs[0]
    with torch.no_grad():
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(-1) + 1e-30)
        ei, ej = ((dist < m.r_cut_edge) & (dist > 1e-10)).nonzero(as_tuple=True)
        ns, nd = ((dist < m.r_cut_neighbor) & (dist > 1e-10)).nonzero(as_tuple=True)
    _, l0_a = m(pos, types, return_embeddings=True, l0_only=True)
    _, l0_b = m.forward_pbc(pos, types, ei, ej, torch.zeros(len(ei), 3, dtype=DTYPE),
                            ns, nd, torch.zeros(len(ns), 3, dtype=DTYPE),
                            return_embeddings=True, l0_only=True)
    dl = (l0_a - l0_b).abs().max()
    assert dl < TOL, f"pbc mismatch: {dl:.3e}"
    print("  softmax read-out consistent across forward variants (incl. zero-edge)")


def test_edge_readout_model():
    """les_readout='edge'/'edge_basis': l0 has width 1, is SO(3)-invariant,
    consistent across forward variants, and zero for zero-edge structures."""
    for mode in ('edge', 'edge_basis'):
        m = make_model(seed=0, les_readout=mode)
        pos, types = random_structure()
        Q = rand_rotation()
        _, l0_a = m(pos, types, return_embeddings=True, l0_only=True)
        _, l0_b = m(pos @ Q.T, types, return_embeddings=True, l0_only=True)
        assert l0_a.shape == (len(types), 1), f"{mode} l0 shape: {tuple(l0_a.shape)}"
        d0 = (l0_a - l0_b).abs().max()
        assert d0 < TOL, f"{mode} read-out charge not invariant: {d0:.3e}"
        assert l0_a.abs().max() > 0, f"{mode} read-out is identically zero " \
            "(zero-init would be a gradient-free saddle — must be standard init)"

        structs = [random_structure(n, seed=s) for n, s in [(5, 1), (1, 2), (7, 3)]]
        energies, l0_list = m.forward_batch_multi(
            [p for p, _ in structs], [t for _, t in structs],
            return_embeddings=True, l0_only=True)
        for b, (pos_b, types_b) in enumerate(structs):
            _, l0_ref = m(pos_b, types_b, return_embeddings=True, l0_only=True)
            dl = (l0_list[b] - l0_ref).abs().max()
            assert dl < TOL, f"{mode} structure {b}: dl0={dl:.3e}"
        assert l0_list[1].shape == (1, 1) and l0_list[1].abs().max() == 0.0
        print(f"  {mode} read-out: width-1 invariant charge ({d0:.1e}), "
              "variants consistent, zero-edge → q=0")

    # 'edge_basis' only: the per-bond charge vanishes exactly at r_cut (the
    # dotted radial basis carries the envelope), and the head is an MLP
    # mirroring the energy readout: same dims (full m=0 invariant set in,
    # n_max_d channels out) — but standard-init last layer, not output_net's
    # near-zero init (the LES energy's q=0 saddle).
    m = make_model(seed=0, les_readout='edge_basis')
    q_dims = [(lin.in_features, lin.out_features)
              for lin in m.les_edge_charge.linears]
    e_dims = [(lin.in_features, lin.out_features)
              for lin in m.output_net.linears]
    assert q_dims == e_dims, f"head does not mirror output_net: {q_dims} vs {e_dims}"
    assert q_dims[0][0] == m.n_features_per_m
    assert q_dims[-1][1] == m.n_max_d
    q_last = m.les_edge_charge.linears[-1].weight.abs().mean()
    e_last = m.output_net.linears[-1].weight.abs().mean()
    assert q_last > 5 * e_last, \
        "edge_basis head last layer looks near-zero-init (q=0 saddle)"
    types2 = torch.tensor([0, 1])
    eps_r = 1e-9
    pos_at = torch.tensor([[0.0, 0.0, 0.0],
                           [0.0, 0.0, m.r_cut_edge - eps_r]], dtype=DTYPE)
    _, q_at = m(pos_at, types2, return_embeddings=True, l0_only=True)
    assert q_at.abs().max() < 1e-6, \
        f"edge_basis charge does not vanish at r_cut: {q_at.abs().max():.3e}"
    print("  edge_basis: MLP head mirrors output_net (standard-init last "
          "layer), per-bond charge → 0 at r_cut")

    # les_charge_scale: q scales exactly linearly (same seed → same weights),
    # and a non-edge readout warns that the scale cannot apply.
    pos, types = random_structure(seed=5)
    q1 = make_model(seed=0, les_readout='edge_basis')(
        pos, types, return_embeddings=True, l0_only=True)[1]
    qs = make_model(seed=0, les_readout='edge_basis', les_charge_scale=0.1)(
        pos, types, return_embeddings=True, l0_only=True)[1]
    dq = (qs - 0.1 * q1).abs().max()
    assert dq < TOL, f"les_charge_scale not exactly linear: {dq:.3e}"
    import warnings as _w
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter('always')
        make_model(seed=0, les_readout='sum', les_charge_scale=0.1)
    assert any('les_charge_scale' in str(r.message) for r in rec), \
        "les_charge_scale with les_readout='sum' should warn"
    print("  les_charge_scale: q(0.1) == 0.1*q(1) exactly; non-edge readout warns")


def test_edge_dipole():
    """les_dipole: l0 packed (N, 4) = [q | u]; u exactly 0 at init (zero-init
    dipole slot), q invariant / u vector-equivariant once trained, u confined
    to the bond span (planar structure → in-plane), batched variants
    consistent, and non-edge read-outs rejected."""
    for mode in ('edge', 'edge_basis'):
        m = make_model(seed=0, les_readout=mode, les_dipole=True)
        pos, types = random_structure()
        _, l0 = m(pos, types, return_embeddings=True, l0_only=True)
        assert l0.shape == (len(types), 4), f"{mode}: {tuple(l0.shape)}"
        assert l0[:, 1:].abs().max() == 0.0, f"{mode}: u must be 0 at init"
        assert l0[:, 0].abs().max() > 0, f"{mode}: q must not be 0 at init"

        # perturb the zero-init dipole slot, then check SO(3) behaviour
        with torch.no_grad():
            if mode == 'edge':
                m.les_edge_charge.weight[1].normal_(std=0.5)
            else:
                m.les_edge_charge.linears[-1].weight[m.n_max_d:].normal_(std=0.5)
        Q = rand_rotation()
        _, l0a = m(pos, types, return_embeddings=True, l0_only=True)
        _, l0b = m(pos @ Q.T, types, return_embeddings=True, l0_only=True)
        assert l0a[:, 1:].abs().max() > 0
        dq = (l0a[:, 0] - l0b[:, 0]).abs().max()
        du = (l0a[:, 1:] @ Q.T - l0b[:, 1:]).abs().max()
        assert dq < TOL, f"{mode}: q not invariant: {dq:.3e}"
        assert du < TOL, f"{mode}: u not vector-equivariant: {du:.3e}"

        # planar structure: every edge lies in z=0, so the bond-dipole span —
        # and mirror symmetry — force u_z = 0 exactly
        g = torch.Generator().manual_seed(9)
        pos_p = torch.randn(6, 3, generator=g, dtype=DTYPE) * 1.8
        pos_p[:, 2] = 0.0
        _, l0p = m(pos_p, types, return_embeddings=True, l0_only=True)
        assert l0p[:, 3].abs().max() < TOL, \
            f"{mode}: planar structure has out-of-plane dipole"
        print(f"  {mode}+dipole: packed (N,4), u=0 at init, q invariant "
              f"({dq:.1e}), u equivariant ({du:.1e}), planar → u_z=0")

    # batched slicing of the packed l0 matches per-structure forwards
    structs = [random_structure(5, seed=1), random_structure(7, seed=3)]
    m = make_model(seed=0, les_readout='edge_basis', les_dipole=True)
    with torch.no_grad():
        m.les_edge_charge.linears[-1].weight[m.n_max_d:].normal_(std=0.5)
    _, l0_list = m.forward_batch_multi([p for p, _ in structs],
                                       [t for _, t in structs],
                                       return_embeddings=True, l0_only=True)
    for b, (pos_b, types_b) in enumerate(structs):
        _, l0_ref = m(pos_b, types_b, return_embeddings=True, l0_only=True)
        dl = (l0_list[b] - l0_ref).abs().max()
        assert dl < TOL, f"structure {b}: dl0={dl:.3e}"

    try:
        ECENet(**COMMON, les_readout='sum', les_dipole=True)
        raise AssertionError("les_dipole with 'sum' should have raised")
    except ValueError as e:
        assert 'les_dipole' in str(e)
    print("  dipole read-out: batched variants consistent; non-edge rejected")


def test_dipoles_only():
    """les_charges=False (dipoles-only ablation): the head emits ONLY the
    dipole block, the q column of the packed l0 is exactly 0, and u is
    NONZERO at init — standard init, because with the charges gone the
    qᵀf_qu·u cross-term is gone too and a zero-init dipole head would sit on
    the uu-quadratic ∂E/∂u ∝ u saddle. Also: u equivariant / bond-span
    confined, les_charge_scale acts on u, E_lr equals the pure dipole-dipole
    term, the head sees a gradient at init, and the flag without les_dipole
    is rejected."""
    for mode in ('edge', 'edge_basis'):
        m = make_model(seed=0, les_readout=mode, les_dipole=True,
                       les_charges=False)
        pos, types = random_structure()
        _, l0 = m(pos, types, return_embeddings=True, l0_only=True)
        assert l0.shape == (len(types), 4), f"{mode}: {tuple(l0.shape)}"
        assert l0[:, 0].abs().max() == 0.0, f"{mode}: q must be exactly 0"
        assert l0[:, 1:].abs().max() > 0, \
            f"{mode}: u must be nonzero at init (standard init — saddle)"

        # SO(3): u vector-equivariant; planar structure → u_z = 0 exactly
        Q = rand_rotation()
        _, l0b = m(pos @ Q.T, types, return_embeddings=True, l0_only=True)
        du = (l0[:, 1:] @ Q.T - l0b[:, 1:]).abs().max()
        assert du < TOL, f"{mode}: u not vector-equivariant: {du:.3e}"
        g = torch.Generator().manual_seed(9)
        pos_p = torch.randn(6, 3, generator=g, dtype=DTYPE) * 1.8
        pos_p[:, 2] = 0.0
        _, l0p = m(pos_p, types, return_embeddings=True, l0_only=True)
        assert l0p[:, 0].abs().max() == 0.0
        assert l0p[:, 3].abs().max() < TOL, \
            f"{mode}: planar structure has out-of-plane dipole"

        # les_charge_scale now acts on the dipole
        m2 = make_model(seed=0, les_readout=mode, les_dipole=True,
                        les_charges=False, les_charge_scale=0.1)
        _, l0s = m2(pos, types, return_embeddings=True, l0_only=True)
        ds = (l0s - 0.1 * l0).abs().max()
        assert ds < TOL, f"{mode}: les_charge_scale on u broken: {ds:.3e}"
        print(f"  {mode}+dipoles-only: q ≡ 0, u ≠ 0 at init, equivariant "
              f"({du:.1e}), planar → u_z=0, scale acts on u")

    if HAVE_LES:
        # E_lr equals the pure dipole-dipole term (upstream fed q = 0
        # directly), and the head has a nonzero gradient at init — the
        # standard init really is off the saddle
        m = make_model(seed=0, les_readout='edge_basis', les_dipole=True,
                       les_charges=False)
        lr = LESLongRange().double()
        pos, types = random_structure()
        _, l0 = m(pos, types, return_embeddings=True, l0_only=True)
        e = lr(l0, pos, **m.les_flags)
        batch0 = torch.zeros(len(types), dtype=torch.long)
        e_uu = lr.les(latent_charges=torch.zeros(len(types), dtype=DTYPE),
                      latent_dipoles=l0[:, 1:4].detach(), positions=pos,
                      cell=None, batch=batch0, compute_energy=True)['E_lr']
        d = (e.sum() - e_uu.sum()).abs()
        assert d < 1e-10, f"E_lr != pure uu term: {d:.3e}"
        assert float(e.abs().sum()) > 0, "E_lr identically zero at init"
        e.sum().backward()
        gmax = max(float(p.grad.abs().max())
                   for p in m.les_edge_charge.parameters()
                   if p.grad is not None)
        assert gmax > 0, "dipole head has zero gradient at init (saddle)"
        print(f"  E_lr == pure uu term ({float(d):.1e}); "
              f"head grad at init {gmax:.2e}")

    try:
        ECENet(**COMMON, les_readout='edge_basis', les_charges=False)
        raise AssertionError("les_charges=False without les_dipole should raise")
    except ValueError as err:
        assert 'les_charges' in str(err)
    print("  dipoles-only: les_charges=False without les_dipole rejected")


def test_edge_dipole_les_energy():
    """Wrapper with les_dipole: the VECTORIZED isolated path (masked
    f_qq/f_qu/f_uu on the concatenated batch) equals upstream's per-structure
    loop in energies and position gradients; u=0 reduces exactly to the
    charges-only energy; coincident cross-structure atoms stay finite; the
    flag without l0_is_charge is rejected."""
    if not HAVE_LES:
        print("  skipped (`les` not installed)")
        return
    lr = LESLongRange().double()
    g = torch.Generator().manual_seed(12)
    sizes = [4, 6]
    pos = torch.randn(sum(sizes), 3, generator=g, dtype=DTYPE) * 2.0
    q = torch.randn(sum(sizes), generator=g, dtype=DTYPE) * 0.3
    u = torch.randn(sum(sizes), 3, generator=g, dtype=DTYPE) * 0.2
    batch = torch.cat([torch.full((n,), b, dtype=torch.long)
                       for b, n in enumerate(sizes)])
    packed = torch.cat([q[:, None], u], dim=1)

    p_a = pos.clone().requires_grad_(True)
    e = lr(packed, p_a, batch=batch, n_struct=2,
           l0_is_charge=True, les_dipole=True)
    f_a = torch.autograd.grad(e.sum(), p_a)[0]
    p_b = pos.clone().requires_grad_(True)
    res = lr.les(latent_charges=q, latent_dipoles=u, positions=p_b,
                 cell=None, batch=batch, compute_energy=True)
    f_b = torch.autograd.grad(res['E_lr'].sum(), p_b)[0]
    de = (e - res['E_lr'].reshape(e.shape)).abs().max()
    df = (f_a - f_b).abs().max()
    assert de < 1e-10 and df < 1e-10, \
        f"vectorized dipole path != upstream loop: dE={de:.3e}, dF={df:.3e}"

    packed0 = torch.cat([q[:, None], torch.zeros_like(u)], dim=1)
    e0 = lr(packed0, pos, batch=batch, n_struct=2,
            l0_is_charge=True, les_dipole=True)
    eq = lr(q[:, None], pos, batch=batch, n_struct=2, l0_is_charge=True)
    d0 = (e0 - eq).abs().max()
    assert d0 < 1e-10, f"u=0 does not reduce to charges-only: {d0:.3e}"
    assert (e - e0).abs().max() > 0, "dipoles changed nothing"

    # exactly coincident atoms in DIFFERENT structures: the grid shift must
    # keep f_qu/f_uu finite exactly as it does f_qq
    pos_c = pos.clone()
    pos_c[sizes[0]] = pos_c[0]
    p_c = pos_c.clone().requires_grad_(True)
    e_c = lr(packed, p_c, batch=batch, n_struct=2,
             l0_is_charge=True, les_dipole=True)
    f_c = torch.autograd.grad(e_c.sum(), p_c)[0]
    assert torch.isfinite(e_c).all() and torch.isfinite(f_c).all(), \
        "coincident cross-structure atoms NaN'd the dipole path"
    res_c = lr.les(latent_charges=q, latent_dipoles=u, positions=pos_c,
                   cell=None, batch=batch, compute_energy=True)
    dc = (e_c - res_c['E_lr'].reshape(e_c.shape)).abs().max()
    assert dc < 1e-10, f"coincident case != upstream: {dc:.3e}"

    try:
        lr(packed, pos, batch=batch, n_struct=2, les_dipole=True)
        raise AssertionError("les_dipole without l0_is_charge should raise")
    except ValueError as err:
        assert 'l0_is_charge' in str(err)
    print(f"  vectorized dipole path == upstream loop (dE={de:.1e}, "
          f"dF={df:.1e}); u=0 → charges-only ({d0:.1e}); coincident finite "
          f"({dc:.1e}); flag misuse rejected")


def test_edge_alpha():
    """les_alpha: l0 packed [q | u | α]; α exactly 0 at init (zero-init
    slots); 'iso' an invariant scalar, 'aniso' a symmetric rank-2 tensor
    transforming as α' = RαRᵀ; a lone bond gives a uniaxial tensor with r̂
    as principal axis (the diatomic α_∥/α_⊥); les_charge_scale scales [q | u]
    but not α; batched variants consistent; non-edge read-outs rejected."""
    from ecenet.les import total_polarizability, unpack_l0
    for mode in ('edge', 'edge_basis'):
        for kind, width in (('iso', 1), ('aniso', 9)):
            m = make_model(seed=0, les_readout=mode, les_dipole=True,
                           les_alpha=kind)
            pos, types = random_structure()
            _, l0 = m(pos, types, return_embeddings=True, l0_only=True)
            assert l0.shape == (len(types), 4 + width), \
                f"{mode}/{kind}: {tuple(l0.shape)}"
            assert l0[:, 4:].abs().max() == 0.0, f"{mode}/{kind}: α≠0 at init"
            assert l0[:, 0].abs().max() > 0, f"{mode}/{kind}: q=0 at init"

            # perturb the zero-init α slots, then check SO(3) behaviour
            with torch.no_grad():
                if mode == 'edge':
                    m.les_edge_charge.weight[2:].normal_(std=0.5)
                else:
                    last = m.les_edge_charge.linears[-1]
                    last.weight[2 * m.n_max_d:].normal_(std=0.5)
            Q = rand_rotation()
            _, l0a = m(pos, types, return_embeddings=True, l0_only=True)
            _, l0b = m(pos @ Q.T, types, return_embeddings=True, l0_only=True)
            qa, ua, aa = unpack_l0(l0a, **m.les_flags)
            qb, ub, ab = unpack_l0(l0b, **m.les_flags)
            assert aa.abs().max() > 0, f"{mode}/{kind}: α still 0"
            dq = (qa - qb).abs().max()
            du = (ua @ Q.T - ub).abs().max()
            if kind == 'iso':
                assert aa.shape == (len(types),)
                da = (aa - ab).abs().max()
            else:
                assert aa.shape == (len(types), 3, 3)
                sym = (aa - aa.transpose(1, 2)).abs().max()
                assert sym == 0.0, f"{mode}: α not symmetric: {sym:.3e}"
                da = (Q @ aa @ Q.T - ab).abs().max()
            assert dq < TOL and du < TOL, f"{mode}/{kind}: q/u broke"
            assert da < TOL, f"{mode}/{kind}: α not equivariant: {da:.3e}"
            P = total_polarizability(aa)
            assert P.shape == (1, 3, 3)
            print(f"  {mode}+α({kind}): packed (N,{4 + width}), α=0 at init, "
                  f"equivariant ({da:.1e})")

    # a single bond: α_i = a·I + b·(r̂r̂ᵀ − I/3) is uniaxial about r̂ — r̂ is
    # an eigenvector and the two perpendicular eigenvalues coincide
    m = make_model(seed=0, les_readout='edge_basis', les_dipole=False,
                   les_alpha='aniso')
    with torch.no_grad():
        m.les_edge_charge.linears[-1].weight[m.n_max_d:].normal_(std=0.5)
    pos2 = torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.9, -0.5]], dtype=DTYPE)
    types2 = torch.tensor([0, 1])
    _, l0 = m(pos2, types2, return_embeddings=True, l0_only=True)
    _, _, a2 = unpack_l0(l0, **m.les_flags)
    r_hat = pos2[1] - pos2[0]
    r_hat = r_hat / r_hat.norm()
    for i in range(2):
        A = a2[i]
        lam = r_hat @ A @ r_hat
        d_axis = (A @ r_hat - lam * r_hat).abs().max()
        assert d_axis < TOL, f"atom {i}: r̂ not an eigenvector ({d_axis:.1e})"
        evals = torch.linalg.eigvalsh(A)
        # two of the three eigenvalues coincide (perpendicular pair)
        gaps = sorted([(evals[1] - evals[0]).abs(), (evals[2] - evals[1]).abs()])
        assert gaps[0] < 1e-9, f"atom {i}: not uniaxial: {evals.tolist()}"
    assert (a2[0] - a2[0].trace() / 3 * torch.eye(3, dtype=DTYPE)).abs().max() > 0, \
        "traceless part vanished"
    print("  lone bond → uniaxial α with r̂ as principal axis")

    # les_charge_scale scales [q | u], leaves α alone
    ms = make_model(seed=0, les_readout='edge_basis', les_dipole=True,
                    les_alpha='iso', les_charge_scale=0.1)
    m1 = make_model(seed=0, les_readout='edge_basis', les_dipole=True,
                    les_alpha='iso', les_charge_scale=1.0)
    for mm in (ms, m1):
        with torch.no_grad():
            mm.les_edge_charge.linears[-1].weight[2 * mm.n_max_d:].normal_(
                std=0.5, generator=torch.Generator().manual_seed(4))
            mm.les_edge_charge.linears[-1].weight[mm.n_max_d:2 * mm.n_max_d]\
                .normal_(std=0.5, generator=torch.Generator().manual_seed(5))
    pos, types = random_structure()
    _, l0s = ms(pos, types, return_embeddings=True, l0_only=True)
    _, l01 = m1(pos, types, return_embeddings=True, l0_only=True)
    dqu = (l0s[:, :4] - 0.1 * l01[:, :4]).abs().max()
    dal = (l0s[:, 4:] - l01[:, 4:]).abs().max()
    assert dqu < TOL and dal < TOL, f"scale: dqu={dqu:.1e}, dα={dal:.1e}"
    print("  les_charge_scale: [q | u] scaled, α untouched")

    # batched slicing of the packed l0 matches per-structure forwards
    structs = [random_structure(5, seed=1), random_structure(7, seed=3)]
    m = make_model(seed=0, les_readout='edge_basis', les_dipole=True,
                   les_alpha='aniso')
    with torch.no_grad():
        m.les_edge_charge.linears[-1].weight[m.n_max_d:].normal_(std=0.5)
    _, l0_list = m.forward_batch_multi([p for p, _ in structs],
                                       [t for _, t in structs],
                                       return_embeddings=True, l0_only=True)
    for b, (pos_b, types_b) in enumerate(structs):
        _, l0_ref = m(pos_b, types_b, return_embeddings=True, l0_only=True)
        dl = (l0_list[b] - l0_ref).abs().max()
        assert dl < TOL, f"structure {b}: dl0={dl:.3e}"

    for bad in (dict(les_readout='sum', les_alpha='iso'),
                dict(les_readout='edge_basis', les_alpha='full')):
        try:
            ECENet(**COMMON, **bad)
            raise AssertionError(f"{bad} should have raised")
        except ValueError as e:
            assert 'les_alpha' in str(e)
    print("  α read-out: batched variants consistent; bad configs rejected")


def test_edge_alpha_les_energy():
    """Wrapper with les_alpha: the vectorized isolated path (field of the
    fixed multipoles from the masked f_qu/f_uu kernels, then −½E·α·E) equals
    upstream's per-structure loop in energies and position gradients for
    'iso'/'aniso' with and without dipoles; forces pass a finite-difference
    check independent of upstream; α=0 reduces exactly to the no-α energy;
    the periodic path passes α through; born_charges with α goes through
    upstream's induced dipoles; flag misuse is rejected."""
    if not HAVE_LES:
        print("  skipped (`les` not installed)")
        return
    from ecenet.les import total_polarizability
    lr = LESLongRange().double()
    g = torch.Generator().manual_seed(12)
    sizes = [4, 6]
    N = sum(sizes)
    pos = torch.randn(N, 3, generator=g, dtype=DTYPE) * 2.0
    q = torch.randn(N, generator=g, dtype=DTYPE) * 0.3
    u = torch.randn(N, 3, generator=g, dtype=DTYPE) * 0.2
    a_iso = torch.rand(N, generator=g, dtype=DTYPE) * 0.5
    S = torch.randn(N, 3, 3, generator=g, dtype=DTYPE)
    a_ani = 0.3 * (S @ S.transpose(1, 2))          # symmetric positive
    batch = torch.cat([torch.full((n,), b, dtype=torch.long)
                       for b, n in enumerate(sizes)])

    worst = 0.0
    for dip in (False, True):
        for kind, alpha in (('iso', a_iso), ('aniso', a_ani)):
            cols = [q[:, None]] + ([u] if dip else []) + [alpha.reshape(N, -1)]
            packed = torch.cat(cols, dim=1)
            p_a = pos.clone().requires_grad_(True)
            e = lr(packed, p_a, batch=batch, n_struct=2, l0_is_charge=True,
                   les_dipole=dip, les_alpha=kind)
            f_a = torch.autograd.grad(e.sum(), p_a)[0]
            p_b = pos.clone().requires_grad_(True)
            res = lr.les(latent_charges=q, latent_dipoles=u if dip else None,
                         latent_alphas=alpha, positions=p_b, cell=None,
                         batch=batch, compute_energy=True)
            f_b = torch.autograd.grad(res['E_lr'].sum(), p_b)[0]
            de = (e - res['E_lr'].reshape(e.shape)).abs().max()
            df = (f_a - f_b).abs().max()
            assert de < 1e-10 and df < 1e-10, \
                f"{kind}/dip={dip}: != upstream loop: dE={de:.3e}, dF={df:.3e}"
            worst = max(worst, float(de), float(df))
            # α=0 reduces to the no-α energy; α≠0 changes it
            cols0 = cols[:-1] + [torch.zeros_like(cols[-1])]
            e0 = lr(torch.cat(cols0, dim=1), pos, batch=batch, n_struct=2,
                    l0_is_charge=True, les_dipole=dip, les_alpha=kind)
            base = (torch.cat(cols[:-1], dim=1) if dip else q[:, None])
            eb = lr(base, pos, batch=batch, n_struct=2, l0_is_charge=True,
                    les_dipole=dip)
            d0 = (e0 - eb).abs().max()
            assert d0 < 1e-12, f"{kind}/dip={dip}: α=0 ≠ no-α: {d0:.3e}"
            assert (e - e0).abs().max() > 1e-3, "α changed nothing"

    # finite-difference forces (aniso + dipoles), independent of upstream
    packed = torch.cat([q[:, None], u, a_ani.reshape(N, 9)], dim=1)
    flags = dict(l0_is_charge=True, les_dipole=True, les_alpha='aniso')
    p_a = pos.clone().requires_grad_(True)
    grad = torch.autograd.grad(lr(packed, p_a, batch=batch, n_struct=2,
                                  **flags).sum(), p_a)[0]
    h = 1e-5
    max_fd = 0.0
    for (i, c) in [(0, 0), (3, 2), (7, 1)]:
        es = []
        for sgn in (+1, -1):
            pp = pos.clone()
            pp[i, c] += sgn * h
            es.append(lr(packed, pp, batch=batch, n_struct=2, **flags).sum())
        fd = (es[0] - es[1]) / (2 * h)
        max_fd = max(max_fd, abs(float(fd - grad[i, c])))
    assert max_fd < 1e-7, f"FD gradient mismatch with α: {max_fd:.2e}"

    # periodic path: α passes straight through to upstream's Ewald
    cell = (torch.eye(3, dtype=DTYPE) * 8.0).expand(2, 3, 3).contiguous()
    e_p = lr(packed, pos, cell=cell, batch=batch, **flags)
    res_p = lr.les(latent_charges=q, latent_dipoles=u, latent_alphas=a_ani,
                   positions=pos, cell=cell, batch=batch)
    dp = (e_p - res_p['E_lr']).abs().max()
    assert dp < 1e-12, f"periodic α path != upstream: {dp:.3e}"

    # born_charges: α=0 through upstream's forward equals the direct BEC
    # module call (no-α path); α≠0 adds the induced dipoles and stays finite
    # (latents must be functions of the positions, as they are through the
    # model: upstream differentiates the polarization w.r.t. r)
    n1 = sizes[0]
    p1 = pos[:n1].clone().requires_grad_(True)
    q1 = q[:n1] + 0.1 * p1.norm(dim=1)
    u1 = u[:n1] + 0.05 * p1
    a1 = a_ani[:n1] + 0.05 * p1[:, :, None] * p1[:, None, :]
    l0_1 = torch.cat([q1[:, None], u1, a1.reshape(n1, 9)], dim=1)
    Z_a = lr.born_charges(l0_1, p1, cell=None, **flags)
    l0_0 = torch.cat([l0_1[:, :4], torch.zeros(n1, 9, dtype=DTYPE)], dim=1)
    Z_0 = lr.born_charges(l0_0, p1, cell=None, **flags)
    Z_ref = lr.born_charges(l0_1[:, :4], p1, cell=None, l0_is_charge=True,
                            les_dipole=True)
    dz0 = (Z_0 - Z_ref).abs().max()
    assert Z_a.shape == (n1, 3, 3) and torch.isfinite(Z_a).all()
    assert dz0 < 1e-12, f"born_charges α=0 != no-α path: {dz0:.3e}"
    assert (Z_a - Z_0).abs().max() > 1e-6, "induced dipoles absent from Z*"

    # total polarizability: per-structure sums, iso → multiple of identity
    P = total_polarizability(a_ani, batch=batch, n_struct=2)
    assert P.shape == (2, 3, 3)
    assert (P[0] - a_ani[:n1].sum(0)).abs().max() < 1e-12
    P_iso = total_polarizability(a_iso, batch=batch)
    assert (P_iso[1] - a_iso[n1:].sum() * torch.eye(3, dtype=DTYPE)).abs().max() < 1e-12

    try:
        lr(packed, pos, batch=batch, n_struct=2, les_dipole=True,
           les_alpha='aniso')
        raise AssertionError("les_alpha without l0_is_charge should raise")
    except ValueError as err:
        assert 'l0_is_charge' in str(err)
    print(f"  vectorized α path == upstream loop (worst {worst:.1e}); FD "
          f"forces {max_fd:.1e}; periodic {dp:.1e}; born_charges α=0 "
          f"{dz0:.1e}; polarizability sums OK; flag misuse rejected")


def test_edge_readout_les_energy():
    """l0_is_charge=True: isolated fast path and upstream's latent_charges
    path agree, and the LES module holds no parameters (head bypassed)."""
    if not HAVE_LES:
        print("  skipped (`les` not installed)")
        return
    torch.manual_seed(4)
    lr = LESLongRange().double()
    g = torch.Generator().manual_seed(11)
    sizes = [4, 6]
    pos = torch.randn(sum(sizes), 3, generator=g, dtype=DTYPE) * 2.0
    q_in = torch.randn(sum(sizes), 1, generator=g, dtype=DTYPE) * 0.3
    batch = torch.cat([torch.full((n,), b, dtype=torch.long)
                       for b, n in enumerate(sizes)])

    p_a = pos.clone().requires_grad_(True)
    e_fast, q_out = lr(q_in, p_a, batch=batch, n_struct=2,
                       return_charges=True, l0_is_charge=True)
    f_a = torch.autograd.grad(e_fast.sum(), p_a)[0]
    assert (q_out - q_in.reshape(-1)).abs().max() == 0.0, \
        "l0_is_charge must return the input charges untouched"
    assert len(list(lr.parameters())) == 0, \
        "LES module should hold no parameters when the head is bypassed"

    p_b = pos.clone().requires_grad_(True)
    res = lr.les(latent_charges=q_in.reshape(-1), positions=p_b, cell=None,
                 batch=batch, compute_energy=True)
    f_b = torch.autograd.grad(res["E_lr"].sum(), p_b)[0]
    de = (e_fast - res["E_lr"].reshape(e_fast.shape)).abs().max()
    df = (f_a - f_b).abs().max()
    assert de < 1e-10 and df < 1e-10, f"mismatch: dE={de:.3e}, dF={df:.3e}"
    print(f"  l0_is_charge: fast path == upstream latent_charges "
          f"(dE={de:.1e}, dF={df:.1e}), module parameter-free")


def test_les_readout_validation():
    try:
        ECENet(**COMMON, les_readout='mean')
        raise AssertionError("les_readout='mean' should have raised")
    except ValueError as e:
        assert 'les_readout' in str(e)
    print("  invalid les_readout rejected")


# ── Wrapper: lazy import / upstream smoke ───────────────────────────────────

def test_lazy_import():
    # The wrapper module and the package attribute resolve without `les`.
    assert ecenet.LESLongRange is LESLongRange


def test_missing_dep_error():
    if HAVE_LES:
        print("  skipped (`les` is installed)")
        return
    try:
        LESLongRange()
    except ImportError as e:
        msg = str(e)
        assert "ChengUCB/les" in msg and "pip install" in msg and _LES_PIN in msg
        assert "CC BY-NC" in msg
    else:
        raise AssertionError("LESLongRange() should raise ImportError without `les`")


def test_smoke_forward():
    if not HAVE_LES:
        print("  skipped (`les` not installed)")
        return
    torch.manual_seed(0)
    lr = LESLongRange().double()
    m = make_model()
    pos, types = random_structure()
    p = pos.clone().requires_grad_(True)
    e_sr, l0 = m(p, types, return_embeddings=True, l0_only=True)
    e = e_sr + lr(l0, p).sum()
    assert torch.isfinite(e).all(), f"non-finite total energy: {e}"
    # cell=None (fast path, no per-structure det check) must equal an explicit
    # zero cell (upstream's det<1e-6 branch) — both mean isolated. The two are
    # independent implementations with different summation orders, so agreement
    # is to float64 rounding, not bitwise.
    with torch.no_grad():
        e_none = lr(l0, p)
        e_zero = lr(l0, p, cell=torch.zeros(1, 3, 3, dtype=p.dtype))
    dz = (e_none - e_zero).abs().max()
    assert dz < 1e-14, f"cell=None != zero cell: {dz:.3e}"
    f = -torch.autograd.grad(e, p)[0]
    assert f.shape == pos.shape and torch.isfinite(f).all()
    print(f"  smoke: E={e.item():.6f} eV, |F|max={f.abs().max():.3f}")


def test_isolated_batched_matches_upstream_loop():
    """The wrapper's vectorized isolated path (one masked full-batch quadratic
    form) must equal upstream's per-structure Python loop exactly — energies,
    latent charges, and position gradients — including a single-atom
    structure mid-batch."""
    if not HAVE_LES:
        print("  skipped (`les` not installed)")
        return
    torch.manual_seed(2)
    lr = LESLongRange().double()
    g = torch.Generator().manual_seed(9)
    sizes = [5, 1, 7]
    pos = torch.randn(sum(sizes), 3, generator=g, dtype=DTYPE) * 2.0
    l0 = torch.randn(sum(sizes), 16, generator=g, dtype=DTYPE)
    batch = torch.cat([torch.full((n,), b, dtype=torch.long)
                       for b, n in enumerate(sizes)])
    with torch.no_grad():
        lr(l0, pos, batch=batch, n_struct=len(sizes))   # materialise the head
        for p in lr.parameters():
            p.add_(0.1 * torch.randn_like(p))

    p_a = pos.clone().requires_grad_(True)
    e_fast, q_fast = lr(l0, p_a, batch=batch, n_struct=len(sizes),
                        return_charges=True)
    f_a = torch.autograd.grad(e_fast.sum(), p_a)[0]

    p_b = pos.clone().requires_grad_(True)
    res = lr.les(desc=l0, positions=p_b, cell=None, batch=batch,
                 compute_energy=True)                    # upstream's own loop
    f_b = torch.autograd.grad(res["E_lr"].sum(), p_b)[0]

    de = (e_fast - res["E_lr"].reshape(e_fast.shape)).abs().max()
    dq = (q_fast.reshape(-1) - res["latent_charges"].reshape(-1)).abs().max()
    df = (f_a - f_b).abs().max()
    # The anti-coincidence grid shift costs ~eps·offset of fp cancellation
    # noise in the intra-structure distances, so equality is to ~1e-12 in
    # float64 rather than bit-exact. A real break is orders louder.
    assert de < 1e-10, f"energy mismatch vs upstream loop: {de:.3e}"
    assert dq == 0.0, f"charge mismatch vs upstream loop: {dq:.3e}"
    assert df < 1e-10, f"gradient mismatch vs upstream loop: {df:.3e}"
    print(f"  vectorized isolated path == upstream loop "
          f"(dE={de:.1e}, dq={dq:.1e}, dF={df:.1e})")


def test_isolated_batched_coincident_cross_atoms():
    """Structures in a batch are individually centered, so atoms of DIFFERENT
    structures can sit at identical coordinates. The dense kernel used to
    produce inf·0 = NaN there (masking cannot clean a NaN); the grid shift
    must keep energies and gradients finite and equal to upstream's loop."""
    if not HAVE_LES:
        print("  skipped (`les` not installed)")
        return
    torch.manual_seed(3)
    lr = LESLongRange().double()
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0], [0.0, 1.5, 0.0]], dtype=DTYPE)
    l0 = torch.randn(4, 8, dtype=DTYPE)
    batch = torch.tensor([0, 0, 1, 1])
    with torch.no_grad():
        lr(l0[:2], pos[:2])                       # materialise the head
        for p in lr.parameters():
            p.add_(0.1 * torch.randn_like(p))

    p_a = pos.clone().requires_grad_(True)
    e_fast = lr(l0, p_a, batch=batch, n_struct=2)
    assert torch.isfinite(e_fast).all(), f"NaN with coincident cross atoms: {e_fast}"
    f_a = torch.autograd.grad(e_fast.sum(), p_a)[0]
    assert torch.isfinite(f_a).all(), "NaN gradient with coincident cross atoms"

    p_b = pos.clone().requires_grad_(True)
    res = lr.les(desc=l0, positions=p_b, cell=None, batch=batch,
                 compute_energy=True)
    f_b = torch.autograd.grad(res["E_lr"].sum(), p_b)[0]
    de = (e_fast - res["E_lr"].reshape(e_fast.shape)).abs().max()
    df = (f_a - f_b).abs().max()
    assert de < 1e-10 and df < 1e-10, f"mismatch: dE={de:.3e}, dF={df:.3e}"
    print(f"  coincident cross-structure atoms: finite and == upstream "
          f"(dE={de:.1e}, dF={df:.1e})")


if __name__ == "__main__":
    print(f"LES integration tests (upstream `les` installed: {HAVE_LES})")
    test_energy_unchanged_and_l0_only_consistent()
    test_l0_rotation_invariant_l1_equivariant()
    test_batch_multi_matches_loop_with_zero_edge_structure()
    test_forward_pbc_zero_shift_matches_forward()
    test_softmax_readout_so3()
    test_softmax_readout_dimer_weight()
    test_softmax_readout_variants_consistent()
    test_edge_readout_model()
    test_edge_dipole()
    test_dipoles_only()
    test_edge_dipole_les_energy()
    test_edge_alpha()
    test_edge_alpha_les_energy()
    test_edge_readout_les_energy()
    test_les_readout_validation()
    test_lazy_import()
    test_missing_dep_error()
    test_smoke_forward()
    test_isolated_batched_matches_upstream_loop()
    test_isolated_batched_coincident_cross_atoms()
    print("All tests passed.")
