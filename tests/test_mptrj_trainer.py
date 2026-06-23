"""Synthetic smoke + finite-difference test for train_ecenet_mptrj.py.

No MPtrj download needed: builds a handful of random periodic structures and
  1. runs the trainer end-to-end (a few epochs, stress on) via injected
     structures, checking it runs and the loss decreases;
  2. validates the stress + force plumbing against finite differences on a
     fresh model — confirming the trainer's strain convention and ASE-based
     PBC neighbor lists agree with the model (same check as test_stress_fd.py
     but standalone, no checkpoint).

Run:  /opt/homebrew/Caskroom/miniconda/base/envs/open/bin/python test_mptrj_trainer.py
"""

import os
import sys  # repo root + scripts/ on path (imports ecenet and the scripts/ trainer)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))   # train_ecenet_mptrj lives in scripts/


import numpy as np
import torch
from train_ecenet_mptrj import (
    build_topology,
    train_ecenet_mptrj,
)

from ecenet import ECENet, elements


def build_type_map(structures):
    """Dense type map over a list of structure dicts (thin test wrapper around
    the shared ecenet.elements.build_type_map, which takes a flat iterable)."""
    return elements.build_type_map(
        z for s in structures for z in s['numbers'])

DTYPE = torch.float64
DEVICE = torch.device('cpu')   # MPS has no float64; FD needs the precision
Z_CHOICES = [1, 6, 8, 14]      # H, C, O, Si  → n_types = 4


def make_structures(n, seed=0, n_atoms_range=(4, 8), box=(7.0, 8.0)):
    """Random periodic structures with random energy/forces/stress (kBar)."""
    rng = np.random.RandomState(seed)
    structs = []
    for _ in range(n):
        na = rng.randint(*n_atoms_range)
        L = rng.uniform(*box)
        cell = np.diag([L, L, L]).astype(np.float64)
        # jitter the cell off-diagonal a little to exercise triclinic shifts
        cell[0, 1] = rng.uniform(-0.5, 0.5)
        cell[1, 2] = rng.uniform(-0.5, 0.5)
        frac = rng.uniform(0, 1, size=(na, 3))
        positions = frac @ cell
        numbers = rng.choice(Z_CHOICES, size=na).astype(np.int64)
        structs.append({
            'numbers': numbers,
            'positions': positions,
            'cell': cell,
            'pbc': True,
            'energy': float(rng.uniform(-5, 5) * na),
            'forces': rng.uniform(-1, 1, size=(na, 3)).astype(np.float64),
            'stress': rng.uniform(-50, 50, size=(3, 3)).astype(np.float64),  # kBar
            'n_atoms': na,
        })
    return structs


def test_smoke_train():
    print("=== Smoke: end-to-end training (energy + force + stress) ===")
    train_structs = make_structures(16, seed=1)
    test_structs = make_structures(4, seed=2)

    _, results = train_ecenet_mptrj(
        train_structures=train_structs, test_structures=test_structs,
        n_val=3,
        l_max=2, n_max=2, embed_dim=8, n_layers=1, n_max_d=4,
        r_cut_edge=4.0, r_cut_neighbor=3.5,
        stress_weight=0.1, force_weight=1.0, energy_weight=1.0,
        n_epochs=4, batch_size=4, lr=5e-3,
        dtype=DTYPE, device=DEVICE, seed=0, verbose=True,
    )
    for k in ('test_energy_mae', 'test_force_mae', 'test_stress_mae'):
        assert np.isfinite(results[k]), f"{k} not finite: {results[k]}"
    print(f"  n_types={results['n_types']}  results OK: "
          f"E={results['test_energy_mae']:.3f} F={results['test_force_mae']:.3f} "
          f"S={results['test_stress_mae']:.3e}\n")


def make_teacher_targets(structs, seed=11):
    """Overwrite energy/forces with the output of a fixed random ECENet 'teacher'
    so the targets are learnable by the same architecture (random targets are not).
    """
    type_map = build_type_map(structs)
    torch.manual_seed(seed)
    teacher = ECENet(n_types=len(type_map), r_cut_edge=4.0, r_cut_neighbor=3.5,
                     l_max=2, n_max=2, embed_dim=8, n_layers=1, n_max_d=4).double().to(DEVICE)
    for p in teacher.parameters():
        with torch.no_grad():
            p.add_(0.1 * torch.randn_like(p))
    for s in structs:
        ei, ej, she, ni, nj, shn = build_topology(
            s['positions'], s['cell'], True, 4.0, 3.5, DEVICE, DTYPE)
        pos = torch.tensor(s['positions'], dtype=DTYPE, device=DEVICE, requires_grad=True)
        types = torch.tensor([type_map[int(z)] for z in s['numbers']],
                             dtype=torch.long, device=DEVICE)
        e = teacher.forward_pbc(pos, types, ei, ej, she, ni, nj, shn)
        f = -torch.autograd.grad(e, pos)[0]
        s['energy'] = float(e.item())
        s['forces'] = f.detach().cpu().numpy()
        s['stress'] = None
    return structs


def test_loss_decreases():
    print("=== Learning: loss drops fitting a teacher model (baseline, no MoE) ===")
    import train_ecenet_mptrj as T
    structs = make_teacher_targets(make_structures(16, seed=3))
    losses = []
    orig = T.print_flush

    def capture(*a, **k):
        s = ' '.join(str(x) for x in a)
        if 'Epoch' in s and 'loss=' in s:
            losses.append(float(s.split('loss=')[1].split('|')[0]))
        orig(*a, **k)

    T.print_flush = capture
    try:
        train_ecenet_mptrj(
            train_structures=structs, n_val=3,
            l_max=2, n_max=2, embed_dim=8, n_layers=1, n_max_d=4,
            r_cut_edge=4.0, r_cut_neighbor=3.5,
            stress_weight=0.0, n_epochs=20, batch_size=4, lr=1e-2,
            eval_every=2, dtype=DTYPE, device=DEVICE, seed=0, verbose=True,
        )
    finally:
        T.print_flush = orig
    assert len(losses) >= 3, f"expected several epoch losses, got {losses}"
    assert losses[-1] < 0.5 * losses[0], \
        f"loss did not drop enough: {losses[0]:.3f} -> {losses[-1]:.3f}"
    print(f"  loss {losses[0]:.3f} -> {losses[-1]:.3f}  OK\n")


def test_stress_and_force_fd():
    print("=== FD check: autograd stress/forces vs finite differences ===")
    structs = make_structures(1, seed=7, n_atoms_range=(6, 7))
    s = structs[0]
    type_map = build_type_map(structs)
    n_types = len(type_map)

    torch.manual_seed(0)
    model = ECENet(n_types=n_types, r_cut_edge=4.0, r_cut_neighbor=3.5,
                   l_max=2, n_max=2, embed_dim=8, n_layers=1, n_max_d=4).double().to(DEVICE)
    # Randomize output layer so energies aren't ~0 and FD is well-conditioned.
    for p in model.parameters():
        with torch.no_grad():
            p.add_(0.05 * torch.randn_like(p))

    edge_i, edge_j, she, nb_src, nb_dst, shn = build_topology(
        s['positions'], s['cell'], True, 4.0, 3.5, DEVICE, DTYPE)
    pos = torch.tensor(s['positions'], dtype=DTYPE, device=DEVICE)
    types = torch.tensor([type_map[int(z)] for z in s['numbers']],
                         dtype=torch.long, device=DEVICE)
    volume = abs(np.linalg.det(s['cell']))
    print(f"  {s['n_atoms']} atoms, {len(edge_i)} edges, {len(nb_src)} nb pairs, V={volume:.2f} Å³")

    def energy_at(strain_np, dpos=None):
        eps = torch.tensor(strain_np, dtype=DTYPE, device=DEVICE)
        p = pos if dpos is None else pos + dpos
        ps = p + p @ eps
        she_s = she + she @ eps
        shn_s = shn + shn @ eps
        with torch.no_grad():
            return model.forward_pbc(ps, types, edge_i, edge_j, she_s,
                                     nb_src, nb_dst, shn_s).item()

    # ── analytic stress + forces ──
    strain = torch.zeros(3, 3, dtype=DTYPE, device=DEVICE, requires_grad=True)
    posv = pos.clone().requires_grad_(True)
    ps = posv + posv @ strain
    she_s = she + she @ strain
    shn_s = shn + shn @ strain
    e = model.forward_pbc(ps, types, edge_i, edge_j, she_s, nb_src, nb_dst, shn_s)
    g_pos, g_strain = torch.autograd.grad(e, [posv, strain])
    stress_ana = (g_strain / volume).cpu().numpy()
    forces_ana = (-g_pos).cpu().numpy()

    # ── FD stress ──
    delta = 1e-5
    stress_fd = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ep = np.zeros((3, 3)); ep[i, j] = delta
            em = np.zeros((3, 3)); em[i, j] = -delta
            stress_fd[i, j] = (energy_at(ep) - energy_at(em)) / (2 * delta) / volume
    s_err = np.abs(stress_ana - stress_fd).max()
    s_scale = max(np.abs(stress_ana).max(), 1e-8)
    print(f"  stress max abs err = {s_err:.2e}  (scale {s_scale:.2e}, rel {s_err/s_scale:.2e})")
    assert s_err < 1e-4 * max(s_scale, 1.0) + 1e-7, f"stress FD mismatch: {s_err:.2e}"

    # ── FD forces (a couple of components) ──
    zero = np.zeros((3, 3))
    f_err = 0.0
    for a in range(min(3, s['n_atoms'])):
        for c in range(3):
            dpos_p = torch.zeros_like(pos); dpos_p[a, c] = delta
            dpos_m = torch.zeros_like(pos); dpos_m[a, c] = -delta
            f_fd = -(energy_at(zero, dpos_p) - energy_at(zero, dpos_m)) / (2 * delta)
            f_err = max(f_err, abs(f_fd - forces_ana[a, c]))
    print(f"  forces max abs err (sampled) = {f_err:.2e}")
    assert f_err < 1e-5, f"force FD mismatch: {f_err:.2e}"
    print("  stress + force conventions match the model.\n")


def test_wigner_pole_gradient():
    """Regression: build_D1_from_rhat must have finite gradients at exact axis
    poles (ry=rz=0 etc.), which crystalline PBC self-image edges hit, and must be
    forward-bit-identical to the prior clamp formulation (so SPICE is unaffected).
    """
    print("=== Regression: Wigner D^1 gradient at exact axis poles ===")
    from ecenet.spherical import build_D1_from_rhat

    poles = torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.],
                          [-1., 0, 0], [0, -1., 0], [0, 0, -1.]], dtype=torch.float64)
    for v in poles:
        r = v[None].clone().requires_grad_(True)
        D = build_D1_from_rhat(r)
        g = torch.autograd.grad(D.sum(), r)[0]
        assert torch.isfinite(D).all(), f"forward NaN at {v.tolist()}"
        assert torch.isfinite(g).all(), f"gradient NaN at {v.tolist()}"

    # forward bit-identical to the original clamp formula (off-pole + poles)
    def _old(r_hat):
        rx, ry, rz = r_hat[:, 0], r_hat[:, 1], r_hat[:, 2]; z = torch.zeros_like(rx)
        s_x = (ry*ry+rz*rz).sqrt().clamp(min=1e-10); ix = 1.0/s_x
        A = torch.stack([rz*ix, ry, -rx*ry*ix, -ry*ix, rz, -rx*rz*ix, z, rx, s_x], -1).reshape(-1, 3, 3)
        s_y = (rx*rx+rz*rz).sqrt().clamp(min=1e-10); iy = 1.0/s_y
        B = torch.stack([z, ry, s_y, rx*iy, rz, -ry*rz*iy, -rz*iy, rx, -rx*ry*iy], -1).reshape(-1, 3, 3)
        return torch.where((rx.abs() >= 0.9)[:, None, None].expand_as(A), B, A)

    torch.manual_seed(0)
    r = torch.randn(5000, 3, dtype=torch.float64); r = r / r.norm(dim=-1, keepdim=True)
    r = torch.cat([r, poles])
    diff = (build_D1_from_rhat(r) - _old(r)).abs().max().item()
    assert diff == 0.0, f"forward changed vs clamp formula: {diff:.2e}"
    print(f"  all axis poles finite; forward bit-identical to clamp (max diff {diff:.1e})\n")


def test_torch_neighbor_list_matches_ase():
    """The vectorized all-images torch neighbor list must reproduce ASE's
    neighbor_list('ijS') exactly — same (i, j, Cartesian shift) edge set,
    including multiple periodic images and self-image edges (i==j, S≠0)."""
    print("=== Neighbor list: torch all-images vs ASE ('ijS') ===")
    from collections import Counter

    from ase import Atoms
    from ase.neighborlist import neighbor_list
    from train_ecenet_mptrj import torch_neighbor_list

    def edge_set(i, j, shift):
        return Counter((int(a), int(b),
                        round(float(s0), 3), round(float(s1), 3), round(float(s2), 3))
                       for a, b, (s0, s1, s2) in zip(i.tolist(), j.tolist(), shift.tolist()))

    def compare(pos, cell, rc, label):
        atoms = Atoms(numbers=[1]*len(pos), positions=pos, cell=cell, pbc=True)
        ei, ej, Se = neighbor_list('ijS', atoms, rc)
        ase_e = edge_set(torch.tensor(ei), torch.tensor(ej),
                         torch.tensor(Se.astype(np.float64) @ cell))
        ti, tj, tsh = torch_neighbor_list(torch.tensor(pos, dtype=DTYPE),
                                          torch.tensor(cell, dtype=DTYPE), rc)
        torch_e = edge_set(ti, tj, tsh)
        assert torch_e == ase_e, (
            f"{label} rc={rc}: torch {sum(torch_e.values())} vs ASE {sum(ase_e.values())} "
            f"edges; only-torch={len(torch_e - ase_e)} only-ASE={len(ase_e - torch_e)}")
        n_self = sum(1 for k in torch_e if k[0] == k[1])
        return sum(torch_e.values()), n_self

    rng = np.random.RandomState(0)
    cells = {
        'cubic-3Å (r_cut>L/2)': np.diag([3.0, 3.0, 3.0]),
        'cubic-7Å':             np.diag([7.0, 7.5, 7.0]),
        'slab-4×4×12':          np.diag([4.0, 4.0, 12.0]),
        'triclinic':            np.array([[5.0, 0, 0], [1.2, 5.3, 0], [0.6, 0.9, 6.1]]),
    }
    for label, cell in cells.items():
        na = rng.randint(4, 9)
        pos = rng.uniform(0, 1, (na, 3)) @ cell
        for rc in (4.0, 5.0):
            n_edges, n_self = compare(pos, cell, rc, label)
        print(f"  {label:24s}: matches ASE (rc=4,5); {n_edges} edges, {n_self} self-image")

    import os
    if os.path.exists('MPtrj_2022.9_full.json'):
        from train_ecenet_mptrj import load_mptrj
        for k, s in enumerate(load_mptrj('MPtrj_2022.9_full.json', max_structures=5, verbose=False)):
            for rc in (4.0, 5.0):
                compare(s['positions'], s['cell'], rc, f'MPtrj#{k}')
        print("  also matches ASE exactly on 5 real MPtrj structures")
    print()


if __name__ == '__main__':
    test_torch_neighbor_list_matches_ase()
    test_wigner_pole_gradient()
    test_stress_and_force_fd()
    test_loss_decreases()
    test_smoke_train()
    print("ALL TESTS PASSED")
