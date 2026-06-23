"""Integration test: ECENet constructs, runs energy/forces, is SO(3)-invariant,
and the training path forward_batch_multi works (with and without message passing).

Run:  python test_ecenet.py
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch

from ecenet import ECENet

torch.manual_seed(0)
DTYPE = torch.float64
N_TYPES = 4

COMMON = dict(
    n_types=N_TYPES, r_cut_edge=5.0, r_cut_neighbor=4.0,
    l_max=2, n_max=3, embed_dim=8, n_layers=2, n_max_d=4,
)


def random_structure(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g, dtype=DTYPE) * 1.8
    types = torch.randint(0, N_TYPES, (n,), generator=g)
    return pos, types


def rand_rotation(seed=1):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g, dtype=DTYPE)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diag(R))           # proper-ish orthogonal
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _energy_and_forces(model, pos, types):
    p = pos.clone().requires_grad_(True)
    e = model(p, types)
    f = -torch.autograd.grad(e, p, create_graph=True)[0]
    return e, f


def test_constructs_and_runs():
    pos, types = random_structure()
    model = ECENet(**COMMON).double()
    e, f = _energy_and_forces(model, pos, types)
    assert e.dim() == 0 and torch.isfinite(e), "energy not a finite scalar"
    assert f.shape == pos.shape and torch.isfinite(f).all(), "bad forces"
    print(f"  ECENet runs: E={e.item():.4f}, |F|max={f.abs().max():.3f}")


def test_so3_invariance():
    pos, types = random_structure(seed=2)
    model = ECENet(**COMMON).double()
    Q = rand_rotation()
    e1 = model(pos, types)
    e2 = model(pos @ Q.T, types)
    err = (e1 - e2).abs().item()
    assert err < 1e-9, f"energy not SO(3)-invariant: {err:.2e}"
    print(f"  SO(3) invariance: |E(Rx) - E(x)| = {err:.1e}")


def test_forces_finite_difference():
    pos, types = random_structure(seed=3)
    model = ECENet(**COMMON).double()
    _, f = _energy_and_forces(model, pos, types)
    eps = 1e-5
    fd = torch.zeros_like(pos)
    for i in range(pos.shape[0]):
        for d in range(3):
            p = pos.clone(); p[i, d] += eps
            ep = model(p, types)
            p = pos.clone(); p[i, d] -= eps
            em = model(p, types)
            fd[i, d] = -(ep - em) / (2 * eps)
    err = (f - fd).abs().max().item()
    assert err < 1e-5, f"analytic vs FD forces mismatch {err:.2e}"
    print(f"  forces match finite-difference (max err {err:.1e})")


def test_training_path_forward_batch_multi():
    model = ECENet(**COMMON).double()
    structs = [random_structure(n=5 + b, seed=10 + b) for b in range(3)]
    pos_list = [s[0].clone().requires_grad_(True) for s in structs]
    typ_list = [s[1] for s in structs]
    energies = model.forward_batch_multi(pos_list, typ_list)
    assert energies.shape == (3,) and torch.isfinite(energies).all()
    grads = torch.autograd.grad(energies.sum(), pos_list, create_graph=True)
    assert all(torch.isfinite(g).all() for g in grads)
    print(f"  forward_batch_multi (training path): energies {energies.detach().numpy().round(3)}")


def test_ecenet_mp():
    """ECENet with message passing (n_mp=2): SO(3)-invariant through the MP
    unrotate/rotate (via D_block), energy/forces finite."""
    pos, types = random_structure(seed=6)
    model = ECENet(**COMMON, n_mp=2).double()
    e, f = _energy_and_forces(model, pos, types)
    assert torch.isfinite(e) and f.shape == pos.shape and torch.isfinite(f).all()
    Q = rand_rotation(seed=4)
    err = (model(pos, types) - model(pos @ Q.T, types)).abs().item()
    assert err < 1e-9, f"ECENet(n_mp=2) not SO(3)-invariant: {err:.2e}"
    print(f"  ECENet(n_mp=2) runs: E={e.item():.4f}, SO(3) err {err:.1e}")


if __name__ == "__main__":
    print("ECENet integration")
    test_constructs_and_runs()
    test_so3_invariance()
    test_forces_finite_difference()
    test_training_path_forward_batch_multi()
    test_ecenet_mp()
    print("All tests passed.")
