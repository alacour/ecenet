# Prototype, mainly implemented by Claude
"""Joint LES training (use_les) in train_ecenet (rMD17/MD22) and
train_ecenet_mptrj (periodic).

Requires the optional `les` package (skips cleanly when absent). Covers:
  1. rMD17 trainer: end-to-end use_les smoke on a synthetic npz ('sum' head
     and parameter-free 'edge_basis'+dipole+iso-α), checkpoint `les` key, and the
     use_les resume-mismatch guard;
  2. MPtrj trainer: end-to-end use_les smoke (periodic Ewald, stress on),
     resume continues, mismatch guard;
  3. force + stress finite differences through E_sr + E_lr with the trainer's
     exact strain composition (positions + shift vectors + CELL) — the first
     check of the Ewald term's explicit cell dependence in this trainer;
  4. prepared-shard mode: LES smoke on freshly prepared shards (which now
     store cells), and a clear error on cell-less shards from before.

Run:  python tests/test_trainer_les.py    (from the repo root)
"""

import os
import sys
import tempfile
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

try:
    import les  # noqa: F401
except ImportError:
    print("SKIP: optional `les` package not installed")
    sys.exit(0)

from test_mptrj_shard_batching import write_real_prepared
from test_mptrj_trainer import build_type_map, make_structures
from train_ecenet import train_ecenet
from train_ecenet_mptrj import (
    _PBCMultiForwardWrapper,
    build_topology,
    train_ecenet_mptrj,
)

from ecenet import ECENet
from ecenet.les import load_les_module

DTYPE = torch.float64
DEVICE = torch.device('cpu')

TINY = dict(l_max=2, n_max=2, embed_dim=8, n_layers=1, n_max_d=4,
            r_cut_edge=4.0, r_cut_neighbor=3.5)


def write_fake_rmd17(out_dir, n_frames=12, na=5, seed=0):
    rng = np.random.RandomState(seed)
    base = rng.uniform(-1.5, 1.5, (na, 3))
    np.savez(os.path.join(out_dir, 'rmd17_ethanol.npz'),
             coords=base[None] + 0.1 * rng.randn(n_frames, na, 3),
             forces=rng.randn(n_frames, na, 3),
             energies=rng.randn(n_frames) * 5,
             nuclear_charges=np.array([6, 6, 8, 1, 1]))


def test_rmd17_use_les():
    print("=== rMD17 trainer: use_les smoke, checkpoint key, mismatch guard ===")
    with tempfile.TemporaryDirectory() as tmp:
        write_fake_rmd17(tmp)
        ckpt = os.path.join(tmp, 'm.mdl')
        common = dict(molecule='ethanol', data_dir=tmp, n_train=8, n_val=2,
                      n_test=2, n_epochs=2, batch_size=4, eval_every=1,
                      dtype=DTYPE, device=DEVICE, seed=0, verbose=False, **TINY)
        for ro, dip, alp in (('sum', False, None), ('edge_basis', True, 'iso')):
            _, res = train_ecenet(use_les=True, les_readout=ro, les_dipole=dip,
                                  les_alpha=alp,
                                  checkpoint_path=ckpt if ro == 'sum' else None,
                                  **common)
            assert np.isfinite(res['val_force_mae']), ro
            assert res['les_module'] is not None
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        assert 'les' in ck and ck['les']['state_dict'] is not None
        # resume with use_les continues from the checkpoint (les_readout must
        # match the checkpoint's — the default now resolves to 'edge_basis')...
        train_ecenet(use_les=True, les_readout='sum', checkpoint_path=ckpt,
                     **{**common, 'n_epochs': 3})
        # ...and a short-range run against an LES checkpoint is refused.
        try:
            train_ecenet(use_les=False, checkpoint_path=ckpt, **common)
            raise AssertionError("use_les mismatch not caught")
        except ValueError as e:
            assert 'use_les' in str(e)
    print("  smoke (sum, edge_basis+dipole+α), 'les' key, resume + guard OK\n")


def test_mptrj_use_les():
    print("=== MPtrj trainer: use_les smoke (Ewald + stress), resume, guard ===")
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, 'm.mdl')
        common = dict(n_val=3, les_readout='edge_basis', les_dipole=True,
                      dtype=DTYPE, device=DEVICE, seed=0, verbose=False, **TINY)
        _, res = train_ecenet_mptrj(
            train_structures=make_structures(12, seed=1), use_les=True,
            stress_weight=0.1, n_epochs=2, batch_size=4,
            checkpoint_path=ckpt, **common)
        for k in ('val_energy_mae', 'val_force_mae', 'val_stress_mae'):
            assert np.isfinite(res[k]), (k, res[k])
        assert res['les_module'] is not None
        assert 'les' in torch.load(ckpt, map_location='cpu', weights_only=False)
        train_ecenet_mptrj(train_structures=make_structures(12, seed=1),
                           use_les=True, stress_weight=0.1, n_epochs=3,
                           batch_size=4, checkpoint_path=ckpt, **common)
        try:
            train_ecenet_mptrj(train_structures=make_structures(12, seed=1),
                               use_les=False, n_epochs=1,
                               checkpoint_path=ckpt, **common)
            raise AssertionError("use_les mismatch not caught")
        except ValueError as e:
            assert 'use_les' in str(e)
    print("  smoke, resume + guard OK\n")


def test_mptrj_les_force_stress_fd():
    """FD check of forces AND stress through E_sr + E_lr with the trainer's
    exact input composition: strained positions, shift vectors, and cell, all
    through _PBCMultiForwardWrapper's batched periodic LES call."""
    print("=== FD: forces + stress through E_sr + E_lr (Ewald cell strain) ===")
    s = make_structures(1, seed=7, n_atoms_range=(6, 7))[0]
    type_map = build_type_map([s])

    torch.manual_seed(0)
    model = ECENet(n_types=len(type_map), les_readout='edge_basis',
                   les_dipole=True, **TINY).double().to(DEVICE)
    for p in model.parameters():
        with torch.no_grad():
            p.add_(0.05 * torch.randn_like(p))
    les_mod = load_les_module({'arguments': None}, model, DEVICE, DTYPE,
                              load_state=False)
    fwd = _PBCMultiForwardWrapper(model, les_mod)
    fwd_sr = _PBCMultiForwardWrapper(model, None)

    ei, ej, she, ni, nj, shn = build_topology(
        s['positions'], s['cell'], True, TINY['r_cut_edge'],
        TINY['r_cut_neighbor'], DEVICE, DTYPE)
    pos = torch.tensor(s['positions'], dtype=DTYPE, device=DEVICE)
    types = torch.tensor([type_map[int(z)] for z in s['numbers']],
                         dtype=torch.long, device=DEVICE)
    cell = torch.tensor(s['cell'], dtype=DTYPE, device=DEVICE)
    volume = abs(np.linalg.det(s['cell']))

    def energy_at(strain_np, dpos=None, f=fwd):
        eps = torch.tensor(strain_np, dtype=DTYPE, device=DEVICE)
        p = pos if dpos is None else pos + dpos
        with torch.no_grad():
            return f([p + p @ eps], [types], [ei], [ej], [she + she @ eps],
                     [ni], [nj], [shn + shn @ eps], [cell + cell @ eps]).item()

    zero = np.zeros((3, 3))
    e_lr = energy_at(zero) - energy_at(zero, f=fwd_sr)
    assert abs(e_lr) > 1e-10, "E_lr is zero — the FD would not test LES at all"

    # analytic forces + stress
    strain = torch.zeros(3, 3, dtype=DTYPE, device=DEVICE, requires_grad=True)
    posv = pos.clone().requires_grad_(True)
    e = fwd([posv + posv @ strain], [types], [ei], [ej], [she + she @ strain],
            [ni], [nj], [shn + shn @ strain], [cell + cell @ strain])
    g_pos, g_strain = torch.autograd.grad(e.sum(), [posv, strain])
    forces_ana = (-g_pos).numpy()
    stress_ana = (g_strain / volume).numpy()

    delta = 1e-5
    stress_fd = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ep = np.zeros((3, 3)); ep[i, j] = delta
            em = np.zeros((3, 3)); em[i, j] = -delta
            stress_fd[i, j] = (energy_at(ep) - energy_at(em)) / (2 * delta) / volume
    s_err = np.abs(stress_ana - stress_fd).max()
    s_scale = max(np.abs(stress_ana).max(), 1e-8)
    print(f"  E_lr = {e_lr:.3e} eV | stress max err {s_err:.2e} "
          f"(scale {s_scale:.2e})")
    assert s_err < 1e-4 * max(s_scale, 1.0) + 1e-7, f"stress FD: {s_err:.2e}"

    f_err = 0.0
    for a in range(3):
        for c in range(3):
            dp = torch.zeros_like(pos); dp[a, c] = delta
            dm = torch.zeros_like(pos); dm[a, c] = -delta
            f_fd = -(energy_at(zero, dp) - energy_at(zero, dm)) / (2 * delta)
            f_err = max(f_err, abs(f_fd - forces_ana[a, c]))
    print(f"  forces max err (sampled) = {f_err:.2e}")
    assert f_err < 1e-5, f"force FD: {f_err:.2e}"
    print("  E_lr nonzero; forces + stress match FD through the Ewald term\n")


def test_mptrj_les_prepared():
    print("=== prepared mode: LES on fresh shards; cell-less shards refused ===")
    with tempfile.TemporaryDirectory() as tmp:
        structs = make_structures(48, seed=5)
        write_real_prepared(tmp, structs, shard_size=8)
        _, res = train_ecenet_mptrj(
            prepared_dir=tmp, val_frac=0.2, use_les=True,
            les_readout='edge_basis', stress_weight=0.1,
            n_epochs=1, batch_size=4, dtype=DTYPE, device=DEVICE,
            seed=0, verbose=False, **TINY)
        assert np.isfinite(res['val_force_mae'])
        # strip the cells (simulating shards prepared before they were stored)
        for p in sorted(Path(tmp).glob('shard_*.pt')):
            shard = torch.load(p, map_location='cpu', weights_only=False)
            for fr in shard:
                fr.pop('cell', None)
            torch.save(shard, p)
        try:
            train_ecenet_mptrj(prepared_dir=tmp, val_frac=0.2, use_les=True,
                               n_epochs=1, batch_size=4, dtype=DTYPE,
                               device=DEVICE, seed=0, verbose=False, **TINY)
            raise AssertionError("cell-less shards not refused under use_les")
        except ValueError as e:
            assert 'cell' in str(e)
    print("  fresh shards train; cell-less shards raise a clear error\n")


if __name__ == '__main__':
    test_rmd17_use_les()
    test_mptrj_use_les()
    test_mptrj_les_force_stress_fd()
    test_mptrj_les_prepared()
    print("ALL TESTS PASSED")
