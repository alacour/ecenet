# Prototype, mainly implemented by Claude
"""
Tests for scripts/train_ecenet_xyz.py — the small-dataset trainer with
optional joint LES long-range training.

No data download needed: synthetic random periodic structures throughout.
Covers:
  1. end-to-end smoke, LES off (energy + force + stress);
  2. end-to-end smoke, LES on — including checkpoint save → resume (the LES
     head is built lazily by upstream, so resume exercises the
     materialise-then-load path) and best-state restore;
  3. finite-difference check of forces THROUGH the LES term (E = E_sr + E_lr
     on one graph) on a fresh model;
  4. ECENetCalculator.from_checkpoint refuses an LES checkpoint unless
     ignore_les=True.

LES-dependent tests skip cleanly when the optional `les` package is missing.

Run:  python tests/test_xyz_trainer.py     (from the repo root)
"""

import os
import sys  # repo root + scripts/ on path (imports ecenet and the scripts/ trainer)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'scripts'))

import tempfile

import numpy as np
import torch
from train_ecenet_mptrj import compute_energy_reference
from train_ecenet_xyz import tensorize, train_ecenet_xyz

from ecenet import elements

DTYPE = torch.float64
DEVICE = torch.device('cpu')   # FD needs float64; MPS has no float64
Z_CHOICES = [1, 8, 11, 17]     # H, O, Na, Cl → n_types = 4


def _has_les():
    try:
        import les  # noqa: F401
        return True
    except ImportError:
        return False


def make_structures(n, seed=0, n_atoms_range=(4, 8), box=(7.0, 8.0)):
    """Random periodic structures with random energy/forces/stress (eV/Å³)."""
    rng = np.random.RandomState(seed)
    structs = []
    for _ in range(n):
        na = rng.randint(*n_atoms_range)
        L = rng.uniform(*box)
        cell = np.diag([L, L, L]).astype(np.float64)
        cell[0, 1] = rng.uniform(-0.5, 0.5)   # exercise triclinic shifts
        cell[1, 2] = rng.uniform(-0.5, 0.5)
        frac = rng.uniform(0, 1, size=(na, 3))
        structs.append({
            'numbers': rng.choice(Z_CHOICES, size=na).astype(np.int64),
            'positions': frac @ cell,
            'cell': cell,
            'pbc': True,
            'energy': float(rng.uniform(-5, 5) * na),
            'forces': rng.uniform(-1, 1, size=(na, 3)).astype(np.float64),
            'stress': rng.uniform(-0.05, 0.05, size=(3, 3)).astype(np.float64),
            'n_atoms': na,
        })
    return structs


COMMON = dict(
    l_max=2, n_max=2, embed_dim=8, n_layers=1, n_max_d=4,
    r_cut_edge=4.0, r_cut_neighbor=3.5,
    dtype=DTYPE, device=DEVICE, seed=0, verbose=True,
)


def test_smoke_train():
    print("=== Smoke: end-to-end training, LES off (E + F + S) ===")
    _, les_module, results = train_ecenet_xyz(
        train_structures=make_structures(12, seed=1),
        test_structures=make_structures(3, seed=2),
        n_val=2, stress_weight=0.1,
        n_epochs=3, batch_size=4, lr=5e-3, **COMMON,
    )
    assert les_module is None
    for k in ('test_energy_mae', 'test_force_mae', 'test_stress_mae'):
        assert np.isfinite(results[k]), f"{k} not finite: {results[k]}"
    print(f"  results OK: E={results['test_energy_mae']:.3f} "
          f"F={results['test_force_mae']:.3f} S={results['test_stress_mae']:.3e}\n")


def test_smoke_train_les():
    if not _has_les():
        print("=== SKIP: LES smoke (optional `les` package not installed) ===\n")
        return
    print("=== Smoke: end-to-end training, LES on, + checkpoint resume ===")
    structs = make_structures(12, seed=4)
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, 'xyz_les.mdl')
        _, les_module, results = train_ecenet_xyz(
            train_structures=[dict(s) for s in structs], n_val=2,
            use_les=True, les_readout='sum', checkpoint_path=ckpt,
            n_epochs=2, batch_size=4, lr=5e-3, **COMMON,
        )
        assert les_module is not None
        assert any(p.requires_grad for p in les_module.parameters()), \
            "LES head has no trainable parameters (lazy build did not run)"
        for k in ('val_energy_mae', 'val_force_mae'):
            assert np.isfinite(results[k]), f"{k} not finite: {results[k]}"

        # Resume: fresh call restores model + LES + optimizer and continues.
        _, les2, results2 = train_ecenet_xyz(
            train_structures=[dict(s) for s in structs], n_val=2,
            use_les=True, les_readout='sum', checkpoint_path=ckpt,
            n_epochs=4, batch_size=4, lr=5e-3, **COMMON,
        )
        assert np.isfinite(results2['val_force_mae'])

        # use_les must match the checkpoint.
        try:
            train_ecenet_xyz(
                train_structures=[dict(s) for s in structs], n_val=2,
                use_les=False, checkpoint_path=ckpt,
                n_epochs=5, batch_size=4, lr=5e-3, **COMMON,
            )
            raise AssertionError("resume with use_les=False should have raised")
        except ValueError as e:
            assert 'use_les' in str(e)
    print("  LES smoke + resume OK\n")


def test_les_force_fd():
    """Forces from the joint graph (E_sr + E_lr) match finite differences —
    with the default 'sum' read-out (upstream atomwise charge head) and with
    'edge_basis' (charge from the model's own MLP head, upstream bypassed)."""
    if not _has_les():
        print("=== SKIP: LES force FD (optional `les` package not installed) ===\n")
        return
    print("=== FD: forces through E_sr + E_lr ===")
    from train_ecenet_mptrj import build_topology

    from ecenet import ECENet
    from ecenet.les import LESLongRange

    structs = make_structures(1, seed=7, n_atoms_range=(5, 6))
    s = structs[0]
    type_map = elements.build_type_map(int(z) for z in s['numbers'])
    types = torch.tensor([type_map[int(z)] for z in s['numbers']],
                         dtype=torch.long, device=DEVICE)
    cell_t = torch.tensor(s['cell'], dtype=DTYPE, device=DEVICE)

    for readout, dip in (('sum', False), ('edge_basis', False),
                         ('edge_basis', True)):
        torch.manual_seed(3)
        model = ECENet(n_types=len(type_map), r_cut_edge=4.0,
                       r_cut_neighbor=3.5, l_max=2, n_max=2, embed_dim=8,
                       n_layers=1, n_max_d=4, les_readout=readout,
                       les_dipole=dip).double().to(DEVICE)
        lr_mod = LESLongRange().double()
        is_charge = readout in ('edge', 'edge_basis')
        if dip:
            # dipole slot is zero-init; perturb it so the FD exercises the
            # charge–dipole and dipole–dipole terms too
            with torch.no_grad():
                model.les_edge_charge.linears[-1].weight[model.n_max_d:
                                                         ].normal_(std=0.5)

        def total_energy(pos_np, requires_grad=False):
            # topology rebuilt per evaluation so FD displacements stay consistent
            ei, ej, she, ni, nj, shn = build_topology(
                pos_np, s['cell'], True, 4.0, 3.5, DEVICE, DTYPE)
            pos = torch.tensor(pos_np, dtype=DTYPE, device=DEVICE,
                               requires_grad=requires_grad)
            e_sr, l0 = model.forward_pbc(pos, types, ei, ej, she, ni, nj, shn,
                                         return_embeddings=True, l0_only=True)
            e = e_sr + lr_mod(l0, pos, cell=cell_t, l0_is_charge=is_charge,
                              les_dipole=dip).sum()
            return e, pos

        if not is_charge:
            # materialise the lazy LES head, then perturb it away from
            # zero-ish init ('edge_basis' bypasses the head: parameter-free)
            with torch.no_grad():
                total_energy(s['positions'])
            for p in lr_mod.parameters():
                with torch.no_grad():
                    p.add_(0.1 * torch.randn_like(p))

        e, pos = total_energy(s['positions'], requires_grad=True)
        forces = -torch.autograd.grad(e, pos)[0].cpu().numpy()

        h = 1e-5
        max_err = 0.0
        for a in range(min(3, s['n_atoms'])):
            for c in range(3):
                pp = s['positions'].copy(); pp[a, c] += h
                pm = s['positions'].copy(); pm[a, c] -= h
                with torch.no_grad():
                    ep, _ = total_energy(pp)
                    em, _ = total_energy(pm)
                f_fd = -(ep.item() - em.item()) / (2 * h)
                max_err = max(max_err, abs(f_fd - forces[a, c]))
        label = readout + ('+dipole' if dip else '')
        assert max_err < 1e-6, \
            f"FD force mismatch through LES ({label}): {max_err:.2e}"
        print(f"  {label}: max |F_autograd - F_fd| = {max_err:.2e}  (incl. E_lr)")
    print()


def test_calculator_rejects_les_checkpoint():
    if not _has_les():
        print("=== SKIP: calculator LES rejection (optional `les` package "
              "not installed) ===\n")
        return
    print("=== Calculator: from_checkpoint refuses LES checkpoints ===")
    from ecenet.calculator import ECENetCalculator

    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, 'xyz_les.mdl')
        train_ecenet_xyz(
            train_structures=make_structures(8, seed=9), n_val=2,
            use_les=True, les_readout='sum', checkpoint_path=ckpt,
            n_epochs=1, batch_size=4, lr=5e-3, **COMMON,
        )
        try:
            ECENetCalculator.from_checkpoint(ckpt, device='cpu')
            raise AssertionError("from_checkpoint should refuse an LES checkpoint")
        except ValueError as e:
            assert 'LES' in str(e), f"unexpected error: {e}"
        calc = ECENetCalculator.from_checkpoint(ckpt, device='cpu', ignore_les=True)
        assert calc is not None
    print("  rejected without ignore_les, loads (SR-only) with it\n")


def test_les_calculator():
    """ECENetLESCalculator: energy/forces equal the manual joint graph
    (E_sr + E_lr + Σe_ref), analytic stress matches FD through the strained
    cell (the Ewald term's cell dependence included), pbc=False takes the
    isolated path, and a short-range checkpoint is refused (symmetric with
    ECENetCalculator refusing LES ones)."""
    if not _has_les():
        print("=== SKIP: LES calculator (optional `les` package not installed) ===\n")
        return
    print("=== ECENetLESCalculator: E_sr + E_lr, forces, stress FD ===")
    from ase import Atoms

    from ecenet.calculator import ECENetLESCalculator

    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, 'les_calc.mdl')
        train_ecenet_xyz(
            train_structures=make_structures(8, seed=11), n_val=2,
            use_les=True, les_readout='edge_basis', les_dipole=True,
            checkpoint_path=ckpt, n_epochs=2, batch_size=4, lr=5e-3, **COMMON)
        calc = ECENetLESCalculator.from_checkpoint(ckpt, device='cpu')

        s = make_structures(1, seed=12, box=(8.5, 9.0))[0]   # MIC-safe box
        atoms = Atoms(numbers=s['numbers'], positions=s['positions'],
                      cell=s['cell'], pbc=True)
        atoms.calc = calc
        e_calc = atoms.get_potential_energy()
        f_calc = atoms.get_forces()

        # manual joint graph, same weights + same (MIC) topology
        types = torch.tensor(
            [calc.element_to_type[sym] for sym in atoms.get_chemical_symbols()],
            dtype=torch.long)
        pos = torch.tensor(s['positions'], dtype=DTYPE).requires_grad_(True)
        ei, ej, she = calc._gpu_neighbor_list(pos.detach(), s['cell'],
                                              calc.model.r_cut_edge)
        ni, nj, shn = calc._gpu_neighbor_list(pos.detach(), s['cell'],
                                              calc.model.r_cut_neighbor)
        e_sr, l0 = calc.model.forward_pbc(pos, types, ei, ej, she, ni, nj, shn,
                                          return_embeddings=True, l0_only=True)
        cell_t = torch.tensor(s['cell'], dtype=DTYPE)
        e_man = e_sr + calc.les_module(l0, pos, cell=cell_t, l0_is_charge=True,
                                       les_dipole=True).sum()
        f_man = -torch.autograd.grad(e_man, pos)[0].numpy()
        e_ref_sum = sum(calc.energy_reference[sym]
                        for sym in atoms.get_chemical_symbols())
        de = abs(e_calc - (e_man.item() + e_ref_sum))
        df = np.abs(f_calc - f_man).max()
        assert de < 1e-10, f"calculator energy != manual joint graph: {de:.3e}"
        assert df < 1e-10, f"calculator forces != manual joint graph: {df:.3e}"
        # E_lr is actually in the number (dipoles were perturbed by training)
        e_lr = calc.les_module(l0.detach(), pos.detach(), cell=cell_t,
                               l0_is_charge=True, les_dipole=True).sum()
        assert abs(float(e_lr)) > 0, "E_lr is identically zero in the test"

        # latent charges + dipoles are stashed on every force call: q is the
        # packed l0's first column (edge mode), u the remaining three
        l0_np = l0.detach().numpy()
        q_calc = atoms.get_charges()
        dq = np.abs(q_calc - l0_np[:, 0]).max()
        du = np.abs(calc.results['les_dipoles'] - l0_np[:, 1:4]).max()
        assert dq < 1e-12, f"exposed charges != l0[:, 0]: {dq:.3e}"
        assert du < 1e-12, f"exposed dipoles != l0[:, 1:4]: {du:.3e}"
        assert abs(float(np.abs(q_calc).max())) > 0, \
            "latent charges identically zero in the test"

        # analytic stress vs FD: deform positions AND cell (x → x·(1+ε)),
        # which is exactly what the strain pass differentiates
        stress_v = atoms.get_stress()   # Voigt [xx, yy, zz, yz, xz, xy]
        # the strain pass evaluates at ε = 0 (unstrained geometry), so the
        # stress call leaves the stashed charges unchanged
        dq_strain = np.abs(calc.results['charges'] - q_calc).max()
        assert dq_strain < 1e-12, \
            f"stress call perturbed the stashed charges: {dq_strain:.3e}"
        V = abs(np.linalg.det(s['cell']))
        eps = 1e-6
        max_err = 0.0
        for (a, b), vi in [((0, 0), 0), ((2, 2), 2), ((0, 1), 5)]:
            E = np.zeros((3, 3)); E[a, b] = eps
            es = []
            for sign in (+1, -1):
                F = np.eye(3) + sign * E
                at = Atoms(numbers=s['numbers'],
                           positions=s['positions'] @ F,
                           cell=s['cell'] @ F, pbc=True)
                at.calc = calc
                es.append(at.get_potential_energy())
            fd = (es[0] - es[1]) / (2 * eps) / V
            max_err = max(max_err, abs(fd - stress_v[vi]))
        assert max_err < 1e-7, f"stress FD mismatch (incl. Ewald): {max_err:.2e}"

        # Born effective charges on the PERIODIC box — the Berry-phase path
        # eval_spice_bec never exercised (it ran cell=None only).
        na = len(atoms)
        Z = calc.compute_bec(atoms)
        assert Z.shape == (na, 3, 3) and np.isfinite(Z).all()
        bec_mod = calc.les_module.les.bec
        cell64 = torch.tensor(s['cell'], dtype=DTYPE)

        # exact identity: with q fixed (no charge flow) the dephasing +
        # projection must give Z* = (q − q̄) ⊗ I to machine precision
        q_fix = torch.tensor(np.linspace(-0.3, 0.4, na), dtype=DTYPE)
        pos_g = torch.tensor(s['positions'], dtype=DTYPE, requires_grad=True)
        Z_fix = bec_mod(q=q_fix, r=pos_g, cell=cell64.view(1, 3, 3))
        q_eff = (q_fix - q_fix.mean() if bec_mod.remove_mean else q_fix)
        expect = (q_eff.numpy()[:, None, None] * np.eye(3)
                  * bec_mod.normalization_factor)
        d_id = np.abs(Z_fix.detach().numpy() - expect).max()
        assert d_id < 1e-10, f"periodic static-charge identity: {d_id:.3e}"

        # FD INCLUDING charge flow: finite-difference the complex
        # polarization (model q and u recomputed at each displaced geometry,
        # frozen topology), dephase with the reference phases, project —
        # exactly what upstream's analytic grad computes
        ei0, ej0, she0, ni0, nj0, shn0 = calc._neighbor_lists(
            torch.tensor(s['positions'], dtype=DTYPE), s['cell'])

        def pol(pos_np):
            p = torch.tensor(pos_np, dtype=DTYPE)
            with torch.no_grad():
                _, l0p = calc.model.forward_pbc(
                    p, types, ei0, ej0, she0, ni0, nj0, shn0,
                    return_embeddings=True, l0_only=True)
            qp = l0p[:, 0:1]
            if bec_mod.remove_mean:
                qp = qp - qp.mean(dim=0, keepdim=True)
            P, phase, proj = bec_mod.compute_pol_pbc(p, qp, cell64)
            P = P * bec_mod.normalization_factor
            P_u = l0p[:, 1:4].sum(0) * bec_mod.normalization_factor
            return P.numpy(), P_u.numpy(), phase.numpy(), proj.numpy()

        _, _, phase0, proj0 = pol(s['positions'])
        eps_fd = 1e-5
        Z_fd = np.zeros((na, 3, 3))
        for i in range(na):
            for b in range(3):
                dp = s['positions'].copy(); dm = s['positions'].copy()
                dp[i, b] += eps_fd; dm[i, b] -= eps_fd
                Pp, Pup, _, _ = pol(dp)
                Pm, Pum, _, _ = pol(dm)
                dP  = (Pp - Pm) / (2 * eps_fd)     # complex (3,)
                dPu = (Pup - Pum) / (2 * eps_fd)   # real (3,)
                zb = (dP * np.conj(phase0[i])).real
                Z_fd[i, :, b] = proj0 @ zb + dPu
        d_bec = np.abs(Z_fd - Z).max()
        assert d_bec < 1e-6, f"periodic BEC vs FD (incl. charge flow): {d_bec:.3e}"
        # charge flow is actually in the number (Z* is not diagonal)
        off = np.abs(Z - np.trace(Z, axis1=1, axis2=2)[:, None, None]
                     / 3 * np.eye(3)).max()
        assert off > 0, "BEC has no off-diagonal/flow structure in the test"

        # pbc=False → isolated path
        atoms_free = Atoms(numbers=s['numbers'], positions=s['positions'])
        atoms_free.calc = calc
        e_free = atoms_free.get_potential_energy()
        pos_f = torch.tensor(s['positions'], dtype=DTYPE)
        with torch.no_grad():
            e_sr_f, l0_f = calc.model(pos_f, types, return_embeddings=True,
                                      l0_only=True)
            e_man_f = e_sr_f + calc.les_module(l0_f, pos_f, l0_is_charge=True,
                                               les_dipole=True).sum()
        dfree = abs(e_free - (float(e_man_f) + e_ref_sum))
        assert dfree < 1e-10, f"isolated path mismatch: {dfree:.3e}"
        Z_free = calc.compute_bec(atoms_free)
        assert Z_free.shape == (na, 3, 3) and np.isfinite(Z_free).all()

        # a short-range checkpoint is refused
        ckpt_sr = os.path.join(td, 'sr.mdl')
        train_ecenet_xyz(
            train_structures=make_structures(6, seed=13), n_val=2,
            use_les=False, checkpoint_path=ckpt_sr,
            n_epochs=1, batch_size=4, lr=5e-3, **COMMON)
        try:
            ECENetLESCalculator.from_checkpoint(ckpt_sr, device='cpu')
            raise AssertionError("LES calculator should refuse an SR checkpoint")
        except ValueError as e:
            assert 'les' in str(e).lower()
    print(f"  energy/forces == manual joint graph (dE={de:.1e}, dF={df:.1e}); "
          f"charges/dipoles exposed (dq={dq:.1e}, du={du:.1e}); "
          f"stress FD {max_err:.1e}; periodic BEC identity {d_id:.1e}, "
          f"FD {d_bec:.1e}; isolated path {dfree:.1e}; SR refused\n")


def test_les_calculator_alpha():
    """les_alpha end to end: an xyz-trained 'aniso' checkpoint loads through
    load_calculator; α moved off its zero init; energy/forces equal the
    manual joint graph with the induced-dipole term; stress FD holds through
    the periodic α path; results['les_alphas'] is the unpacked l0 slot;
    compute_polarizability is the symmetric sum Σα_i (isolated and periodic)
    and equals dP/dE_ext via upstream's e_ext; compute_bec is finite and
    differs from the α-less Z*; an SR/α-less checkpoint refuses it."""
    if not _has_les():
        print("=== SKIP: LES α calculator (optional `les` package not installed) ===\n")
        return
    print("=== ECENetLESCalculator + les_alpha: forces, stress FD, α, BEC ===")
    from ase import Atoms

    from ecenet.calculator import load_calculator
    from ecenet.les import total_polarizability, unpack_l0

    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, 'les_alpha.mdl')
        train_ecenet_xyz(
            train_structures=make_structures(8, seed=21), n_val=2,
            use_les=True, les_readout='edge_basis', les_dipole=True,
            les_alpha='aniso', checkpoint_path=ckpt, n_epochs=2,
            batch_size=4, lr=5e-3, **COMMON)
        calc = load_calculator(ckpt, device='cpu', verbose=False)
        assert calc.les_flags['les_alpha'] == 'aniso'
        flags = calc.les_flags

        s = make_structures(1, seed=22, box=(8.5, 9.0))[0]
        atoms = Atoms(numbers=s['numbers'], positions=s['positions'],
                      cell=s['cell'], pbc=True)
        atoms.calc = calc
        e_calc = atoms.get_potential_energy()
        f_calc = atoms.get_forces()
        na = len(atoms)

        # manual joint graph with the same weights/topology
        types = torch.tensor(
            [calc.element_to_type[sym] for sym in atoms.get_chemical_symbols()],
            dtype=torch.long)
        pos = torch.tensor(s['positions'], dtype=DTYPE).requires_grad_(True)
        ei, ej, she = calc._gpu_neighbor_list(pos.detach(), s['cell'],
                                              calc.model.r_cut_edge)
        ni, nj, shn = calc._gpu_neighbor_list(pos.detach(), s['cell'],
                                              calc.model.r_cut_neighbor)
        e_sr, l0 = calc.model.forward_pbc(pos, types, ei, ej, she, ni, nj, shn,
                                          return_embeddings=True, l0_only=True)
        cell_t = torch.tensor(s['cell'], dtype=DTYPE)
        e_man = e_sr + calc.les_module(l0, pos, cell=cell_t, **flags).sum()
        f_man = -torch.autograd.grad(e_man, pos)[0].numpy()
        e_ref_sum = sum(calc.energy_reference[sym]
                        for sym in atoms.get_chemical_symbols())
        de = abs(e_calc - (e_man.item() + e_ref_sum))
        df = np.abs(f_calc - f_man).max()
        assert de < 1e-10 and df < 1e-10, f"joint graph: dE={de:.1e} dF={df:.1e}"

        # α trained off its zero init, and it changes E_lr
        q_l, u_l, a_l = unpack_l0(l0.detach(), **flags)
        assert a_l.shape == (na, 3, 3) and a_l.abs().max() > 0, "α stayed 0"
        l0_noa = l0.detach()[:, :4]
        e_lr_a = calc.les_module(l0.detach(), pos.detach(), cell=cell_t,
                                 **flags).sum()
        e_lr_0 = calc.les_module(l0_noa, pos.detach(), cell=cell_t,
                                 l0_is_charge=True, les_dipole=True).sum()
        assert abs(float(e_lr_a - e_lr_0)) > 0, "α term contributes nothing"

        # stash: α slot exposed on every force call
        da = np.abs(calc.results['les_alphas'] - a_l.numpy()).max()
        du = np.abs(calc.results['les_dipoles'] - u_l.numpy()).max()
        assert da < 1e-12 and du < 1e-12, f"stash: dα={da:.1e} du={du:.1e}"

        # stress FD through the periodic α path (one normal, one shear)
        stress_v = atoms.get_stress()
        V = abs(np.linalg.det(s['cell']))
        eps = 1e-6
        max_err = 0.0
        for (a, b), vi in [((1, 1), 1), ((0, 2), 4)]:
            E = np.zeros((3, 3)); E[a, b] = eps
            es = []
            for sign in (+1, -1):
                F = np.eye(3) + sign * E
                at = Atoms(numbers=s['numbers'], positions=s['positions'] @ F,
                           cell=s['cell'] @ F, pbc=True)
                at.calc = calc
                es.append(at.get_potential_energy())
            fd = (es[0] - es[1]) / (2 * eps) / V
            max_err = max(max_err, abs(fd - stress_v[vi]))
        assert max_err < 1e-7, f"stress FD with α: {max_err:.2e}"

        # polarizability: Σα_i, symmetric, same on the isolated path, and
        # equal to dP/dE_ext through upstream (the induced dipoles' response
        # to a uniform external field IS Σα for this non-self-consistent model)
        P = calc.compute_polarizability(atoms)
        assert P.shape == (3, 3)
        dP = np.abs(P - total_polarizability(a_l)[0].numpy()).max()
        assert dP < 1e-12 and np.abs(P - P.T).max() < 1e-12
        atoms_free = Atoms(numbers=s['numbers'], positions=s['positions'])
        atoms_free.calc = calc
        P_free = calc.compute_polarizability(atoms_free)
        pos_f = torch.tensor(s['positions'], dtype=DTYPE)
        with torch.no_grad():
            _, l0_f = calc.model(pos_f, types, return_embeddings=True,
                                 l0_only=True)
        _, _, a_f = unpack_l0(l0_f, **flags)
        dPf = np.abs(P_free - a_f.sum(0).numpy()).max()
        assert dPf < 1e-12, f"isolated polarizability: {dPf:.1e}"
        e_ext = torch.zeros(3, dtype=DTYPE, requires_grad=True)
        q_f, u_f, _ = unpack_l0(l0_f, **flags)
        res = calc.les_module.les(latent_charges=q_f, latent_dipoles=u_f,
                                  latent_alphas=a_f, positions=pos_f,
                                  cell=None, e_ext=e_ext, compute_energy=True)
        P_tot = res['latent_dipoles'].reshape(-1, 3).sum(0)     # Σ(u + Δu)
        dPdE = torch.stack([torch.autograd.grad(P_tot[c], e_ext,
                                                retain_graph=True)[0]
                            for c in range(3)]).numpy()
        dE = np.abs(dPdE - P_free).max()
        assert dE < 1e-10, f"Σα != dP/dE_ext: {dE:.1e}"

        # BEC with the induced dipoles: finite, periodic and isolated, and
        # not the α-less tensor
        Z = calc.compute_bec(atoms)
        Z_free = calc.compute_bec(atoms_free)
        assert Z.shape == (na, 3, 3) and np.isfinite(Z).all()
        assert Z_free.shape == (na, 3, 3) and np.isfinite(Z_free).all()
        pos_g = torch.tensor(s['positions'], dtype=DTYPE, requires_grad=True)
        _, l0_g = calc.model(pos_g, types, return_embeddings=True, l0_only=True)
        Z_noa = calc.les_module.born_charges(l0_g[:, :4], pos_g, cell=None,
                                             l0_is_charge=True, les_dipole=True)
        dz = np.abs(Z_free - Z_noa.detach().numpy()).max()
        assert dz > 1e-9, "induced dipoles missing from Z*"

        # an α-less checkpoint refuses compute_polarizability
        ckpt0 = os.path.join(td, 'les_noalpha.mdl')
        train_ecenet_xyz(
            train_structures=make_structures(6, seed=23), n_val=2,
            use_les=True, les_readout='edge_basis', checkpoint_path=ckpt0,
            n_epochs=1, batch_size=4, lr=5e-3, **COMMON)
        calc0 = load_calculator(ckpt0, device='cpu', verbose=False)
        try:
            calc0.compute_polarizability(atoms_free)
            raise AssertionError("α-less checkpoint should refuse")
        except ValueError as e:
            assert 'les_alpha' in str(e)
    print(f"  joint graph (dE={de:.1e}, dF={df:.1e}); stress FD {max_err:.1e}; "
          f"α stashed ({da:.1e}); Σα == dP/dE_ext ({dE:.1e}); BEC finite, "
          f"induced part {dz:.1e}; α-less refused\n")


def test_tensorize_keeps_cell():
    print("=== tensorize: cell kept for periodic, None otherwise ===")
    structs = make_structures(2, seed=5)
    structs[1]['cell'] = None
    structs[1]['pbc'] = False
    type_map = elements.build_type_map(
        int(z) for s in structs for z in s['numbers'])
    e_ref = compute_energy_reference(structs, type_map)
    data = tensorize(structs, type_map, e_ref, 4.0, 3.5, 1.0, DTYPE, DEVICE)
    assert data[0]['cell'] is not None and data[0]['cell'].shape == (3, 3)
    assert data[1]['cell'] is None
    print("  OK\n")


if __name__ == '__main__':
    test_tensorize_keeps_cell()
    test_smoke_train()
    test_les_force_fd()
    test_smoke_train_les()
    test_calculator_rejects_les_checkpoint()
    test_les_calculator()
    test_les_calculator_alpha()
    print("ALL TESTS PASSED")
