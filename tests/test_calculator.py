"""Behavioural tests for ECENetCalculator.

These pin the *observable* contract of the calculator and of `from_checkpoint`.
ecenet/calculator.py is dataset-agnostic: it reads only generic, self-describing
checkpoint keys (element mapping, reference energies, units) — no knowledge of
rMD17 / MD22 / SPICE / MPtrj.

What is locked down:
  * unit handling (kcal/mol → eV) via `energy_units`
  * per-element reference energies are added back atom-by-atom
  * the training mean energy is added back (in eV)
  * the element→type mapping reaches `calculate`
  * unsupported elements raise
  * from_checkpoint reconstructs the model + metadata from a saved dict, building
    energy_reference from an 'e_ref' array via the checkpoint's OWN mapping, and
    raising if no element mapping is present

Run:  python tests/test_calculator.py     (also collectable by pytest)
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile

import numpy as np
import torch
from ase import Atoms
from ase import units as ase_units

from ecenet import ECENet
from ecenet.calculator import ECENetCalculator

torch.manual_seed(0)

_KCAL = ase_units.kcal / ase_units.mol

# Small float64 model config — same idiom as test_ecenet.py.
_HPARAMS = dict(
    r_cut_edge=5.0, r_cut_neighbor=4.0,
    l_max=2, n_max=2, embed_dim=8, n_layers=1, n_max_d=4,
)


def _tiny_model(n_types):
    return ECENet(n_types=n_types, **_HPARAMS).double()


def _mol(symbols=('H', 'C', 'O')):
    """A small non-periodic molecule with the given elements."""
    pos = [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.2, 0.0]]
    return Atoms(symbols=list(symbols), positions=pos[:len(symbols)])


def _periodic_mol(symbols=('H', 'C', 'O'), a=12.0):
    """Same atoms in a cubic box big enough that r_cut <= L/2 (no MIC warning)."""
    atoms = _mol(symbols)
    atoms.set_cell([a, a, a])
    atoms.set_pbc(True)
    return atoms


def _energy(calc, atoms):
    a = atoms.copy()
    a.calc = calc
    return a.get_potential_energy()


def _save_ckpt(path, model, n_types, **extra):
    """Write a checkpoint dict the way the trainers do (hparams + state + extras)."""
    hp = dict(n_types=n_types, n_mp=1, **_HPARAMS)
    ckpt = {'model': model.state_dict(), 'hparams': hp}
    ckpt.update(extra)
    torch.save(ckpt, path)


# ── unit handling ────────────────────────────────────────────────────────────

def test_energy_units_kcal_vs_ev_scaling():
    """A kcal/mol calculator scales the same model output by kcal→eV vs eV."""
    model = _tiny_model(3)
    e2t = {'H': 0, 'C': 1, 'O': 2}
    atoms = _mol()

    calc_ev   = ECENetCalculator(model, element_to_type=e2t, energy_units='eV')
    calc_kcal = ECENetCalculator(model, element_to_type=e2t, energy_units='kcal/mol')

    assert abs(calc_ev._to_ev - 1.0) < 1e-15
    assert abs(calc_kcal._to_ev - _KCAL) < 1e-15

    e_ev   = _energy(calc_ev, atoms)
    e_kcal = _energy(calc_kcal, atoms)
    # Same raw model energy, just a different unit conversion factor.
    assert abs(e_kcal - e_ev * _KCAL) < 1e-9
    print(f"  units: E(eV)={e_ev:.4f}  E(kcal→eV)={e_kcal:.6f}  ratio={_KCAL:.5f}")


# ── reference energies ───────────────────────────────────────────────────────

def test_energy_reference_added_per_atom():
    """Per-element references shift the energy by exactly Σ_atoms ref[element]."""
    model = _tiny_model(3)
    e2t = {'H': 0, 'C': 1, 'O': 2}
    eref = {'H': 1.5, 'C': -2.0, 'O': 0.25}   # eV/atom
    atoms = _mol(('H', 'C', 'O'))

    base = ECENetCalculator(model, element_to_type=e2t, energy_reference={})
    ref  = ECENetCalculator(model, element_to_type=e2t, energy_reference=eref)

    shift = _energy(ref, atoms) - _energy(base, atoms)
    expected = sum(eref[s] for s in atoms.get_chemical_symbols())
    assert abs(shift - expected) < 1e-9, f"{shift} != {expected}"
    print(f"  energy_reference: shift={shift:.4f} eV == Σref={expected:.4f}")


def test_energy_mean_added_in_ev():
    """The training mean energy is added back, converted to eV by the unit factor."""
    model = _tiny_model(3)
    e2t = {'H': 0, 'C': 1, 'O': 2}
    atoms = _mol()

    base = ECENetCalculator(model, element_to_type=e2t,
                            energy_units='kcal/mol', energy_mean=0.0)
    shifted = ECENetCalculator(model, element_to_type=e2t,
                               energy_units='kcal/mol', energy_mean=10.0)

    shift = _energy(shifted, atoms) - _energy(base, atoms)
    assert abs(shift - 10.0 * _KCAL) < 1e-9, f"{shift} != {10.0 * _KCAL}"
    print(f"  energy_mean: shift={shift:.6f} eV == 10*kcal={10.0 * _KCAL:.6f}")


# ── element mapping ──────────────────────────────────────────────────────────

def test_unsupported_element_raises():
    model = _tiny_model(3)
    calc = ECENetCalculator(model, element_to_type={'H': 0, 'C': 1, 'O': 2})
    atoms = _mol(('H', 'C', 'O'))
    atoms[1].symbol = 'Fe'  # not in the mapping
    a = atoms.copy(); a.calc = calc
    try:
        a.get_potential_energy()
    except ValueError as e:
        assert 'Fe' in str(e)
        print(f"  unsupported element raises: {str(e)[:48]}…")
        return
    raise AssertionError("expected ValueError for unsupported element")


def test_forces_finite_and_shaped():
    model = _tiny_model(3)
    calc = ECENetCalculator(model, element_to_type={'H': 0, 'C': 1, 'O': 2})
    a = _mol(); a.calc = calc
    f = a.get_forces()
    assert f.shape == (3, 3) and np.isfinite(f).all()
    print(f"  forces: shape={f.shape} |F|max={np.abs(f).max():.3f}")


# ── periodic path (_compute_pbc / _compute_stress) ───────────────────────────

def test_pbc_energy_forces_stress_shapes():
    """The periodic path produces finite energy, (N,3) forces, and a 6-vector
    Voigt stress (exercises _compute_pbc and the strain-based _compute_stress)."""
    model = _tiny_model(3)
    calc = ECENetCalculator(model, element_to_type={'H': 0, 'C': 1, 'O': 2})
    a = _periodic_mol(); a.calc = calc
    e = a.get_potential_energy()
    f = a.get_forces()
    s = a.get_stress()                      # requests 'stress' → strain path
    assert np.isfinite(e)
    assert f.shape == (3, 3) and np.isfinite(f).all()
    assert s.shape == (6,) and np.isfinite(s).all()
    print(f"  pbc: E={e:.4f} |F|max={np.abs(f).max():.3f} |σ|max={np.abs(s).max():.3e}")


def test_pbc_forces_match_finite_difference():
    """Autograd forces on the periodic path agree with a central difference on
    the energy (validates _compute_pbc's grad wiring after the refactor)."""
    model = _tiny_model(3)
    calc = ECENetCalculator(model, element_to_type={'H': 0, 'C': 1, 'O': 2})
    atoms = _periodic_mol(); atoms.calc = calc
    f = atoms.get_forces()

    eps = 1e-5
    fd = np.zeros_like(f)
    for i in range(len(atoms)):
        for d in range(3):
            a = atoms.copy(); a.calc = calc
            p = a.get_positions(); p[i, d] += eps; a.set_positions(p)
            ep = a.get_potential_energy()
            p[i, d] -= 2 * eps; a.set_positions(p)
            em = a.get_potential_energy()
            fd[i, d] = -(ep - em) / (2 * eps)
    err = np.abs(f - fd).max()
    assert err < 1e-5, f"PBC forces vs finite-difference mismatch: {err:.2e}"
    print(f"  pbc forces match finite-difference (max err {err:.1e})")


# ── from_checkpoint ──────────────────────────────────────────────────────────

def test_from_checkpoint_type_to_idx_fallback():
    """Atomic-number 'type_to_idx' fallback (converted to symbols) + kcal/mol +
    energy_mean. Trainers now write 'element_to_type', but the calculator still
    accepts an atomic-number-keyed map for convenience."""
    model = _tiny_model(3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'tti.mdl')
        _save_ckpt(path, model, 3,
                   type_to_idx={1: 0, 6: 1, 8: 2},
                   energy_units='kcal/mol',
                   energy_mean=3.0)
        calc = ECENetCalculator.from_checkpoint(path)

    assert calc.element_to_type == {'H': 0, 'C': 1, 'O': 2}
    assert abs(calc._to_ev - _KCAL) < 1e-15
    assert abs(calc._energy_mean_ev - 3.0 * _KCAL) < 1e-12
    e = _energy(calc, _mol())
    assert np.isfinite(e)
    print(f"  from_checkpoint(type_to_idx fallback): map={calc.element_to_type} E={e:.4f}")


def test_from_checkpoint_defaults_to_ev_without_units():
    """No 'energy_units' key → defaults to eV (no dataset-based unit guessing)."""
    model = _tiny_model(3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'nounits.mdl')
        _save_ckpt(path, model, 3, type_to_idx={1: 0, 6: 1, 8: 2})  # no energy_units
        calc = ECENetCalculator.from_checkpoint(path)
    assert abs(calc._to_ev - 1.0) < 1e-15
    print(f"  from_checkpoint(no units key): defaults eV → _to_ev={calc._to_ev:.3f}")


def test_from_checkpoint_dtype_inferred_from_weights():
    model = _tiny_model(3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'dt.mdl')
        _save_ckpt(path, model, 3, type_to_idx={1: 0, 6: 1, 8: 2}, energy_units='eV')
        calc = ECENetCalculator.from_checkpoint(path)  # dtype=None
    assert calc.dtype == torch.float64
    print(f"  from_checkpoint(dtype): inferred {calc.dtype}")


def test_from_checkpoint_spice_style():
    """SPICE-style: a symbol-keyed 'element_to_type' + an 'e_ref' array. The
    calculator builds energy_reference from e_ref indexed by the checkpoint's
    OWN mapping — no import from the training scripts, no hardcoded element list.
    """
    e2t = {'H': 0, 'C': 1, 'O': 2}
    n = len(e2t)
    model = _tiny_model(n)
    e_ref = np.array([0.5, -1.0, 2.5])   # eV/atom, indexed by type idx
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'spice.mdl')
        _save_ckpt(path, model, n, element_to_type=e2t, e_ref=e_ref,
                   energy_units='eV')
        calc = ECENetCalculator.from_checkpoint(path)

    assert calc.element_to_type == e2t
    assert abs(calc._to_ev - 1.0) < 1e-15
    # energy_reference built generically from e_ref via the checkpoint's mapping
    for sym, idx in e2t.items():
        assert abs(calc.energy_reference[sym] - e_ref[idx]) < 1e-12
    print(f"  from_checkpoint(SPICE-style): refs={calc.energy_reference}")


def test_from_checkpoint_missing_mapping_raises():
    """A checkpoint with neither 'element_to_type' nor 'type_to_idx' fails loudly."""
    model = _tiny_model(3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'nomap.mdl')
        _save_ckpt(path, model, 3, energy_units='eV')  # no mapping at all
        try:
            ECENetCalculator.from_checkpoint(path)
        except ValueError as e:
            assert 'element mapping' in str(e)
            print(f"  from_checkpoint(no mapping): raises — {str(e)[:46]}…")
            return
    raise AssertionError("expected ValueError when no element mapping is stored")


def test_from_checkpoint_missing_hparams_raises():
    """A checkpoint without stored 'hparams' fails loudly (no reconstruction)."""
    model = _tiny_model(3)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'nohp.mdl')
        # Write a checkpoint dict deliberately lacking 'hparams'.
        torch.save({'model': model.state_dict(),
                    'element_to_type': {'H': 0, 'C': 1, 'O': 2}}, path)
        try:
            ECENetCalculator.from_checkpoint(path)
        except ValueError as e:
            assert 'hparams' in str(e)
            print(f"  from_checkpoint(no hparams): raises — {str(e)[:46]}…")
            return
    raise AssertionError("expected ValueError when no hparams are stored")


# ── real committed checkpoint (trained weights, not synthetic) ───────────────

_ETHANOL_MDL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'examples', 'ethanol.mdl')


def test_real_ethanol_checkpoint_single_point():
    """Load the committed ethanol checkpoint (real trained weights) and run a
    single-point on an ethanol molecule. The synthetic tests use random weights;
    this exercises from_checkpoint + the molecular path on a real model."""
    if not os.path.exists(_ETHANOL_MDL):
        print(f"  [skip] {_ETHANOL_MDL} not present")
        return
    from ase.build import molecule

    calc = ECENetCalculator.from_checkpoint(_ETHANOL_MDL)
    assert calc.element_to_type == {'H': 0, 'C': 1, 'O': 2}
    assert abs(calc._to_ev - _KCAL) < 1e-15          # rMD17 model → kcal/mol

    atoms = molecule('CH3CH2OH')                     # 9-atom ethanol (H/C/O)
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    assert np.isfinite(e)
    assert f.shape == (len(atoms), 3) and np.isfinite(f).all()
    print(f"  real ethanol.mdl: {len(atoms)} atoms, E={e:.2f} eV, |F|max={np.abs(f).max():.3f}")


if __name__ == '__main__':
    print("ECENetCalculator behaviour")
    test_energy_units_kcal_vs_ev_scaling()
    test_energy_reference_added_per_atom()
    test_energy_mean_added_in_ev()
    test_unsupported_element_raises()
    test_forces_finite_and_shaped()
    test_pbc_energy_forces_stress_shapes()
    test_pbc_forces_match_finite_difference()
    test_from_checkpoint_type_to_idx_fallback()
    test_from_checkpoint_defaults_to_ev_without_units()
    test_from_checkpoint_dtype_inferred_from_weights()
    test_from_checkpoint_spice_style()
    test_from_checkpoint_missing_mapping_raises()
    test_from_checkpoint_missing_hparams_raises()
    test_real_ethanol_checkpoint_single_point()
    print("All tests passed.")
