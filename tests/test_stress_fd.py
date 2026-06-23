"""Finite-difference test for stress (pressure) in ECENetCalculator.

For each of the 9 strain components ε_ij the analytic stress
  σ_ij = (1/V) dE/dε_ij
is compared against the central-difference estimate
  σ_ij ≈ (E(+δ·e_ij) - E(-δ·e_ij)) / (2δV)

The FD calculation uses the *same* frozen neighbor topology as the analytic
one (standard MLIP approximation), so the two should agree to machine
precision modulo the O(δ²) FD truncation error.

Usage:
    python test_stress_fd.py --checkpoint spice.mdl --box water_box.xyz
    python test_stress_fd.py --checkpoint spice.mdl --box water_box.xyz \\
        --delta 1e-4 --tol 1e-4 --float32
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse

import numpy as np
import torch
from ase import units
from ase.io import read

from ecenet.calculator import ECENetCalculator

# ── helpers ────────────────────────────────────────────────────────────────────

def _energy_at_strain(model, pos, types, edge_i, edge_j,
                      shift_vecs_edge, nb_src, nb_dst, shift_vecs_nb,
                      strain_np, dtype, device):
    """Energy (model units, scalar) with a given 3×3 strain applied."""
    eps = torch.tensor(strain_np, dtype=dtype, device=device)
    pos_s      = pos + pos @ eps
    shift_e_s  = shift_vecs_edge + shift_vecs_edge @ eps
    shift_nb_s = shift_vecs_nb   + shift_vecs_nb   @ eps
    with torch.no_grad():
        e = model.forward_pbc(pos_s, types, edge_i, edge_j,
                              shift_e_s, nb_src, nb_dst, shift_nb_s)
    return e.item()


def fd_stress_matrix(model, pos, types, edge_i, edge_j,
                     shift_vecs_edge, nb_src, nb_dst, shift_vecs_nb,
                     volume, to_ev, delta, dtype, device):
    """Full 3×3 stress matrix (eV/Å³) via central differences."""
    sigma = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            eps_p = np.zeros((3, 3)); eps_p[i, j] =  delta
            eps_m = np.zeros((3, 3)); eps_m[i, j] = -delta
            e_p = _energy_at_strain(
                model, pos, types, edge_i, edge_j,
                shift_vecs_edge, nb_src, nb_dst, shift_vecs_nb,
                eps_p, dtype, device)
            e_m = _energy_at_strain(
                model, pos, types, edge_i, edge_j,
                shift_vecs_edge, nb_src, nb_dst, shift_vecs_nb,
                eps_m, dtype, device)
            sigma[i, j] = (e_p - e_m) / (2 * delta) * to_ev / volume
    return sigma


def analytic_stress_matrix(model, pos, types, edge_i, edge_j,
                            shift_vecs_edge, nb_src, nb_dst, shift_vecs_nb,
                            volume, to_ev, dtype, device):
    """Full 3×3 stress matrix (eV/Å³) via autograd — same path as the calculator."""
    strain = torch.zeros(3, 3, dtype=dtype, device=device, requires_grad=True)
    pos_s      = pos + pos @ strain
    shift_e_s  = shift_vecs_edge + shift_vecs_edge @ strain
    shift_nb_s = shift_vecs_nb   + shift_vecs_nb   @ strain
    with torch.enable_grad():
        e = model.forward_pbc(pos_s, types, edge_i, edge_j,
                              shift_e_s, nb_src, nb_dst, shift_nb_s)
    stress_grad = torch.autograd.grad(e, strain)[0]
    return stress_grad.detach().cpu().numpy() * to_ev / volume


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='FD stress test')
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--box',         required=True,
                        help='Periodic box in extxyz format')
    parser.add_argument('--frame_idx',   type=int, default=0)
    parser.add_argument('--delta',       type=float, default=1e-4,
                        help='Strain perturbation magnitude (default: 1e-4)')
    parser.add_argument('--tol',         type=float, default=3e-4,
                        help='Relative tolerance for pass/fail (default: 3e-4)')
    parser.add_argument('--device',      default=None)
    parser.add_argument('--float32',     action='store_true')
    args = parser.parse_args()

    dtype = torch.float32 if args.float32 else torch.float64

    # ── load ──────────────────────────────────────────────────────────────────
    atoms = read(args.box, index=args.frame_idx)
    atoms.set_pbc(True)
    if not atoms.cell.any():
        raise ValueError("No cell in xyz file; set PBC cell before running.")

    print(f"System: {len(atoms)} atoms, cell {atoms.cell.lengths()} Å")

    calc = ECENetCalculator.from_checkpoint(
        args.checkpoint, device=args.device, dtype=dtype)
    model  = calc.model
    device = calc.device
    to_ev  = calc._to_ev

    symbols = atoms.get_chemical_symbols()
    pos = torch.tensor(atoms.get_positions(),
                       dtype=dtype, device=device)
    types = torch.tensor([calc.element_to_type[s] for s in symbols],
                         dtype=torch.long, device=device)
    cell   = atoms.get_cell().array
    volume = abs(np.linalg.det(cell))

    edge_i, edge_j, shift_e = calc._gpu_neighbor_list(
        pos, cell, model.r_cut_edge)
    nb_src, nb_dst, shift_nb = calc._gpu_neighbor_list(
        pos, cell, model.r_cut_neighbor)

    print(f"Edges: {len(edge_i)}  NB pairs: {len(nb_src)}  Volume: {volume:.3f} Å³\n")

    # ── compute ───────────────────────────────────────────────────────────────
    print("Computing analytic stress (autograd)...")
    sigma_ana = analytic_stress_matrix(
        model, pos, types, edge_i, edge_j, shift_e, nb_src, nb_dst, shift_nb,
        volume, to_ev, dtype, device)

    print(f"Computing FD stress (δ = {args.delta}, 18 forward passes)...")
    sigma_fd = fd_stress_matrix(
        model, pos, types, edge_i, edge_j, shift_e, nb_src, nb_dst, shift_nb,
        volume, to_ev, args.delta, dtype, device)

    # ── report ────────────────────────────────────────────────────────────────
    comp_labels = [
        ('xx', 0, 0), ('yy', 1, 1), ('zz', 2, 2),
        ('yz', 1, 2), ('xz', 0, 2), ('xy', 0, 1),
        ('zy', 2, 1), ('zx', 2, 0), ('yx', 1, 0),
    ]

    print(f"\n{'Component':<10} {'Analytic (eV/Å³)':>18} {'FD (eV/Å³)':>18} "
          f"{'Abs err':>12} {'Rel err':>10}  Status")
    print('─' * 80)

    # Scale for absolute tolerance: largest diagonal stress or 1e-8 floor
    stress_scale = max(np.abs(sigma_ana).max(), 1e-8)

    all_pass = True
    for label, i, j in comp_labels:
        ana     = sigma_ana[i, j]
        fd      = sigma_fd[i, j]
        abs_err = abs(ana - fd)
        ref     = max(abs(ana), 1e-10)
        rel_err = abs_err / ref
        # Pass if relative error is small OR absolute error is small relative
        # to the overall stress scale (handles near-zero off-diagonal components)
        ok      = rel_err < args.tol or abs_err < stress_scale * 1e-3
        status  = 'PASS' if ok else 'FAIL'
        if not ok:
            all_pass = False
        print(f"{label:<10} {ana:>18.8f} {fd:>18.8f} "
              f"{abs_err:>12.2e} {rel_err:>10.2e}  {status}")

    # Symmetry check: stress tensor should be symmetric
    sym_err = np.abs(sigma_ana - sigma_ana.T).max()
    print(f"\nSymmetry of analytic σ (max |σ - σᵀ|): {sym_err:.2e}")

    # Pressure summary
    p_ana = -sigma_ana.trace() / 3
    p_fd  = -sigma_fd.trace()  / 3
    print("\nPressure:")
    print(f"  Analytic: {p_ana:.8f} eV/Å³  = {p_ana/units.GPa:>10.4f} GPa"
          f"  = {p_ana/units.bar:>12.2f} bar")
    print(f"  FD:       {p_fd:.8f} eV/Å³  = {p_fd/units.GPa:>10.4f} GPa"
          f"  = {p_fd/units.bar:>12.2f} bar")
    print(f"  Abs err:  {abs(p_ana-p_fd):.2e} eV/Å³")

    print(f"\n{'ALL PASS' if all_pass else 'SOME FAILURES'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
