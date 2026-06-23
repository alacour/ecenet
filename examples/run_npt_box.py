"""Run an NPT MD simulation starting from a pre-built periodic box.

Uses IsotropicMTKNPT (Nosé-Hoover chain thermostat + MTK barostat with
purely isotropic volume fluctuations), so the cell shape is preserved
as a cube throughout the simulation.  Stress is computed via the
strain-based method in ECENetCalculator.

Usage:
    python examples/run_npt_box.py --checkpoint spice.mdl --box water_box.xyz \\
        --temperature 300 --pressure 1.0 --n_steps 100000 \\
        --timestep 0.5 --output traj.xyz --log md.log

Pressure units: bar (default 1.0 bar ≈ 1 atm).
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse

import numpy as np
import torch
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ecenet.calculator import ECENetCalculator


def run_npt_box(checkpoint, box, cell=None, frame_idx=-1, seed=0,
                temperature=300, pressure=1.0, tdamp=25.0, pdamp=75.0,
                timestep=0.5, n_steps=100000, log_every=100,
                output='traj_npt.xyz', log='md_npt.log', device=None,
                float32=False, log_timings=False):
    """Run NPT MD from a pre-built periodic box.

    Importable entry point (see main() for the equivalent CLI). Writes a
    trajectory to `output` and a step log to `log`; returns the final Atoms.
    """
    dtype  = torch.float32 if float32 else torch.float64

    # ── Load box ───────────────────────────────────────────────────────────
    atoms = read(box, index=frame_idx)
    atoms.set_pbc(True)

    if cell is not None:
        atoms.set_cell([cell, cell, cell])

    if not atoms.cell.any():
        raise ValueError(
            "No cell found in the xyz file. "
            "Pass --cell <side_length_angstrom> to set it manually."
        )

    print(f"Box: {len(atoms)} atoms")
    print(f"Cell: {atoms.cell.lengths()} Å")
    print(f"Initial volume: {atoms.get_volume():.3f} Å³")
    masses = atoms.get_masses()
    density_g_cm3 = masses.sum() / atoms.get_volume() / 0.60221  # amu/Å³ → g/cm³
    print(f"Initial density: {density_g_cm3:.4f} g/cm³")
    print(f"Elements: {sorted(set(atoms.get_chemical_symbols()))}")

    # ── Calculator ─────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {checkpoint}")
    calc = ECENetCalculator.from_checkpoint(
        checkpoint, device=device, dtype=dtype,
        log_timings=log_timings)
    atoms.calc = calc

    # Quick sanity check (requests stress to verify it works)
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    s = atoms.get_stress()          # (6,) Voigt, eV/Å³
    p_gpa = -s[:3].mean() / units.GPa
    print(f"Initial energy:    {e:.4f} eV  ({e/len(atoms)*1000:.2f} meV/atom)")
    print(f"Initial max force: {np.abs(f).max():.4f} eV/Å")
    print(f"Initial pressure:  {p_gpa:.4f} GPa  ({p_gpa*10000:.1f} bar)")

    # ── Initialise velocities ──────────────────────────────────────────────
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature,
                                 rng=np.random.default_rng(seed))

    # ── NPT integrator ─────────────────────────────────────────────────────
    # IsotropicMTKNPT: Nosé-Hoover chain thermostat + MTK barostat.
    # Cell shape is preserved (cubic stays cubic); only volume changes.
    dyn = IsotropicMTKNPT(
        atoms,
        timestep=timestep * units.fs,
        temperature_K=temperature,
        pressure_au=pressure * units.bar,
        tdamp=tdamp * units.fs,
        pdamp=pdamp * units.fs,
    )

    # ── Trajectory writer ──────────────────────────────────────────────────
    if output.endswith('.traj'):
        traj = Trajectory(output, 'w', atoms)
        dyn.attach(traj.write, interval=log_every)
    else:
        def write_frame():
            write(output, atoms, append=True)
        dyn.attach(write_frame, interval=log_every)

    # ── Logger ─────────────────────────────────────────────────────────────
    log_file = open(log, 'w')
    header = (f"{'step':>8}  {'time_ps':>9}  {'E_pot(eV)':>14}  "
              f"{'E_kin(eV)':>14}  {'T(K)':>8}  "
              f"{'P(bar)':>10}  {'V(Å³)':>12}  {'rho(g/cc)':>10}")
    log_file.write(header + '\n')
    log_file.flush()
    print(header, flush=True)

    def log_step():
        step   = dyn.get_number_of_steps()
        t_ps   = step * timestep * 1e-3
        e_pot  = atoms.get_potential_energy()
        e_kin  = atoms.get_kinetic_energy()
        temp   = atoms.get_temperature()
        stress = atoms.get_stress(include_ideal_gas=True)  # Voigt (6,), eV/Å³, includes kinetic
        p_bar  = -stress[:3].mean() / units.bar
        vol    = atoms.get_volume()
        rho    = masses.sum() / vol / 0.60221
        line   = (f"{step:>8d}  {t_ps:>9.4f}  {e_pot:>14.6f}  "
                  f"{e_kin:>14.6f}  {temp:>8.2f}  "
                  f"{p_bar:>10.2f}  {vol:>12.4f}  {rho:>10.5f}")
        print(line, flush=True)
        log_file.write(line + '\n')
        log_file.flush()

    dyn.attach(log_step, interval=log_every)

    # ── Run ────────────────────────────────────────────────────────────────
    print(f"\nRunning {n_steps} NPT steps at {temperature} K, "
          f"{pressure} bar  (dt={timestep} fs)\n")
    log_step()
    dyn.run(n_steps)

    log_file.close()
    print(f"\nDone. Trajectory: {output}  Log: {log}")
    return atoms


def main():
    parser = argparse.ArgumentParser(description='NPT MD from a periodic box')
    parser.add_argument('--checkpoint',   required=True)
    parser.add_argument('--box',          required=True,
                        help='xyz file of the pre-built periodic box (extxyz)')
    parser.add_argument('--cell',         type=float, default=None,
                        help='Override cubic cell side length in Å')
    parser.add_argument('--frame_idx',    type=int, default=-1,
                        help='Frame index to use as starting point (default: last)')
    parser.add_argument('--seed',         type=int, default=0)
    # Thermostat / barostat
    parser.add_argument('--temperature',  type=float, default=300,
                        help='Target temperature in K')
    parser.add_argument('--pressure',     type=float, default=1.0,
                        help='Target pressure in bar (default: 1 bar)')
    parser.add_argument('--tdamp',         type=float, default=25.0,
                        help='Thermostat damping time in fs (default: 25)')
    parser.add_argument('--pdamp',         type=float, default=75.0,
                        help='Barostat damping time in fs (default: 75)')
    # MD settings
    parser.add_argument('--timestep',     type=float, default=0.5,
                        help='Timestep in fs')
    parser.add_argument('--n_steps',      type=int,   default=100000)
    parser.add_argument('--log_every',    type=int,   default=100)
    # Output
    parser.add_argument('--output',       default='traj_npt.xyz')
    parser.add_argument('--log',          default='md_npt.log')
    # Calculator
    parser.add_argument('--device',       default=None)
    parser.add_argument('--float32',      action='store_true')
    parser.add_argument('--log_timings',  action='store_true',
                        help='Print NL/fwd timings for each step')
    args = parser.parse_args()
    run_npt_box(**vars(args))


if __name__ == '__main__':
    main()
