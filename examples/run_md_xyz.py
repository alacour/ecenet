"""Run an MD simulation (NVT or NVE) starting from an arbitrary xyz file.

Usage:
    python examples/run_md_xyz.py --xyz water.xyz --checkpoint model.mdl \\
        --ensemble nvt --temperature 300 --n_steps 10000 --timestep 0.5 \\
        --output traj.xyz --log md.log

The input is read with ASE, so extended-xyz headers (Lattice="..." and
pbc="...") are honoured automatically; a plain xyz is treated as a
non-periodic cluster. For a multi-frame file, pick the starting frame with
--frame_idx (default: last frame, ASE convention).

--ensemble nvt  -> Langevin thermostat at --temperature (uses --friction)
--ensemble nve  -> VelocityVerlet at constant energy (--temperature only
                   seeds the initial velocities)
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
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from ecenet.calculator import ECENetCalculator


def run_md_xyz(checkpoint, xyz, frame_idx=-1, seed=0, ensemble='nvt',
               temperature=300, timestep=0.5, friction=0.01, n_steps=10000,
               log_every=100, output='traj.xyz', log='md.log', device=None,
               float32=False, energy_units=None):
    """Run NVT/NVE MD from an arbitrary xyz file.

    Importable entry point (see main() for the equivalent CLI). Writes a
    trajectory to `output` and a step log to `log`; returns the final Atoms.
    """
    dtype = torch.float32 if float32 else torch.float64

    # ── Load starting frame ────────────────────────────────────────────────
    atoms = read(xyz, index=frame_idx)
    print(f"Loaded {xyz} (frame {frame_idx}): {len(atoms)} atoms")
    print(f"Elements: {sorted(set(atoms.get_chemical_symbols()))}")
    print(f"PBC: {atoms.pbc.tolist()}")

    # ── Calculator ─────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint}")
    calc = ECENetCalculator.from_checkpoint(
        checkpoint, device=device, dtype=dtype,
        energy_units=energy_units)
    atoms.calc = calc

    # Quick sanity check
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    print(f"Initial energy: {e:.4f} eV")
    print(f"Initial max force: {np.abs(f).max():.4f} eV/Å")

    # ── Initialise velocities ──────────────────────────────────────────────
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature,
                                 rng=np.random.default_rng(seed))

    # ── MD ─────────────────────────────────────────────────────────────────
    if ensemble == 'nvt':
        dyn = Langevin(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature,
            friction=friction / units.fs,
            rng=np.random.default_rng(seed + 1),
            # fixcm=True (ASE default) biases the kinetic temperature high,
            # markedly so for small systems. fixcm=False samples NVT correctly.
            fixcm=False,
        )
    else:  # nve
        dyn = VelocityVerlet(
            atoms,
            timestep=timestep * units.fs,
        )

    # Trajectory writer
    if output.endswith('.traj'):
        traj = Trajectory(output, 'w', atoms)
        dyn.attach(traj.write, interval=log_every)
    else:
        def write_frame():
            write(output, atoms, append=True)
        dyn.attach(write_frame, interval=log_every)

    # Logger
    log_file = open(log, 'w')
    log_file.write(f"{'step':>8}  {'time_ps':>10}  {'E_pot (eV)':>14}  "
                   f"{'E_kin (eV)':>14}  {'T (K)':>8}\n")
    log_file.flush()

    def log_step():
        step   = dyn.get_number_of_steps()
        t_ps   = step * timestep * 1e-3
        e_pot  = atoms.get_potential_energy()
        e_kin  = atoms.get_kinetic_energy()
        temp   = atoms.get_temperature()
        line   = f"{step:>8d}  {t_ps:>10.4f}  {e_pot:>14.6f}  {e_kin:>14.6f}  {temp:>8.2f}"
        print(line, flush=True)
        log_file.write(line + '\n')
        log_file.flush()

    dyn.attach(log_step, interval=log_every)

    print(f"\nRunning {n_steps} {ensemble.upper()} steps "
          f"(T={temperature} K, dt={timestep} fs)...\n")
    log_step()  # log initial state
    dyn.run(n_steps)

    log_file.close()
    print(f"\nDone. Trajectory: {output}  Log: {log}")
    return atoms


def main():
    parser = argparse.ArgumentParser(description='NVT/NVE MD with ECENet from an xyz file')
    parser.add_argument('--xyz',          required=True,
                        help='Input structure (xyz / extended-xyz)')
    parser.add_argument('--checkpoint',   required=True,
                        help='Path to trained .mdl checkpoint')
    parser.add_argument('--frame_idx',    type=int, default=-1,
                        help='Frame index to start from for multi-frame files '
                             '(default: -1, the last frame)')
    parser.add_argument('--seed',         type=int, default=0)
    # MD settings
    parser.add_argument('--ensemble',     default='nvt', choices=['nvt', 'nve'],
                        help='nvt = Langevin thermostat; nve = constant-energy VelocityVerlet')
    parser.add_argument('--temperature',  type=float, default=300,
                        help='Temperature in K (NVT target; for NVE only seeds initial velocities)')
    parser.add_argument('--timestep',     type=float, default=0.5,
                        help='Timestep in fs')
    parser.add_argument('--friction',     type=float, default=0.01,
                        help='Langevin friction in 1/fs (NVT only)')
    parser.add_argument('--n_steps',      type=int,   default=10000,
                        help='Number of MD steps')
    parser.add_argument('--log_every',    type=int,   default=100,
                        help='Print/log every N steps')
    # Output
    parser.add_argument('--output',       default='traj.xyz',
                        help='Trajectory output file (.xyz or .traj)')
    parser.add_argument('--log',          default='md.log')
    # Calculator
    parser.add_argument('--device',       default=None)
    parser.add_argument('--float32',      action='store_true')
    parser.add_argument('--energy_units', default=None,
                        choices=['eV', 'kcal/mol'],
                        help='Override unit conversion (auto-detected from checkpoint '
                             'if not set). Use kcal/mol for train_ecenet.py models, '
                             'eV for train_ecenet_spice.py models.')
    args = parser.parse_args()
    run_md_xyz(**vars(args))


if __name__ == '__main__':
    main()
