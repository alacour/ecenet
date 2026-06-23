"""Run an NVT MD simulation starting from a random rMD17 frame.

Usage:
    python examples/run_md_rmd17_langevin.py --checkpoint aspirin.mdl --molecule aspirin \\
        --temperature 300 --n_steps 10000 --timestep 0.5 \\
        --output traj.xyz --log md.log

The starting frame is drawn from the rMD17 dataset (random by default,
or specify --frame_idx to pick a specific one).
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse

import numpy as np
import torch
from ase import Atoms, units
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ecenet import elements
from ecenet.calculator import ECENetCalculator
from scripts.train_ecenet import MD22_MOLECULES, RMD17_MOLECULES


def load_starting_frame(molecule, data_dir=None, frame_idx=None, seed=0):
    """Load one frame from rMD17/MD22 as an ASE Atoms object."""
    if molecule in RMD17_MOLECULES:
        data_dir = data_dir or 'rmd17/npz_data'
        data = np.load(f'{data_dir}/rmd17_{molecule}.npz')
        positions      = data['coords']    # (N_frames, N_atoms, 3) Å
        atomic_numbers = data['nuclear_charges']
    elif molecule in MD22_MOLECULES:
        data_dir = data_dir or 'md22'
        data = np.load(f'{data_dir}/md22_{molecule}.npz')
        positions      = data['R']
        atomic_numbers = data['z']
    else:
        raise ValueError(f"Unknown molecule '{molecule}'")

    if frame_idx is None:
        rng = np.random.default_rng(seed)
        frame_idx = int(rng.integers(len(positions)))
        print(f"Randomly selected frame {frame_idx} / {len(positions)}")
    else:
        print(f"Using frame {frame_idx} / {len(positions)}")

    symbols = [elements.symbol(z) for z in atomic_numbers]
    atoms = Atoms(symbols=symbols, positions=positions[frame_idx])
    return atoms


def run_md_rmd17_langevin(checkpoint, molecule='aspirin', data_dir=None,
                          frame_idx=None, seed=0, temperature=300.0,
                          timestep=0.5, friction=0.01, n_steps=10000,
                          log_every=100, output='traj.xyz', log='md.log',
                          device=None, float32=False, energy_units=None):
    """Run NVT Langevin MD from an rMD17/MD22 starting frame.

    Importable entry point (see main() for the equivalent CLI). Writes a
    trajectory to `output` and a step log to `log`; returns the final Atoms.
    """
    dtype = torch.float32 if float32 else torch.float64

    # ── Load starting frame ────────────────────────────────────────────────
    atoms = load_starting_frame(molecule, data_dir, frame_idx, seed)
    print(f"Molecule: {molecule}, {len(atoms)} atoms")
    print(f"Elements: {sorted(set(atoms.get_chemical_symbols()))}")

    # ── Calculator ─────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint}")
    calc = ECENetCalculator.from_checkpoint(
        checkpoint, device=device, dtype=dtype, energy_units=energy_units)
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

    print(f"\nRunning {n_steps} steps at {temperature} K "
          f"(dt={timestep} fs)...\n")
    log_step()  # log initial state
    dyn.run(n_steps)

    log_file.close()
    print(f"\nDone. Trajectory: {output}  Log: {log}")
    return atoms


def main():
    parser = argparse.ArgumentParser(description='NVT MD with ECENet on rMD17')
    parser.add_argument('--checkpoint',   required=True,
                        help='Path to trained .mdl checkpoint')
    parser.add_argument('--molecule',     default='aspirin',
                        help='rMD17/MD22 molecule name')
    parser.add_argument('--data_dir',     default=None)
    parser.add_argument('--frame_idx',    type=int, default=None,
                        help='Starting frame index (default: random)')
    parser.add_argument('--seed',         type=int, default=0)
    # MD settings
    parser.add_argument('--temperature',  type=float, default=300,
                        help='Temperature in K')
    parser.add_argument('--timestep',     type=float, default=0.5,
                        help='Timestep in fs')
    parser.add_argument('--friction',     type=float, default=0.01,
                        help='Langevin friction in 1/fs')
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
    run_md_rmd17_langevin(**vars(args))


if __name__ == '__main__':
    main()
