"""Run an NVT MD simulation starting from a SPICE dataset frame.

Usage:
    python examples/run_md_spice.py --checkpoint spice.mdl \\
        --xyz test_large_neut_all.xyz \\
        --config_type "DES370K Monomers" \\
        --temperature 300 --n_steps 20000 --timestep 0.5 \\
        --output traj.xyz --log md.log

A frame is drawn at random (or via --frame_idx) from the xyz file,
optionally filtered to a specific config_type subset.
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import re

import numpy as np
import torch
from ase import Atoms, units
from ase.io import write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from ecenet.calculator import ECENetCalculator
from scripts.train_ecenet_spice import ELEMENT_TO_TYPE

_CONFIG_TYPE_RE = re.compile(r'config_type=(?:"([^"]*)"|(\S+))')


def parse_xyz_frames(path, config_type_filter=None):
    """Read frames from a SPICE xyz file, return list of (symbols, positions) tuples."""
    frames = []
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                n_atoms = int(line)
            except ValueError:
                continue

            comment = f.readline()
            mc = _CONFIG_TYPE_RE.search(comment)
            config_type = (mc.group(1) or mc.group(2)) if mc else 'unknown'

            symbols   = []
            positions = []
            ok = True
            for i in range(n_atoms):
                parts = f.readline().split()
                elem = parts[0]
                if elem not in ELEMENT_TO_TYPE:
                    ok = False
                    for _ in range(n_atoms - i - 1):
                        f.readline()
                    break
                symbols.append(elem)
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

            if not ok:
                continue
            if config_type_filter is not None and config_type != config_type_filter:
                continue

            frames.append((symbols, np.array(positions, dtype=np.float64), config_type))

    return frames


def load_starting_frame(xyz_path, config_type=None, frame_idx=None, seed=0):
    """Load one frame from a SPICE xyz file as an ASE Atoms object."""
    print(f"Scanning {xyz_path}" +
          (f" (config_type='{config_type}')" if config_type else "") + "...")
    frames = parse_xyz_frames(xyz_path, config_type_filter=config_type)

    if not frames:
        raise ValueError(
            "No frames found" +
            (f" with config_type='{config_type}'" if config_type else "") +
            f" in {xyz_path}"
        )
    print(f"  {len(frames)} frames available")

    if frame_idx is None:
        rng = np.random.default_rng(seed)
        frame_idx = int(rng.integers(len(frames)))
        print(f"  Randomly selected frame {frame_idx}")
    else:
        print(f"  Using frame {frame_idx}")

    symbols, positions, ct = frames[frame_idx]
    print(f"  config_type: {ct}")
    return Atoms(symbols=symbols, positions=positions)


def run_md_spice(checkpoint, xyz='test_large_neut_all.xyz', config_type=None,
                 frame_idx=None, seed=0, temperature=300, timestep=0.5,
                 friction=0.01, n_steps=20000, log_every=100,
                 output='traj.xyz', log='md.log', device=None, float32=False,
                 log_timings=False):
    """Run NVT Langevin MD from a SPICE xyz starting frame.

    Importable entry point (see main() for the equivalent CLI). Writes a
    trajectory to `output` and a step log to `log`; returns the final Atoms.
    """
    dtype  = torch.float32 if float32 else torch.float64

    # ── Load starting frame ────────────────────────────────────────────────
    atoms = load_starting_frame(xyz, config_type,
                                frame_idx, seed)
    print(f"Molecule: {len(atoms)} atoms, "
          f"elements: {sorted(set(atoms.get_chemical_symbols()))}")

    # ── Calculator ─────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint}")
    # Newer checkpoints store their own element mapping; pass the SPICE mapping
    # as a fallback so checkpoints saved before that still load here.
    calc = ECENetCalculator.from_checkpoint(
        checkpoint, device=device, dtype=dtype,
        element_to_type=ELEMENT_TO_TYPE,
        log_timings=log_timings)
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
        step  = dyn.get_number_of_steps()
        t_ps  = step * timestep * 1e-3
        e_pot = atoms.get_potential_energy()
        e_kin = atoms.get_kinetic_energy()
        temp  = atoms.get_temperature()
        line  = f"{step:>8d}  {t_ps:>10.4f}  {e_pot:>14.6f}  {e_kin:>14.6f}  {temp:>8.2f}"
        print(line, flush=True)
        log_file.write(line + '\n')
        log_file.flush()

    dyn.attach(log_step, interval=log_every)

    print(f"\nRunning {n_steps} steps at {temperature} K "
          f"(dt={timestep} fs)...\n")
    log_step()
    dyn.run(n_steps)

    log_file.close()
    print(f"\nDone. Trajectory: {output}  Log: {log}")
    return atoms


def main():
    parser = argparse.ArgumentParser(description='NVT MD with ECENet on SPICE')
    parser.add_argument('--checkpoint',   required=True,
                        help='Path to trained .mdl checkpoint')
    parser.add_argument('--xyz',          default='test_large_neut_all.xyz',
                        help='SPICE xyz file to draw starting frame from')
    parser.add_argument('--config_type',  default=None,
                        help='Filter to a specific subset, e.g. "DES370K Monomers"')
    parser.add_argument('--frame_idx',    type=int, default=None,
                        help='Frame index within the (filtered) set (default: random)')
    parser.add_argument('--seed',         type=int, default=0)
    # MD settings
    parser.add_argument('--temperature',  type=float, default=300)
    parser.add_argument('--timestep',     type=float, default=0.5,
                        help='Timestep in fs')
    parser.add_argument('--friction',     type=float, default=0.01,
                        help='Langevin friction in 1/fs')
    parser.add_argument('--n_steps',      type=int,   default=20000)
    parser.add_argument('--log_every',    type=int,   default=100)
    # Output
    parser.add_argument('--output',       default='traj.xyz')
    parser.add_argument('--log',          default='md.log')
    # Calculator
    parser.add_argument('--device',       default=None)
    parser.add_argument('--float32',      action='store_true')
    parser.add_argument('--log_timings',  action='store_true',
                        help='Print NL/fwd/bwd timings for each step')
    args = parser.parse_args()
    run_md_spice(**vars(args))


if __name__ == '__main__':
    main()
