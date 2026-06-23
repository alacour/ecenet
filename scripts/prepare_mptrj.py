"""Pre-tensorize MPtrj into .pt shards for fast training (approach #3).

One-time preprocessing: parse the raw 11 GB JSON, build per-frame torch
tensors + PBC neighbor lists, write ~158 sharded .pt files plus a manifest +
type_map + e_ref. Training then opens the manifest and streams shards via
``ecenet.datasets.mptrj.MPtrjShardDataset`` — startup drops from ~35 min to
~10 s, and host RAM is bounded by the shard size, not the full dataset.

Two passes over the JSON:
  1. Collect type_map + per-frame composition + energy → least-squares e_ref.
  2. Tensorize each frame, build topology (optionally on GPU for speed),
     hold all frames in memory, **shuffle globally** so val-by-shard-suffix
     gives a frame-level random split, then write shards.

Usage:
    python scripts/prepare_mptrj.py --in MPtrj_2022.9_full.json \\
        --out $SCRATCH/mptrj_prepared \\
        --r_cut_edge 5.0 --r_cut_neighbor 4.0 \\
        --shard_size 10000 [--float32] [--device cuda:0] [--include_stress]

Notes:
- Single-process; ~30 min wall-clock for the full 1.5M frames on one A100.
- Peak host RAM during pass 2 ≈ 45 GB (all tensorized frames before shuffle +
  shard write). Run on a node with ≥64 GB RAM.
- r_cut_edge / r_cut_neighbor / dtype are baked into the prepared data; the
  trainer reads manifest.json and refuses to run if its args disagree.
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

# Reuse the trainer's JSON stream + structure decode + topology builder.
sys.path.insert(0, str(Path(__file__).parent))
from train_ecenet_mptrj import (
    STRESS_KBAR_TO_EVA3,
    _stream_json_materials,
    _structure_dict_to_arrays,
    build_topology,
    print_flush,
)

from ecenet import elements


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--in', dest='in_path', required=True,
                   help='Input MPtrj JSON (or .gz).')
    p.add_argument('--out', required=True,
                   help='Output directory for shards + manifest + type_map + e_ref.')
    p.add_argument('--r_cut_edge', type=float, default=5.0)
    p.add_argument('--r_cut_neighbor', type=float, default=4.0)
    p.add_argument('--energy_key', default='corrected_total_energy')
    p.add_argument('--shard_size', type=int, default=10_000,
                   help='Frames per shard file. 10k → ~158 files for full MPtrj.')
    p.add_argument('--float32', action='store_true',
                   help='Store tensors as float32 (default float64).')
    p.add_argument('--device', default=None,
                   help='Device for topology compute (e.g. cuda:0). Storage is CPU.')
    p.add_argument('--include_stress', action='store_true',
                   help='Convert + store stress tensors (eV/Å³). Otherwise None.')
    p.add_argument('--shuffle_seed', type=int, default=0,
                   help='Seed for the pre-shuffle (frame order across shards).')
    p.add_argument('--max_frames', type=int, default=None,
                   help='Stop after N frames (useful for quick sanity prep).')
    return p.parse_args()


def pass1_typemap_eref(in_path, energy_key, max_frames=None):
    """Stream the JSON; collect element set and (composition, energy) per frame;
    solve e_ref = argmin_x ||A x - E||² where A counts atoms of each type."""
    print_flush(f'Pass 1: scanning {in_path} for type_map + e_ref...')
    t0 = time.time()
    compositions = []   # list of (numbers: np.int32, energy: float)
    all_zs = set()
    for mp_id, frames in _stream_json_materials(in_path):
        for frame_key, fr in frames.items():
            struct = fr.get('structure')
            if struct is None:
                continue
            energy = fr.get(energy_key)
            if energy is None:
                for k in ('corrected_total_energy', 'uncorrected_total_energy', 'energy'):
                    if fr.get(k) is not None:
                        energy = fr[k]
                        break
            if energy is None:
                continue
            if fr.get('force', fr.get('forces')) is None:
                continue
            numbers, _, _ = _structure_dict_to_arrays(struct)
            numbers = numbers.astype(np.int32)
            all_zs.update(int(z) for z in numbers)
            compositions.append((numbers, float(energy)))
            if len(compositions) % 200_000 == 0:
                print_flush(f'  scanned {len(compositions):,} frames ({time.time()-t0:.0f}s)')
            if max_frames is not None and len(compositions) >= max_frames:
                break
        if max_frames is not None and len(compositions) >= max_frames:
            break

    type_map = elements.build_type_map(all_zs)
    n_types = len(type_map)
    print_flush(f'  Pass 1 done: {len(compositions):,} valid frames, {n_types} elements')

    # Compose A (count matrix) + solve.
    A = np.zeros((len(compositions), n_types), dtype=np.float64)
    E = np.zeros(len(compositions), dtype=np.float64)
    for i, (numbers, energy) in enumerate(compositions):
        for z in numbers:
            A[i, type_map[int(z)]] += 1.0
        E[i] = energy
    e_ref, *_ = np.linalg.lstsq(A, E, rcond=None)
    residual = float(np.linalg.norm(A @ e_ref - E))
    print_flush(f'  e_ref solved (residual L2 = {residual:.2f} eV across {len(compositions):,} frames)')
    return type_map, e_ref


def tensorize_frame(fr, mp_id, type_map, e_ref, args, dtype, compute_device):
    """Build a single per-frame dict (CPU tensors), or None to skip."""
    struct = fr.get('structure')
    if struct is None:
        return None
    energy = fr.get(args.energy_key)
    if energy is None:
        for k in ('corrected_total_energy', 'uncorrected_total_energy', 'energy'):
            if fr.get(k) is not None:
                energy = fr[k]
                break
    if energy is None:
        return None
    forces = fr.get('force', fr.get('forces'))
    if forces is None:
        return None
    forces = np.asarray(forces, dtype=np.float64)
    stress_arr = fr.get('stress')
    if stress_arr is not None:
        stress_arr = np.asarray(stress_arr, dtype=np.float64)

    numbers, positions, cell = _structure_dict_to_arrays(struct)
    types_np = np.array([type_map[int(z)] for z in numbers], dtype=np.int64)
    ref = sum(e_ref[type_map[int(z)]] for z in numbers)

    ei, ej, shift_e, ni, nj, shift_nb = build_topology(
        positions, cell, True, args.r_cut_edge, args.r_cut_neighbor,
        compute_device, dtype)

    volume = abs(np.linalg.det(cell))
    stress_t = None
    if args.include_stress and stress_arr is not None and volume > 0:
        stress_t = torch.tensor(stress_arr * STRESS_KBAR_TO_EVA3,
                                dtype=dtype, device='cpu')

    return {
        'pos':     torch.tensor(positions, dtype=dtype, device='cpu'),
        'types':   torch.tensor(types_np, dtype=torch.long, device='cpu'),
        'energy':  torch.tensor(float(energy) - ref, dtype=dtype, device='cpu'),
        'forces':  torch.tensor(forces, dtype=dtype, device='cpu'),
        'stress':  stress_t,
        'volume':  float(volume),
        'edge_i':  ei.cpu(),  'edge_j': ej.cpu(),  'shift_e': shift_e.cpu(),
        'nb_src':  ni.cpu(),  'nb_dst': nj.cpu(),  'shift_nb': shift_nb.cpu(),
        'n_atoms': int(len(numbers)),
        'mp_id':   mp_id,
    }


def pass2_tensorize(in_path, type_map, e_ref, args, dtype, compute_device):
    """Stream the JSON; tensorize each frame; accumulate ALL in memory so we
    can shuffle globally before sharding (so val-by-shard-suffix gives a
    frame-level random split that doesn't concentrate on specific materials).
    Returns the full list of frame dicts."""
    print_flush(f'Pass 2: tensorizing on {compute_device}, storing on CPU...')
    t0 = time.time()
    all_frames = []
    for mp_id, frames in _stream_json_materials(in_path):
        for frame_key, fr in frames.items():
            d = tensorize_frame(fr, mp_id, type_map, e_ref, args, dtype, compute_device)
            if d is None:
                continue
            all_frames.append(d)
            if len(all_frames) % 50_000 == 0:
                print_flush(f'  tensorized {len(all_frames):,} frames ({time.time()-t0:.0f}s)')
            if args.max_frames is not None and len(all_frames) >= args.max_frames:
                break
        if args.max_frames is not None and len(all_frames) >= args.max_frames:
            break
    return all_frames


def write_shards(all_frames, out_dir, shard_size, shuffle_seed):
    """Globally shuffle frame order (so val-as-shard-suffix is a uniform random
    sample of frames), then write fixed-size .pt shards. Returns shard filenames
    in write order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(all_frames)
    print_flush(f'Shuffling {n:,} frames (seed={shuffle_seed}) and writing shards of {shard_size}...')

    rng = np.random.RandomState(shuffle_seed)
    perm = rng.permutation(n)

    t0 = time.time()
    shard_paths = []
    n_shards = (n + shard_size - 1) // shard_size
    for si in range(n_shards):
        sl = perm[si * shard_size:(si + 1) * shard_size]
        shard = [all_frames[i] for i in sl]
        path = out_dir / f'shard_{si:05d}.pt'
        torch.save(shard, path)
        shard_paths.append(path.name)
        # Free this shard's references; the next-shard build still pulls from
        # all_frames so we can't fully drop until the loop ends.
        del shard
        if (si + 1) % 20 == 0 or si == n_shards - 1:
            print_flush(f'  wrote {si+1}/{n_shards} shards ({time.time()-t0:.0f}s)')
    return shard_paths


def main():
    args = parse_args()
    dtype = torch.float32 if args.float32 else torch.float64
    compute_device = torch.device(args.device) if args.device else torch.device('cpu')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: type_map + e_ref
    type_map, e_ref = pass1_typemap_eref(args.in_path, args.energy_key, args.max_frames)

    # Pass 2: tensorize all frames (peak ~45 GB RAM for the full set)
    all_frames = pass2_tensorize(args.in_path, type_map, e_ref, args,
                                 dtype, compute_device)

    # Shuffle + shard
    shard_paths = write_shards(all_frames, out_dir, args.shard_size, args.shuffle_seed)

    # Sidecar metadata
    torch.save(type_map, out_dir / 'type_map.pt')
    torch.save(torch.from_numpy(np.asarray(e_ref, dtype=np.float64)),
               out_dir / 'e_ref.pt')

    manifest = {
        'n_frames': len(all_frames),
        'n_types': len(type_map),
        'n_shards': len(shard_paths),
        'shard_size': args.shard_size,
        'r_cut_edge': args.r_cut_edge,
        'r_cut_neighbor': args.r_cut_neighbor,
        'dtype': 'float32' if args.float32 else 'float64',
        'energy_key': args.energy_key,
        'include_stress': bool(args.include_stress),
        'shuffle_seed': int(args.shuffle_seed),
        'source': str(Path(args.in_path).resolve()),
        'shards': shard_paths,
    }
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    print_flush(f'\nDone: {len(all_frames):,} frames in {len(shard_paths)} shards → {out_dir}')


if __name__ == '__main__':
    main()
