"""Evaluate a trained ECENet checkpoint on the SPICE test set.

Breaks down results by config_type (data source subset).

Usage (run from the repo root):
    python scripts/eval_spice.py \
        --checkpoint molecule.mdl \
        --test_xyz ../test_large_neut_all.xyz \
        --l_max 3 --n_max 10 --embed_dim 32 --n_max_d 16 \
        --r_cut_edge 8.0 --r_cut_neighbor 6.0 \
        --n_mp 2 \
        --output_hidden_dims 128 128 \
        --n_dist_basis 16 --n_grid 16 \
        --batch_size 8
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path
from train_ecenet_spice import (
    _ENERGY_RE,
    ELEMENT_TO_TYPE,
    N_TYPES,
    TYPE_NAMES,
    compute_energy_reference,
)

from ecenet import ECENet

_CONFIG_TYPE_RE = re.compile(r'config_type=(?:"([^"]*)"|(\S+))')


def parse_xyz_file(path, dtype=np.float32):
    structures = []
    unknown_elements = set()
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
            m = _ENERGY_RE.search(comment)
            energy = float(m.group(1)) if m else 0.0
            mc = _CONFIG_TYPE_RE.search(comment)
            config_type = (mc.group(1) or mc.group(2)) if mc else 'unknown'

            positions = np.empty((n_atoms, 3), dtype=dtype)
            forces    = np.empty((n_atoms, 3), dtype=dtype)
            types     = np.empty(n_atoms, dtype=np.int16)

            ok = True
            for i in range(n_atoms):
                parts = f.readline().split()
                elem = parts[0]
                if elem not in ELEMENT_TO_TYPE:
                    unknown_elements.add(elem)
                    ok = False
                    for _ in range(n_atoms - i - 1):
                        f.readline()
                    break
                types[i]     = ELEMENT_TO_TYPE[elem]
                positions[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
                forces[i]    = [float(parts[4]), float(parts[5]), float(parts[6])]

            if not ok:
                continue

            structures.append({
                'positions':   positions,
                'forces':      forces,
                'energy':      energy,
                'types':       types,
                'n_atoms':     n_atoms,
                'config_type': config_type,
            })

    if unknown_elements:
        print(f"Warning: skipped structures with unknown elements: {unknown_elements}")
    return structures


def evaluate_by_subset(energy_fn, structures, e_ref, dtype, device, batch_size=8):
    """Evaluate energy_fn(pos_list, typ_list) on structures, returning per-subset MAE."""
    # Group indices by subset
    subsets = defaultdict(list)
    for i, s in enumerate(structures):
        subsets[s['config_type']].append(i)

    results = {}
    for subset_name, indices in sorted(subsets.items()):
        energy_abs  = 0.0
        force_abs   = 0.0
        force_count = 0

        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            pos_b, frc_b, typ_b, eng_b_target = [], [], [], []

            for i in batch:
                s = structures[i]
                pos_b.append(
                    torch.tensor(s['positions'], dtype=dtype, device=device)
                    .requires_grad_(True)
                )
                frc_b.append(torch.tensor(s['forces'], dtype=dtype, device=device))
                typ_b.append(torch.tensor(s['types'].astype(np.int64),
                                          dtype=torch.long, device=device))
                ref = sum(e_ref[t] for t in s['types'])
                eng_b_target.append(s['energy'] - ref)

            with torch.enable_grad():
                eng_pred = energy_fn(pos_b, typ_b)
                grads = torch.autograd.grad(eng_pred.sum(), pos_b)

            for k in range(len(batch)):
                n_at = structures[batch[k]]['n_atoms']
                energy_abs  += abs(eng_pred[k].item() - eng_b_target[k]) / n_at
                force_abs   += (-grads[k] - frc_b[k]).abs().sum().item()
                force_count += frc_b[k].numel()

        n = len(indices)
        e_mae = energy_abs / n          # eV/atom
        f_mae = force_abs / force_count  # eV/Å
        results[subset_name] = {
            'n': n,
            'energy_mae_ev_atom':  e_mae,
            'force_mae_ev_ang':    f_mae,
            'energy_mae_mev_atom': e_mae * 1000,
            'force_mae_mev_ang':   f_mae * 1000,
        }
        print(f"  {subset_name:30s} (n={n:6d}): "
              f"E={e_mae*1000:6.2f} meV/atom  F={f_mae*1000:6.1f} meV/Å")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',       required=True)
    parser.add_argument('--test_xyz',         default='test_large_neut_all.xyz')
    parser.add_argument('--train_xyz',        default='train_large_neut_no_bad_clean.xyz',
                        help='Used only to compute energy reference (needs same ref as training)')
    parser.add_argument('--n_train_ref',      type=int, default=50000,
                        help='Structures to use for reference energy fit')
    # Architecture (must match checkpoint)
    parser.add_argument('--r_cut_edge',       type=float, default=5.0)
    parser.add_argument('--r_cut_neighbor',   type=float, default=4.0)
    parser.add_argument('--l_max',            type=int,   default=3)
    parser.add_argument('--n_max',            type=int,   default=4)
    parser.add_argument('--embed_dim',        type=int,   default=32)
    parser.add_argument('--n_layers',         type=int,   default=2)
    parser.add_argument('--n_max_d',          type=int,   default=8)
    parser.add_argument('--n_grid',           type=int,   default=None)
    parser.add_argument('--output_hidden_dims', type=int, nargs='+', default=None)
    parser.add_argument('--activation',       default='silu')
    parser.add_argument('--no_nonlinearity',  action='store_true')
    parser.add_argument('--n_dist_embed',     type=int,   default=0)
    parser.add_argument('--n_mp', type=int, default=1)
    parser.add_argument('--n_dist_basis',     type=int,   default=8)
    # Eval options
    parser.add_argument('--batch_size',       type=int,   default=8)
    parser.add_argument('--float32',          action='store_true')
    parser.add_argument('--device',           default=None)
    args = parser.parse_args()

    dtype  = torch.float32 if args.float32 else torch.float64
    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}, dtype: {dtype}")

    # ── Load checkpoint ────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # ── Energy reference ───────────────────────────────────────────────────
    if 'e_ref' in ckpt:
        e_ref = np.array(ckpt['e_ref'], dtype=np.float64)
        print("Reference energies from checkpoint (eV/atom):")
    else:
        print(f"Loading {args.n_train_ref:,} train structures for energy reference...")
        train_raw = parse_xyz_file(args.train_xyz)[:args.n_train_ref]
        e_ref = compute_energy_reference(train_raw)
        del train_raw
        print("Reference energies (computed, eV/atom):")
    for t, name in enumerate(TYPE_NAMES):
        print(f"  {name}: {e_ref[t]:.4f}")

    # ── Test data ──────────────────────────────────────────────────────────
    print(f"\nLoading test data from {args.test_xyz}...")
    test_structs = parse_xyz_file(args.test_xyz)
    print(f"  {len(test_structs):,} structures")
    subsets = defaultdict(int)
    for s in test_structs:
        subsets[s['config_type']] += 1
    for k, v in sorted(subsets.items()):
        print(f"  {k}: {v:,}")

    # ── Build model from hparams ───────────────────────────────────────────
    # Use hparams from checkpoint if available, fall back to CLI args
    hp = ckpt.get('hparams', {})
    if hp:
        print("Using hyperparameters from checkpoint.")
    else:
        print("No hparams in checkpoint — using CLI arguments.")
        hp = dict(
            n_types=N_TYPES,
            r_cut_edge=args.r_cut_edge, r_cut_neighbor=args.r_cut_neighbor,
            l_max=args.l_max, n_max=args.n_max, embed_dim=args.embed_dim,
            n_layers=args.n_layers, n_max_d=args.n_max_d, n_grid=args.n_grid,
            cutoff_type='cosine', activation=args.activation,
            use_nonlinearity=not args.no_nonlinearity,
            output_hidden_dims=args.output_hidden_dims,
            analytic_ace_basis=True,
            n_dist_embed=args.n_dist_embed,
            n_mp=args.n_mp, n_dist_basis=args.n_dist_basis,
        )

    # Strip any long-range-only keys from old checkpoints (not SR constructor args)
    for k in ['r_cut_lr', 'lr_n_rbf', 'lr_embed_dim', 'lr_hidden_layers', 'lr_module_type']:
        hp.pop(k, None)

    # nonlinearity_type removed (the only kind was 'realspace'). A legacy 'none'
    # checkpoint is linear-only → map it to use_nonlinearity=False so its
    # (nonlin-free) weights still load, then drop the key.
    if hp.pop('nonlinearity_type', 'realspace') == 'none':
        hp['use_nonlinearity'] = False

    n_mp = hp.pop('n_mp', None)
    legacy_used_mp = bool(hp.get('use_message_passing'))
    for k in ['use_message_passing', 'n_mp_steps', 'n_layers_per_mp', 'n_final_layers']:
        hp.pop(k, None)
    if n_mp is None:
        if legacy_used_mp:
            print("WARNING: legacy message-passing checkpoint — the MP architecture "
                  "changed (n_mp stages); MP/final-layer weights will not load, "
                  "evaluating with n_mp=1 (no MP). Re-save hparams with 'n_mp' to load MP.")
        n_mp = 1

    model = ECENet(**hp, n_mp=n_mp)
    if dtype == torch.float64:
        model = model.double()
    model = model.to(device)

    state = ckpt.get('best_state') or ckpt.get('model')
    # Old long-range checkpoints stored best_state = {'model': ..., 'lr_module': ...};
    # load just the short-range model.
    if isinstance(state, dict) and 'model' in state and 'lr_module' in state:
        state = state['model']
    model.load_state_dict(state, strict=False)
    model.eval()
    epoch = ckpt.get('epoch', '?')
    print(f"\nLoaded checkpoint (epoch {epoch}, "
          f"best val F-MAE={ckpt.get('best_val_force_mae', float('nan')):.4f} eV/Å)")

    def energy_fn(pos_list, typ_list):
        return model.forward_batch_multi(pos_list, typ_list)

    # ── Evaluate ───────────────────────────────────────────────────────────
    print(f"\nEvaluating on test set (batch_size={args.batch_size})...\n")
    results = evaluate_by_subset(energy_fn, test_structs, e_ref, dtype, device,
                                 batch_size=args.batch_size)

    # Overall
    total_e = sum(r['energy_mae_ev_atom'] * r['n'] for r in results.values())
    total_f_num = sum(r['force_mae_ev_ang'] * r['n'] for r in results.values())
    total_n = sum(r['n'] for r in results.values())
    print(f"\n  {'Overall':30s} (n={total_n:6d}): "
          f"E={total_e/total_n*1000:6.2f} meV/atom  "
          f"F={total_f_num/total_n*1000:6.1f} meV/Å")


if __name__ == '__main__':
    main()
