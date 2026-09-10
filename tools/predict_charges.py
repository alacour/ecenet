"""Predict LES latent charges for one frame with a trained ECENet+LES model.

Loads a checkpoint saved by ``scripts/train_ecenet_xyz.py`` with
``use_les=True`` (the top-level ``les`` dict is required), runs the model's
``l0`` read-out on one frame of an ASE-readable file, and prints the per-atom
latent charges plus the long-range energy. If the frame carries reference
charges (a ``q`` column in extxyz, e.g. electrolyte.xyz), it also reports
MAE and correlation against them.

Sign note: the LES energy is quadratic in the latent charges, so a global
sign flip of ALL charges leaves every prediction unchanged — which sign the
model converges to is chance. The comparison therefore also reports the
sign-aligned numbers (flip applied if it improves the correlation).

Usage (from the repo root):
    python tools/predict_charges.py --checkpoint electrolyte_les.mdl \
        --data ../imports/data/electrolyte.xyz --frame 0
    python tools/predict_charges.py ... --save charges.npz   # positions+q to npz
    python tools/predict_charges.py ... --frame 0:1000 --save traj.npz
        # RANGE mode (ASE slice syntax: ':', '100:200', '::10'): model loaded
        # once, arrays stacked per frame — charges (n_frames, N), dipoles
        # (n_frames, N, 3) — for e.g. IR from an MD trajectory
"""

import argparse
import os
import sys  # repo root + scripts/ on path (run from anywhere)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

import numpy as np
import torch


def load_les_model(checkpoint_path, device):
    """Rebuild (model, les_module, hparams, element_to_type, dtype).

    Uses the best-val weights when present. The returned ``les_module`` is
    ready to use — materialised and loaded via ``ecenet.les.load_les_module``
    (which owns the lazy-upstream-head ordering trap) — and ``hparams`` is
    the checkpoint's dict so callers need no second ``torch.load``. The l0
    convention comes off the model: ``model.les_flags``.
    """
    from ecenet import ECENet
    from ecenet.les import load_les_module

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'les' not in ckpt:
        raise ValueError(
            f"{checkpoint_path} carries no 'les' dict — this tool is for "
            "checkpoints trained with use_les=True.")

    hparams = dict(ckpt['hparams'])
    hp = dict(hparams)
    n_mp = hp.pop('n_mp', 1)
    state = ckpt.get('best_state') or ckpt['model']
    dtype = next(iter(state.values())).dtype

    model = ECENet(**hp, n_mp=n_mp)
    if dtype == torch.float64:
        model = model.double()
    model = model.to(device)
    model.load_state_dict(state)
    model.eval()

    les_module = load_les_module(ckpt['les'], model, device, dtype)
    return model, les_module, hparams, ckpt['element_to_type'], dtype


def _predict_frame(model, les_module, hp, elem_to_type, dtype, atoms, device):
    """Latent charges (+ dipoles) for one ASE Atoms with an already-loaded model."""
    from train_ecenet_mptrj import build_topology

    symbols = atoms.get_chemical_symbols()
    missing = sorted({s for s in symbols if s not in elem_to_type})
    if missing:
        raise ValueError(f"Frame contains element(s) {missing} the checkpoint "
                         f"was not trained on (knows: {sorted(elem_to_type)}).")
    types = torch.tensor([elem_to_type[s] for s in symbols],
                         dtype=torch.long, device=device)

    periodic = bool(atoms.pbc.any()) and bool(atoms.cell.any())
    cell_np = np.asarray(atoms.get_cell(), dtype=np.float64) if periodic else None
    pos = torch.tensor(atoms.get_positions(), dtype=dtype, device=device)
    cell = (torch.tensor(cell_np, dtype=dtype, device=device)
            if periodic else None)

    ei, ej, she, ni, nj, shn = build_topology(
        atoms.get_positions(), cell_np, periodic,
        hp['r_cut_edge'], hp['r_cut_neighbor'], device, dtype)

    with torch.no_grad():
        _, l0 = model.forward_pbc(pos, types, ei, ej, she, ni, nj, shn,
                                  return_embeddings=True, l0_only=True)
        e_lr, q = les_module(l0, pos, cell=cell, return_charges=True,
                             **model.les_flags)

    out = {
        'symbols': np.array(symbols),
        'positions': atoms.get_positions(),
        'charges': q.cpu().numpy().reshape(-1),
        'e_lr': float(e_lr.sum()),
    }
    if model.les_dipole or model.les_alpha:
        from ecenet.les import unpack_l0
        _, u, alpha = unpack_l0(l0, **model.les_flags)
        if u is not None:
            out['dipoles'] = u.cpu().numpy()
        if alpha is not None:          # (N,) 'iso' or (N, 3, 3) 'aniso'
            out['alphas'] = alpha.cpu().numpy()
    if 'q' in atoms.arrays:
        out['charges_ref'] = np.asarray(atoms.arrays['q'], dtype=np.float64)
    return out


def predict_charges(checkpoint_path, data_path, frame=0, device='cpu'):
    """Latent charges + E_lr for one frame. Returns a dict of numpy arrays."""
    from ase.io import read

    device = torch.device(device)
    model, les_module, hp, elem_to_type, dtype = load_les_model(
        checkpoint_path, device)
    atoms = read(data_path, index=frame)
    return _predict_frame(model, les_module, hp, elem_to_type, dtype,
                          atoms, device)


def predict_charges_frames(checkpoint_path, data_path, index=':', device='cpu',
                           verbose=False):
    """Latent charges (+ dipoles) for a RANGE of frames.

    ``index`` uses ASE slice syntax: ``':'`` (all), ``'0:1000'``, ``'::10'``.
    The model is loaded once and reused. When every frame shares one
    composition (an MD trajectory) the per-frame arrays are stacked —
    charges (n_frames, N), dipoles (n_frames, N, 3), ... — otherwise they
    come back as object arrays of per-frame results (np.load needs
    allow_pickle=True for those).
    """
    from ase.io import read

    device = torch.device(device)
    model, les_module, hp, elem_to_type, dtype = load_les_model(
        checkpoint_path, device)
    frames = read(data_path, index=index)
    if not isinstance(frames, list):
        frames = [frames]

    per = []
    for k, atoms in enumerate(frames):
        per.append(_predict_frame(model, les_module, hp, elem_to_type, dtype,
                                  atoms, device))
        if verbose and ((k + 1) % 50 == 0 or k + 1 == len(frames)):
            print(f"  frame {k + 1}/{len(frames)}", flush=True)

    out = {'e_lr': np.array([r['e_lr'] for r in per])}
    keys = [k for k in ('positions', 'charges', 'dipoles', 'charges_ref')
            if all(k in r for r in per)]
    if all(np.array_equal(r['symbols'], per[0]['symbols']) for r in per):
        out['symbols'] = per[0]['symbols']
        for k in keys:
            out[k] = np.stack([r[k] for r in per])
    else:
        def _ragged(vals):
            a = np.empty(len(vals), dtype=object)
            a[:] = vals
            return a
        out['symbols'] = _ragged([r['symbols'] for r in per])
        for k in keys:
            out[k] = _ragged([r[k] for r in per])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', required=True,
                    help='.mdl from train_ecenet_xyz(use_les=True)')
    ap.add_argument('--data', required=True, help='ASE-readable file (extxyz, ...)')
    ap.add_argument('--frame', default='0',
                    help="frame index (default 0), or an ASE-style range: "
                         "':' (all), '0:1000', '::10'")
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--save', default=None, help='write results to this .npz')
    ap.add_argument('--per_atom', action='store_true',
                    help='print every atom, not just the summary (single frame only)')
    args = ap.parse_args()

    try:
        frame = int(args.frame)
    except ValueError:
        r = predict_charges_frames(args.checkpoint, args.data, args.frame,
                                   args.device, verbose=True)
        q, e_lr = r['charges'], r['e_lr']
        n = len(e_lr)
        print(f"Frames {args.frame!r}: {n} frames | "
              f"E_lr mean = {e_lr.mean():+.6f} eV (min {e_lr.min():+.6f}, "
              f"max {e_lr.max():+.6f})")
        if q.dtype != object:
            print(f"Latent charges: per-frame sum mean = {q.sum(1).mean():+.4f}, "
                  f"min = {q.min():+.4f}, max = {q.max():+.4f}"
                  + (" | dipoles saved too" if 'dipoles' in r else ""))
        else:
            print("(mixed compositions — object arrays; np.load with "
                  "allow_pickle=True)")
        if args.per_atom:
            print("--per_atom applies to a single frame only; ignored.")
        if args.save:
            np.savez(args.save, **r)
            print(f"Saved to {args.save}")
        return

    r = predict_charges(args.checkpoint, args.data, frame, args.device)
    q = r['charges']

    print(f"Frame {args.frame}: {len(q)} atoms | E_lr = {r['e_lr']:+.6f} eV")
    print(f"Latent charges: sum = {q.sum():+.4f}, "
          f"min = {q.min():+.4f}, max = {q.max():+.4f}")
    if 'dipoles' in r:
        u = r['dipoles']
        mu = (q - q.mean())[:, None] * r['positions']
        mu = mu.sum(0) + u.sum(0)
        print(f"Latent dipoles: |u| mean = {np.linalg.norm(u, axis=1).mean():.4f}, "
              f"max = {np.linalg.norm(u, axis=1).max():.4f} e·Å | "
              f"molecular μ = [{mu[0]:+.4f} {mu[1]:+.4f} {mu[2]:+.4f}]")
    for s in sorted(set(r['symbols'])):
        qs = q[r['symbols'] == s]
        print(f"  {s:2s}: mean {qs.mean():+.4f}  std {qs.std():.4f}  (n={len(qs)})")

    if 'charges_ref' in r:
        ref = r['charges_ref']
        corr = float(np.corrcoef(q, ref)[0, 1])
        sign = -1.0 if corr < 0 else 1.0    # global sign is not identifiable
        mae = np.abs(sign * q - ref).mean()
        print(f"vs reference q: corr = {corr:+.4f} "
              f"(sign-aligned: {sign * corr:.4f}), sign-aligned MAE = {mae:.4f}")

    if args.per_atom:
        ref = r.get('charges_ref')
        for i, (s, qi) in enumerate(zip(r['symbols'], q)):
            extra = f"   ref {ref[i]:+.4f}" if ref is not None else ""
            print(f"  {i:4d} {s:2s} {qi:+.5f}{extra}")

    if args.save:
        np.savez(args.save, **r)
        print(f"Saved to {args.save}")


if __name__ == '__main__':
    main()
