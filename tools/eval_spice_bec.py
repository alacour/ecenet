"""Zero-shot Born-effective-charge (BEC) comparison on the SPICE test slices.

The Born effective charge Z*_i = dP/dr_i is the polarization response to
displacing atom i. With latent charges q(r) (and, with ``les_dipole``,
latent dipoles u(r)) predicted by the model, P = sum_j q_j r_j + sum_j u_j
and autograd delivers Z* INCLUDING the charge-flow terms sum_j r_j dq_j/dr_i
— the part a static-charge picture misses, and the reason BECs are a real
test of whether the latent charges learned charge physics. Nothing here was
ever trained on: energies and forces only, so this is the same zero-shot
validation MACELES-OFF ran.

Upstream's own BEC module does the differentiation (dephasing, mean-charge
removal, and the sqrt(epsilon_infty) normalisation exactly as configured in
the checkpoint's les_arguments); this tool just puts q and u on the
positions' autograd graph and compares against the per-atom ``bec:R:9``
reference column (wB97m-D3(BJ)/Def2SVP) in the les_fit test slices — same
files and download location as tools/eval_spice_dipoles.py.

Sign note: as with the dipoles, the global (q, u) sign is not identifiable
from energy/force training; one sign is chosen for the whole checkpoint
(best least-squares alignment over every BEC component of every atom) and
reported.

Usage (from the repo root):
    python tools/eval_spice_bec.py --checkpoint spice_les.mdl \
        --data_dir ~/Research/imports/data/spice-test-bec
    python tools/eval_spice_bec.py ... --files water --max_frames 20 \
        --save bec.npz
"""

import argparse
import glob
import os
import sys  # repo root + scripts/ + tools/ on path (run from anywhere)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch


def predict_becs(checkpoint_path, xyz_files, device='cpu', max_frames=None):
    """Predicted vs reference per-atom BEC tensors for every usable frame.

    Returns (records, skipped): records have 'subset', 'frame', 'symbols',
    'bec_pred' (N, 3, 3) carrying the model's arbitrary global sign, and
    'bec_ref' (N, 3, 3).
    """
    import itertools

    from ase.io import iread
    from predict_charges import load_les_model
    from train_ecenet_mptrj import build_topology

    device = torch.device(device)
    model, les_module, hp, elem_to_type, dtype = load_les_model(
        checkpoint_path, device)

    records, skipped = [], {}
    for path in xyz_files:
        subset = os.path.splitext(os.path.basename(path))[0]
        # lazy: --max_frames stops reading after N instead of parsing the file
        frames = iread(path)
        if max_frames is not None:
            frames = itertools.islice(frames, max_frames)
        skipped[subset] = 0
        for fi, atoms in enumerate(frames):
            symbols = atoms.get_chemical_symbols()
            if ('bec' not in atoms.arrays
                    or any(s not in elem_to_type for s in symbols)):
                skipped[subset] += 1
                continue
            types = torch.tensor([elem_to_type[s] for s in symbols],
                                 dtype=torch.long, device=device)
            pos_np = atoms.get_positions()
            # topology from detached positions (frozen, standard); q/u stay
            # functions of pos through the model, so grad(P, pos) carries
            # the charge-flow terms
            ei, ej, she, ni, nj, shn = build_topology(
                pos_np, None, False,
                hp['r_cut_edge'], hp['r_cut_neighbor'], device, dtype)
            pos = torch.tensor(pos_np, dtype=dtype, device=device,
                               requires_grad=True)
            _, l0 = model.forward_pbc(pos, types, ei, ej, she, ni, nj, shn,
                                      return_embeddings=True, l0_only=True)
            # same implementation as ECENetLESCalculator.compute_bec (with
            # les_alpha the induced dipoles are part of the polarization)
            bec = les_module.born_charges(l0, pos, cell=None,
                                          **model.les_flags)
            records.append({
                'subset': subset, 'frame': fi, 'symbols': np.array(symbols),
                'bec_pred': bec.detach().cpu().numpy(),
                'bec_ref': np.asarray(atoms.arrays['bec'],
                                      dtype=np.float64).reshape(-1, 3, 3),
            })
    return records, skipped


def summarise(records, skipped):
    """Global sign alignment + per-subset metric table. Returns the sign."""
    pred = np.concatenate([r['bec_pred'].reshape(-1, 9) for r in records])
    ref = np.concatenate([r['bec_ref'].reshape(-1, 9) for r in records])
    sign = 1.0 if float((pred * ref).sum()) >= 0 else -1.0
    pred = sign * pred
    subset_per_atom = np.concatenate([
        np.full(len(r['bec_pred']), r['subset']) for r in records])

    diag = np.array([0, 4, 8])
    off = np.array([1, 2, 3, 5, 6, 7])

    def row(p, r):
        mae = np.abs(p - r).mean()
        mae_d = np.abs(p[:, diag] - r[:, diag]).mean()
        mae_o = np.abs(p[:, off] - r[:, off]).mean()
        corr = float(np.corrcoef(p.reshape(-1), r.reshape(-1))[0, 1])
        scale = float((p * r).sum() / np.maximum((r * r).sum(), 1e-12))
        return mae, mae_d, mae_o, corr, scale

    print(f"\nGlobal charge sign: {'+' if sign > 0 else '-'} "
          "(aligned on all BEC components; not identifiable from training)")
    print(f"{'subset':22s} {'atoms':>7s} {'skip':>4s} {'MAE':>8s} "
          f"{'MAE_diag':>9s} {'MAE_off':>8s} {'corr':>7s} {'scale':>7s}   [e]")
    subsets = sorted({r['subset'] for r in records})
    for s in subsets:
        m = subset_per_atom == s
        n_frames = sum(1 for r in records if r['subset'] == s)
        mae, mae_d, mae_o, corr, scale = row(pred[m], ref[m])
        print(f"{s:22s} {int(m.sum()):7d} {skipped.get(s, 0):4d} {mae:8.4f} "
              f"{mae_d:9.4f} {mae_o:8.4f} {corr:7.3f} {scale:7.3f}"
              f"   ({n_frames} frames)")
    for s in sorted(set(skipped) - set(subsets)):
        print(f"{s:22s} {0:7d} {skipped[s]:4d}   (all frames skipped — "
              "unknown elements or no reference)")
    mae, mae_d, mae_o, corr, scale = row(pred, ref)
    print(f"{'ALL':22s} {len(pred):7d} {sum(skipped.values()):4d} {mae:8.4f} "
          f"{mae_d:9.4f} {mae_o:8.4f} {corr:7.3f} {scale:7.3f}")
    # acoustic sum rule: for a neutral system sum_i Z*_i = 0 exactly in the
    # reference; how close the prediction comes is a free consistency check
    asr = np.array([np.abs(sign * r['bec_pred'].sum(axis=0)).max()
                    for r in records])
    print(f"acoustic sum rule |sum_i Z*_i|_max: mean {asr.mean():.4f}, "
          f"worst {asr.max():.4f} e")
    return sign


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', required=True,
                    help='.mdl trained with use_les=True (xyz or SPICE trainer)')
    ap.add_argument('--data_dir', required=True,
                    help='directory with the les_fit data-spice-test-bec .xyz '
                         'files (see module docstring)')
    ap.add_argument('--files', nargs='+', default=None,
                    help='subset names to evaluate (default: every .xyz found)')
    ap.add_argument('--max_frames', type=int, default=None,
                    help='cap frames per file (smoke runs)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--tf32', action='store_true',
                    help='enable TF32 matmuls (CUDA + float32 checkpoints only) '
                         '— evaluate at the same precision the model is '
                         'benchmarked/deployed at')
    ap.add_argument('--save', default=None,
                    help='write per-atom sign-aligned results to this file: '
                         '.csv (one row per atom: subset, frame, symbol, 9 '
                         'pred + 9 ref components) or .npz (arrays)')
    args = ap.parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        if not args.device.startswith('cuda'):
            print('[tf32] requested but device is not CUDA — no effect')

    data_dir = os.path.expanduser(args.data_dir)
    if args.files:
        xyz_files = [os.path.join(data_dir, f if f.endswith('.xyz')
                                  else f + '.xyz') for f in args.files]
        missing = [f for f in xyz_files if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"missing: {missing}")
    else:
        xyz_files = sorted(glob.glob(os.path.join(data_dir, '*.xyz')))
        if not xyz_files:
            raise FileNotFoundError(f"no .xyz files in {data_dir}")

    records, skipped = predict_becs(args.checkpoint, xyz_files,
                                    device=args.device,
                                    max_frames=args.max_frames)
    if not records:
        raise SystemExit("no usable frames (unknown elements everywhere? "
                         f"skipped: {skipped})")
    sign = summarise(records, skipped)

    if args.save:
        subset = np.concatenate([np.full(len(r['bec_pred']), r['subset'])
                                 for r in records])
        frame = np.concatenate([np.full(len(r['bec_pred']), r['frame'])
                                for r in records])
        symbol = np.concatenate([r['symbols'] for r in records])
        bec_pred = sign * np.concatenate([r['bec_pred'].reshape(-1, 9)
                                          for r in records])
        bec_ref = np.concatenate([r['bec_ref'].reshape(-1, 9)
                                  for r in records])
        if args.save.endswith('.csv'):
            comp = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
            with open(args.save, 'w') as fh:
                fh.write('subset,frame,symbol,'
                         + ','.join(f'pred_{c}' for c in comp) + ','
                         + ','.join(f'ref_{c}' for c in comp) + '\n')
                for i in range(len(subset)):
                    fh.write(f"{subset[i]},{frame[i]},{symbol[i]},"
                             + ','.join(f'{v:.6f}' for v in bec_pred[i]) + ','
                             + ','.join(f'{v:.6f}' for v in bec_ref[i]) + '\n')
        else:
            np.savez(args.save, subset=subset, frame=frame, symbol=symbol,
                     bec_pred=bec_pred, bec_ref=bec_ref)
        print(f"Saved to {args.save}")


if __name__ == '__main__':
    main()
