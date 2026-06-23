#!/usr/bin/env python
"""Numerical equivalence check: working-tree ECENet vs a git reference.

Proves a refactor is behaviour-preserving by giving the working-tree model and a
model built from another git ref **identical weights** (state_dict transfer) and
**identical inputs**, then diffing energy / forces / training-batch outputs. Same
weights + same maths => agreement to ~1e-12 (typically exactly 0.0).

It runs the reference code in a SUBPROCESS whose cwd is a throwaway worktree of
the ref, so the ref's ``import ecenet`` (or the pre-rename ``import features``)
resolves to the ref's code — sidestepping the "two packages both named ecenet in
one process" collision. The reference API is auto-detected:
  * new   : ``from ecenet import ECENet``     (ECENet(n_mp=...))
  * legacy: ``from features.equece import EquECE, EquECEMP``

Message-passing convention (working tree): the model is ``n_mp`` stages of
``n_layers`` equivariant layers with one MP layer between consecutive stages
(``n_mp-1`` MP layers, no trailing MP). ``n_mp=1`` is the plain model. Against a
legacy ref this maps to ``EquECEMP(n_mp_steps=n_mp-1, n_layers_per_mp=n_layers,
n_final_layers=n_layers)`` and the new ``layers.{n_mp-1}`` stage is remapped onto
the legacy ``final_layers`` for the weight transfer.

Usage:
    python tools/equiv_vs_ref.py                 # compare against main
    python tools/equiv_vs_ref.py --ref <sha>     # against any commit/branch
    python tools/equiv_vs_ref.py --tol 1e-10 --keep-worktree

Exit code is non-zero if any config diverges beyond --tol (CI-friendly).
"""
import argparse
import os
import subprocess
import sys
import tempfile

import torch

DT = torch.float64

# Base architecture shared by every config; each config layers extra kwargs on top.
BASE = dict(n_types=4, r_cut_edge=5.0, r_cut_neighbor=4.0, l_max=2, n_max=3,
            embed_dim=8, n_layers=2, n_max_d=4)

# (name, extra_kwargs, n_mp).  n_mp = number of equivariant-layer stages
# (n_mp-1 MP layers between them); n_mp=1 is the plain, no-MP model.
CONFIGS = [
    ("base, no MP (n_mp=1)",     dict(), 1),
    ("base, MP (n_mp=2)",        dict(), 2),
    ("base, MP (n_mp=3)",        dict(), 3),
    ("edge_type_linear (typed)", dict(edge_type_linear=True, edge_type_output=True), 1),
]


def have_new_api():
    try:
        import ecenet  # noqa: F401
        return True
    except ImportError:
        return False


def rand_struct(n, seed):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, 3, generator=g, dtype=DT) * 1.9
    typ = torch.randint(0, 4, (n,), generator=g)
    return pos, typ


def make_inputs():
    return {'single': rand_struct(7, 3), 'batch': [rand_struct(6, 11), rand_struct(8, 12)]}


def build_model(extra, n_mp):
    """Build from whichever API is importable.

    New API uses ECENet(n_mp=...). Legacy API maps n_mp -> EquECE (n_mp==1) or
    EquECEMP(n_mp_steps=n_mp-1, n_layers_per_mp=n_layers, n_final_layers=n_layers).
    """
    kw = {**BASE, **extra}
    L = kw['n_layers']
    torch.manual_seed(1)
    if have_new_api():
        from ecenet import ECENet
        model = ECENet(n_mp=n_mp, **kw)
    else:
        from features.equece import EquECE, EquECEMP
        if n_mp == 1:
            model = EquECE(**kw)
        else:
            model = EquECEMP(n_mp_steps=n_mp - 1, n_layers_per_mp=L,
                             n_final_layers=L, **kw)
    return model.to(DT)


def remap_new_to_legacy(sd, n_mp):
    """Rename new-API keys to legacy: the last stage `layers.{n_mp-1}.*` becomes
    the legacy `final_layers.*` block. Other keys are unchanged."""
    if n_mp < 2:
        return sd
    pref = f'layers.{n_mp - 1}.'
    out = {}
    for k, v in sd.items():
        out['final_layers.' + k[len(pref):] if k.startswith(pref) else k] = v
    return out


def outputs_for(model, inputs):
    model.eval()
    pos, typ = inputs['single']
    p = pos.clone().requires_grad_(True)
    E = model(p, typ)
    F = torch.autograd.grad(E, p)[0]
    B = model.forward_batch_multi([s for s, _ in inputs['batch']],
                                  [t for _, t in inputs['batch']])
    return {'E': E.detach(), 'F': F.detach(), 'B': B.detach()}


# ── Reference role: runs inside the ref worktree, fed weights by the driver ──
def run_ref_role(workdir):
    sys.path.insert(0, os.getcwd())   # ref worktree's packages (cwd not auto-added)
    legacy = not have_new_api()
    inputs = torch.load(os.path.join(workdir, 'inputs.pt'))
    ref_out, meta = [], []
    for i, (name, extra, n_mp) in enumerate(CONFIGS):
        model = build_model(extra, n_mp)
        new_sd = torch.load(os.path.join(workdir, f'sd_{i}.pt'))
        if legacy:
            new_sd = remap_new_to_legacy(new_sd, n_mp)
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        meta.append({'keys_match': set(model.state_dict()) == set(new_sd),
                     'missing': len(missing), 'unexpected': len(unexpected)})
        ref_out.append(outputs_for(model, inputs))
    torch.save({'out': ref_out, 'meta': meta}, os.path.join(workdir, 'ref_out.pt'))


# ── Driver role: working tree. Owns inputs/weights, spawns the ref subprocess ──
def run_driver(args):
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    workdir = tempfile.mkdtemp(prefix='ecenet_equiv_')
    wt = os.path.join(workdir, 'ref_worktree')

    print(f"ref={args.ref}  tol={args.tol}")
    subprocess.run(['git', '-C', repo, 'worktree', 'add', '--detach', wt, args.ref],
                   check=True, capture_output=True)
    try:
        inputs = make_inputs()
        torch.save(inputs, os.path.join(workdir, 'inputs.pt'))
        new_out = []
        for i, (name, extra, n_mp) in enumerate(CONFIGS):
            model = build_model(extra, n_mp)            # working-tree ecenet
            torch.save(model.state_dict(), os.path.join(workdir, f'sd_{i}.pt'))
            new_out.append(outputs_for(model, inputs))

        env = dict(os.environ); env.pop('PYTHONPATH', None)
        subprocess.run([sys.executable, os.path.abspath(__file__),
                        '--role', 'ref', '--workdir', workdir],
                       check=True, cwd=wt, env=env)

        ref = torch.load(os.path.join(workdir, 'ref_out.pt'))
        ref_out, meta = ref['out'], ref['meta']
        print(f"\n{'config':28} {'keys':5} {'|dE|':>10} {'|dF|':>10} {'|dBatch|':>10}")
        all_ok = True
        for (name, *_), no, ro, m in zip(CONFIGS, new_out, ref_out, meta):
            dE = (no['E'] - ro['E']).abs().max().item()
            dF = (no['F'] - ro['F']).abs().max().item()
            dB = (no['B'] - ro['B']).abs().max().item()
            ok = (m['keys_match'] and not m['missing'] and not m['unexpected']
                  and dE <= args.tol and dF <= args.tol and dB <= args.tol)
            all_ok &= ok
            print(f"[{'PASS' if ok else 'FAIL'}] {name:22} {'ok' if m['keys_match'] else 'DIFF':5}"
                  f" {dE:10.2e} {dF:10.2e} {dB:10.2e}"
                  + ("" if m['keys_match'] else f"  (miss={m['missing']} unexp={m['unexpected']})"))
        print("\n==>", "ALL EQUIVALENT" if all_ok else "DIVERGENCES FOUND")
        return 0 if all_ok else 1
    finally:
        if not args.keep_worktree:
            subprocess.run(['git', '-C', repo, 'worktree', 'remove', '--force', wt],
                           capture_output=True)
        else:
            print(f"\n(worktree kept at {wt})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref', default='main', help='git ref to compare against (default: main)')
    ap.add_argument('--tol', type=float, default=1e-9, help='max allowed abs difference')
    ap.add_argument('--keep-worktree', action='store_true', help='do not delete the ref worktree')
    ap.add_argument('--role', choices=['driver', 'ref'], default='driver', help=argparse.SUPPRESS)
    ap.add_argument('--workdir', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.role == 'ref':
        run_ref_role(args.workdir)
        return 0
    return run_driver(args)


if __name__ == '__main__':
    sys.exit(main())
