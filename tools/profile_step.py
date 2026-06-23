"""Profile individual components of a single MLIP calculator step.

Usage (run from the repo root):
    python tools/profile_step.py --checkpoint spice.mdl --box water_box.xyz
    python tools/profile_step.py --checkpoint spice.mdl --box water_box.xyz --float32
"""
import argparse
import os
import sys
import time

import torch
from ase.io import read
from ase.neighborlist import neighbor_list as ase_neighbor_list

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root → import ecenet
from ecenet.calculator import ECENetCalculator
from ecenet.radial import radial_basis
from ecenet.spherical import build_D_block_from_list, recursive_wigner_D, wigner_rotate

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--box',        required=True)
parser.add_argument('--frame_idx',  type=int, default=-1)
parser.add_argument('--float32',    action='store_true')
parser.add_argument('--tf32',       action='store_true',
                    help='Enable TF32 for float32 matmuls (Ampere+ GPUs). No effect '
                         'unless --float32 (TF32 is a float32-only mode).')
parser.add_argument('--n_warmup',   type=int, default=3)
parser.add_argument('--n_time',     type=int, default=10)
args = parser.parse_args()

dtype  = torch.float32 if args.float32 else torch.float64
if args.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    if dtype == torch.float64:
        print("[tf32] requested but dtype=float64 → no effect (TF32 is float32-only); "
              "add --float32 to use it")
atoms  = read(args.box, index=args.frame_idx)
atoms.set_pbc(True)

calc   = ECENetCalculator.from_checkpoint(args.checkpoint, dtype=dtype)
model  = calc.model
device = calc.device

# ── Utilities ─────────────────────────────────────────────────────────────────

def sync():
    if device.type == 'cuda':
        torch.cuda.synchronize()

def time_fn(label, fn, n=args.n_time, indent=2):
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        result = fn()
    sync()
    ms = (time.perf_counter() - t0) / n * 1000
    pad = ' ' * indent
    print(f"{pad}{label:<44s} {ms:8.2f} ms")
    return ms, result

def sep(title=''):
    if title:
        print(f"\n── {title} {'─'*(54-len(title))}")
    else:
        print()

# ── Warm up ───────────────────────────────────────────────────────────────────
print(f"Warming up ({args.n_warmup} steps)...")
atoms.calc = calc
for _ in range(args.n_warmup):
    atoms.get_potential_energy()
    atoms.get_forces()

# ── Setup inputs ──────────────────────────────────────────────────────────────
symbols      = atoms.get_chemical_symbols()
positions_np = atoms.get_positions()
cell         = atoms.get_cell().array

types = torch.tensor([calc.element_to_type[s] for s in symbols],
                     dtype=torch.long, device=device)
pos_d = torch.tensor(positions_np, dtype=dtype, device=device)

print(f"\nSystem: {len(atoms)} atoms, "
      f"cell={atoms.cell.lengths().round(2)}, dtype={dtype}\n")

# ── 1. Neighbor list ──────────────────────────────────────────────────────────
sep("Neighbor list")
time_fn("ASE (CPU)",
        lambda: (ase_neighbor_list('ijS', atoms, model.r_cut_edge),
                 ase_neighbor_list('ijS', atoms, model.r_cut_neighbor)))
time_fn("GPU O(N²)",
        lambda: (calc._gpu_neighbor_list(pos_d, cell, model.r_cut_edge),
                 calc._gpu_neighbor_list(pos_d, cell, model.r_cut_neighbor)))

edge_i, edge_j, shift_e = calc._gpu_neighbor_list(pos_d, cell, model.r_cut_edge)
nb_src, nb_dst, shift_n  = calc._gpu_neighbor_list(pos_d, cell, model.r_cut_neighbor)

print(f"  edges: {len(edge_i):,}   neighbors: {len(nb_src):,}")

# Fresh pos with grad for all model timing
def fresh_pos():
    return pos_d.clone().requires_grad_(True)

p = fresh_pos()

# ── 2. Forward sub-components ─────────────────────────────────────────────────
sep("Forward pass sub-components")

# Geometry
diff_ij = pos_d[edge_j] - pos_d[edge_i] + shift_e
dist_ij = torch.sqrt((diff_ij ** 2).sum(-1) + 1e-30)
r_hat   = diff_ij / dist_ij.unsqueeze(-1)

time_fn("edge geometry (diff, dist, r_hat)",
        lambda: (pos_d[edge_j] - pos_d[edge_i] + shift_e,))

# ACE basis
time_fn("ACE basis (_compute_ace_basis)",
        lambda: model._compute_ace_basis(
            pos_d.unsqueeze(0), nb_src, nb_dst, types, shift_vecs_nb=shift_n))

A_batch = model._compute_ace_basis(
    pos_d.unsqueeze(0), nb_src, nb_dst, types, shift_vecs_nb=shift_n)
A = A_batch.squeeze(0)

# Embed
time_fn("embed (einsum A×W → A_emb)",
        lambda: model._embed(A, types))

A_emb = model._embed(A, types)
A_both = torch.cat([A_emb[edge_i], A_emb[edge_j]], dim=1)

# Wigner rotation breakdown
_, D_list = time_fn("  Wigner D (recursive_wigner_D)",
        lambda: recursive_wigner_D(r_hat, model.l_max))
_, D_block_main = time_fn("  Wigner D (build_D_block_from_list)",
        lambda: build_D_block_from_list(D_list, len(r_hat), model.l_max,
                                        r_hat.device, r_hat.dtype))
time_fn("  Wigner bmm (A_both @ D)",
        lambda: torch.bmm(A_both, D_block_main))
time_fn("Wigner rotate total (main, w/ cached D)",
        lambda: wigner_rotate(A_both, D_block_main))

A_rot = wigner_rotate(A_both, D_block_main)

# sph_to_angular
time_fn("sph_to_angular (repeat_interleave + gather)",
        lambda: model.sph_to_angular(A_rot))

A_cos, A_sin = model.sph_to_angular(A_rot)

type_i = types[edge_i]
type_j = types[edge_j]

# Equivariant layers + message passing
if hasattr(model, 'layers') and hasattr(model, 'mp_layers'):
    for step_idx, (layer_group, mp) in enumerate(
            zip(model.layers, model.mp_layers)):
        sep(f"MP step {step_idx+1}")
        for li, layer in enumerate(layer_group):
            time_fn(f"  EquivariantLinear [{step_idx},{li}]",
                    lambda l=layer: l.linear(A_cos, A_sin, type_i=type_i, type_j=type_j))
            if layer.use_nonlinearity:
                A_cos_l, A_sin_l = layer.linear(A_cos, A_sin, type_i=type_i, type_j=type_j)
                time_fn(f"  RealSpaceNonlinearity [{step_idx},{li}]",
                        lambda l=layer, c=A_cos_l, s=A_sin_l:
                            l.nonlin(c, s, type_i=type_i, type_j=type_j))
            A_cos, A_sin = layer(A_cos, A_sin, type_i=type_i, type_j=type_j)

        # Message passing internals
        n_e = len(edge_i)
        lp1 = model.l_max + 1
        Ac = A_cos.view(n_e, mp.n_base, lp1, mp.n_angular)
        As = A_sin.view(n_e, mp.n_base, lp1, mp.n_angular)

        def _pack():
            h = torch.zeros(n_e, mp.n_base, mp.n_sph, device=device, dtype=dtype)
            for l in range(lp1):
                m_out = min(l, mp.m_max)
                h[:, :, l*l+l : l*l+l+m_out+1] = Ac[:, :, l, :m_out+1]
                if m_out > 0:
                    h[:, :, l*l+l-m_out : l*l+l] = As[:, :, l, 1:m_out+1].flip(-1)
            return h
        _, h_packed = time_fn("  MP pack cos/sin → n_sph", _pack)

        f_d = radial_basis(dist_ij, mp.r_cut, mp.n_dist_basis, cutoff_type=mp.cutoff_type)
        ti, tj = types[edge_i], types[edge_j]
        w = torch.einsum('en,enc->ec', f_d, mp.W_msg[ti, tj])   # plain n_types²
        D_block_main_T = D_block_main.transpose(-1, -2).contiguous()
        time_fn("  MP bmm unrotate (h @ D^T, contiguous)",
                lambda: torch.bmm(h_packed, D_block_main_T))
        h_global = torch.bmm(h_packed, D_block_main_T) * w.unsqueeze(-1)

        idx = edge_j[:, None, None].expand_as(h_global)
        Delta = torch.zeros(len(types), mp.n_base, mp.n_sph,
                            device=device, dtype=dtype).scatter_add(0, idx, h_global)
        time_fn("  MP scatter_add (aggregate to atoms)",
                lambda: torch.zeros(len(types), mp.n_base, mp.n_sph,
                                    device=device, dtype=dtype
                                    ).scatter_add(0, idx, h_global))

        time_fn("  MP bmm rotate back (Delta @ D)",
                lambda: torch.bmm(Delta[edge_i], D_block_main))

        A_cos, A_sin = mp(A_cos, A_sin, r_hat, dist_ij, edge_i, edge_j,
                          len(types), types[edge_i], types[edge_j],
                          False, D_block=D_block_main)

sep("Output")
invariants = model._contract(A_cos, A_sin)
if model.n_max_d is not None:
    rij_basis = radial_basis(dist_ij, model.r_cut_edge, model.n_max_d,
                             cutoff_type=model.cutoff_type)
    time_fn("output MLP",
            lambda: model.output_net(invariants, type_i, type_j))
    mlp_out = model.output_net(invariants, type_i, type_j)
    time_fn("dot(MLP_out × rij_basis)",
            lambda: (mlp_out * rij_basis).sum(-1))
else:
    time_fn("output MLP",
            lambda: model.output_net(invariants, type_i, type_j))

# ── 3. Total forward ──────────────────────────────────────────────────────────
sep("Totals")
time_fn("forward_pbc (full)",
        lambda: model.forward_pbc(fresh_pos(), types, edge_i, edge_j,
                                  shift_e, nb_src, nb_dst, shift_n))

def fwd_forces():
    p = fresh_pos()
    with torch.enable_grad():
        e = model.forward_pbc(p, types, edge_i, edge_j, shift_e,
                              nb_src, nb_dst, shift_n)
        return torch.autograd.grad(e, p)[0]

time_fn("forward_pbc + autograd.grad (forces)", fwd_forces)
print()
