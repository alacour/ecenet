"""Training script for ECENet on the SPICE multi-molecule dataset.

Expected file format: extended XYZ with columns
    species x y z fx fy fz MACE_fx MACE_fy MACE_fz
and comment line containing  energy=<float>  (DFT, eV).

Usage:
    Set hyperparameters in the ``train_ecenet_spice(...)`` call at the bottom of
    this file (or import the function from your own driver), then launch:

        # single process
        python scripts/train_ecenet_spice.py

        # multi-GPU data-parallel (DDP) via torchrun
        torchrun --nproc_per_node=4 scripts/train_ecenet_spice.py

    Every training/model option is a keyword argument of ``train_ecenet_spice``.
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import math
import re
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ecenet import ECENet


class _MultiForwardWrapper(nn.Module):
    """Thin wrapper so DDP intercepts forward_batch_multi for gradient sync.

    With a LES module attached, also computes the long-range energy from the
    per-structure ``l0`` embeddings and returns ``E_sr + E_lr``. Registering
    the LES module HERE (not just in the optimizer) is what puts its
    parameters inside DDP's bucket reduction — and the head runs on every
    step, so no parameter is ever unused (``find_unused_parameters=False``
    stays valid). The LES call is batched: one call over the concatenated
    atoms with a structure-index vector, zero cells → the isolated
    (non-periodic) pairwise path, which is what SPICE molecules are.
    """
    def __init__(self, model, les_module=None):
        super().__init__()
        self.model = model
        self.les = les_module
        # l0 convention (l0_is_charge / les_dipole) read off the model —
        # the single source of truth for how l0 is interpreted.
        self.les_flags = model.les_flags

    def forward(self, positions_list, types_list, topology=None):
        if self.les is None:
            return self.model.forward_batch_multi(positions_list, types_list,
                                                  topology=topology)
        e_sr, l0_list = self.model.forward_batch_multi(
            positions_list, types_list, return_embeddings=True, l0_only=True,
            topology=topology)
        l0 = torch.cat(l0_list, dim=0)
        pos = torch.cat(positions_list, dim=0)
        batch = torch.cat([
            torch.full((p.shape[0],), b, dtype=torch.long, device=p.device)
            for b, p in enumerate(positions_list)])
        return e_sr + self.les(l0, pos, batch=batch,
                               n_struct=len(positions_list),
                               **self.les_flags)  # (B,)


def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def size_aware_batches(all_idx, n_atoms, world_size, rank, epoch, seed,
                       batch_size=None, max_atoms_per_batch=None,
                       max_batch_count=None, bucket_sort=True, verbose=False):
    """Size-grouped, cross-rank-aligned batches for one epoch; returns ``rank``'s.

    Every rank holds the whole dataset, so this is a *global* length-grouped
    sampler: sort the epoch's indices by atom count, form batches from that sorted
    order, group the batches into rounds of ``world_size`` adjacent ones, shuffle
    the round order, and give rank r the r-th batch of each round.

    Two properties fall out, both of which matter under DDP:

    * Batches within a round have similar cost, so per-step work is aligned across
      ranks — no straggler from molecule-size variance (the biggest win
      multi-node).
    * Every rank derives the same assignment from the same data and seed, and only
      full rounds are used, so all ranks run the *same number* of batches. That is
      load-bearing: a mismatch deadlocks the collective in backward rather than
      raising.

    Batches are formed either at fixed ``batch_size`` (size bucketing) or packed to
    a total-atom budget (``max_atoms_per_batch``, optionally capped at
    ``max_batch_count`` structures). The budget keeps memory and compute per step
    roughly uniform, so a batch of several large molecules cannot OOM the run.

    ``bucket_sort`` trades DDP balance against batch diversity:

    * True (default): sort by atom count first. Batches are size-homogeneous and
      per-step cost is tightly aligned across ranks — but the sort undoes the
      epoch's shuffle, so batches come out nearly identical every epoch and the
      largest, rare-size structures are always grouped together. That costs
      gradient diversity exactly where there is least data.
    * False: greedy-pack the already-shuffled order. Batches are random and differ
      every epoch. With an atom budget each batch still holds about
      ``max_atoms_per_batch`` atoms, so per-rank cost stays bounded; only the
      within-round spread loosens. Meaningful mainly with an atom budget — without
      one this is just fixed-size batches plus the round alignment.
    """
    order = (all_idx[np.argsort(n_atoms[all_idx], kind='stable')]
             if bucket_sort else all_idx)
    if max_atoms_per_batch is not None:
        batches, cur, cur_atoms = [], [], 0
        for idx in order:
            a = int(n_atoms[idx])
            if cur and (cur_atoms + a > max_atoms_per_batch
                        or (max_batch_count and len(cur) >= max_batch_count)):
                batches.append(np.array(cur))
                cur, cur_atoms = [], 0
            cur.append(int(idx))
            cur_atoms += a
        if cur:
            batches.append(np.array(cur))
    else:
        batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

    # Whole rounds only, so every rank gets exactly n_rounds batches.
    n_rounds = len(batches) // world_size
    if n_rounds == 0:
        # Degenerate tiny epoch. Still hand every rank exactly one batch so the
        # ranks stay aligned (a deadlock here would be far worse), but they now
        # share structures — worth saying out loud.
        if verbose:
            print_flush(f"  WARNING: only {len(batches)} batches for "
                        f"world_size={world_size}; ranks will share structures "
                        "this epoch. Raise n_per_epoch or lower the batch size.")
        return [batches[rank % len(batches)]]
    # Decorrelated from the epoch's sampling RNG, which is seeded on (seed+epoch);
    # reusing that here would tie the round order to the sample draw.
    round_order = np.random.RandomState(seed + 7919 * (epoch + 1)).permutation(n_rounds)
    return [batches[int(r) * world_size + rank] for r in round_order]


# ---------------------------------------------------------------------------
# Fixed element → type mapping (10 elements in SPICE)
# ---------------------------------------------------------------------------

ELEMENT_TO_TYPE = {
    'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4,
    'P': 5, 'S': 6, 'Cl': 7, 'Br': 8, 'I': 9,
}
N_TYPES = len(ELEMENT_TO_TYPE)
TYPE_NAMES = [k for k, v in sorted(ELEMENT_TO_TYPE.items(), key=lambda x: x[1])]

_ENERGY_RE = re.compile(r'(?<![A-Z_a-z])energy=([-+0-9.eE]+)')


# ---------------------------------------------------------------------------
# Extended XYZ parser
# ---------------------------------------------------------------------------

def parse_xyz_file(path, max_structures=None, dtype=np.float32, verbose=True):
    """Parse an extended XYZ file into a list of structure dicts.

    Each dict contains:
        positions : (N, 3) float array  — Å
        forces    : (N, 3) float array  — eV/Å
        energy    : float               — eV
        types     : (N,)  int16 array   — element type indices (ELEMENT_TO_TYPE)
        n_atoms   : int
    """
    structures = []
    t0 = time.time()
    unknown_elements = set()

    with open(path, 'r') as f:
        while True:
            # --- atom count line ---
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

            # --- comment line ---
            comment = f.readline()
            m = _ENERGY_RE.search(comment)
            energy = float(m.group(1)) if m else 0.0

            # --- atom lines ---
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
                    # consume remaining atom lines and skip structure
                    for _ in range(n_atoms - i - 1):
                        f.readline()
                    break
                types[i]     = ELEMENT_TO_TYPE[elem]
                positions[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
                forces[i]    = [float(parts[4]), float(parts[5]), float(parts[6])]

            if not ok:
                continue

            structures.append({
                'positions': positions,
                'forces':    forces,
                'energy':    energy,
                'types':     types,
                'n_atoms':   n_atoms,
            })

            if max_structures is not None and len(structures) >= max_structures:
                break

            if verbose and len(structures) % 50000 == 0 and len(structures) > 0:
                elapsed = time.time() - t0
                print_flush(f"  Parsed {len(structures):,} structures ({elapsed:.0f}s)...")

    if unknown_elements:
        print_flush(f"  Warning: skipped structures with unknown elements: {unknown_elements}")
    return structures


# ---------------------------------------------------------------------------
# Per-element energy reference (linear regression)
# ---------------------------------------------------------------------------

def compute_energy_reference(structures):
    """Fit per-element reference energies via least squares.

    Returns:
        e_ref: (N_TYPES,) array of reference energies (eV/atom per element type)
    """
    n = len(structures)
    A = np.zeros((n, N_TYPES), dtype=np.float64)
    E = np.zeros(n, dtype=np.float64)
    for i, s in enumerate(structures):
        for t in s['types']:
            A[i, t] += 1
        E[i] = s['energy']
    e_ref, _, _, _ = np.linalg.lstsq(A, E, rcond=None)
    return e_ref


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def to_device_tensors(structures, e_ref, dtype, device):
    """Convert list of structure dicts to lists of tensors on device.

    Subtracts per-element reference energy from each structure's energy.
    """
    positions_list = []
    forces_list    = []
    energies       = []
    types_list     = []

    for s in structures:
        pos = torch.tensor(s['positions'], dtype=dtype, device=device)
        frc = torch.tensor(s['forces'],    dtype=dtype, device=device)
        typ = torch.tensor(s['types'].astype(np.int64), dtype=torch.long, device=device)

        # subtract reference energy
        ref = sum(e_ref[t] for t in s['types'])
        eng = torch.tensor(s['energy'] - ref, dtype=dtype, device=device)

        positions_list.append(pos)
        forces_list.append(frc)
        energies.append(eng)
        types_list.append(typ)

    return positions_list, forces_list, torch.stack(energies), types_list


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_ecenet_spice(
    train_xyz='train_large_neut_no_bad_clean.xyz',
    test_xyz='test_large_neut_all.xyz',
    # Data splits (None = use all available)
    n_train=None,
    n_val=5000,
    n_test=None,
    n_per_epoch=None,   # subsample per epoch (None = full train set)
    cycle_data=False,   # cycle through full dataset in chunks rather than random subsampling
    # Geometry
    r_cut_edge=5.0,
    r_cut_neighbor=4.0,
    l_max=3,
    n_max=4,
    cutoff_type='cosine',
    # Architecture
    embed_dim=32,
    n_layers=2,
    n_max_d=8,
    m_max=None,
    activation='silu',
    use_nonlinearity=True,
    n_grid=None,
    output_hidden_dims=None,
    analytic_ace_basis=True,
    bottleneck_dim=None,
    # Message passing
    n_mp=1,
    mp_type='sum',
    mp_dim=None,
    mp_n_heads=1,
    mp_msg_envelope=True,
    mp_l_attention=False,
    # FiLM gate
    element_film=True,
    film_embed_dim=16,
    film_n_rbf=0,
    film_hidden=None,
    film_per_m=False,
    film_shift=False,
    les_readout=None,     # None -> 'edge_basis' if use_les else 'sum'
    les_charge_scale=1.0,  # fixed multiplier on the edge-mode latent charge (MACELES: 0.1)
    les_dipole=False,      # edge head also emits bond dipoles; l0 packed [q | u]
    les_charges=True,      # False (needs les_dipole): dipoles-only — q hard zero, standard-init dipole head
    les_alpha=None,        # None | 'iso' | 'aniso': edge head also emits polarizabilities α (induced dipoles)
    # Long-range (LES): E = E_sr + E_lr on one autograd graph. Needs the
    # optional `les` package (see ecenet/les.py for install + licensing).
    use_les=False,
    les_arguments=None,    # extra kwargs for upstream les.Les
    # Optimiser
    lr=1e-3,
    weight_decay=1e-5,
    grad_clip=None,
    scheduler_patience=10,      # 'plateau' only
    lr_schedule='plateau',      # 'plateau' | 'cosine' | 'multistep'
    warmup_epochs=0,            # cosine/multistep: linear LR ramp 0→lr
    lr_min_factor=0.0,          # 'cosine' only: LR floor as a fraction of lr
    lr_milestones=None,         # 'multistep': epochs at which lr *= lr_gamma
    lr_gamma=0.1,               # 'multistep': decay factor at each milestone
    early_stopping_patience=None,
    # Training
    n_epochs=100,
    batch_size=8,
    energy_weight=1.0,
    force_weight=1.0,
    loss='mse',
    huber_delta=0.01,
    eval_metric='mae',       # 'mae' | 'rmse' — reported metrics AND the best-val
                             # selection (the training loss itself is unchanged)
    eval_every=1,
    eval_batch_size=32,
    seed=42,
    dtype=torch.float64,
    tf32=False,              # route float32 matmuls to TF32 tensor cores (Ampere+)
    precompute_topology=False,  # build neighbour lists once at startup (fixed
                             # training positions) → skip the per-step nonzero syncs
    # Batching
    bucket=False,              # size-bucketed, cross-rank-aligned batching (DDP load balance)
    bucket_sort=True,          # sort by atom count before packing; False → greedy on the
                               # shuffled order (random, diverse batches; no size grouping)
    max_atoms_per_batch=None,  # atom-budget batching: cap total atoms/batch (no OOM); implies bucket
    max_batch_count=None,      # optional cap on structures/batch in atom-budget mode
    device=None,
    checkpoint_path=None,
    verbose=True,
    # DDP (set automatically by __main__ when torchrun is detected)
    rank=0,
    world_size=1,
    local_rank=0,
):
    is_ddp = world_size > 1
    is_main = (rank == 0)
    verbose = verbose and is_main
    if eval_metric not in ('mae', 'rmse'):
        raise ValueError(f"eval_metric must be 'mae' or 'rmse', got {eval_metric!r}")

    if device is None:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    # TF32: on Ampere+ the step is dominated by fp32 matmuls (forward and the
    # force double-backward). Routing those to TF32 tensor cores is the cheapest
    # large speedup, but TF32 keeps only ~10 mantissa bits — A/B the val
    # force/energy MAE before trusting it. No effect under float64 (TF32 is a
    # float32-only mode), so warn rather than silently do nothing.
    if tf32:
        if dtype == torch.float64:
            if verbose:
                print_flush("  [tf32] requested but dtype=float64 → no effect "
                            "(TF32 is float32-only); use dtype=torch.float32")
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
            if verbose:
                print_flush("  [tf32] enabled: float32 matmuls → TF32 tensor cores "
                            "(A/B the val MAE against a tf32=False run)")

    # Use same seed on all ranks for data splitting so every rank trains on
    # the same structures; rank-specific seed only for training stochasticity.
    np.random.seed(seed)
    torch.manual_seed(seed + rank)

    # ── Load data ─────────────────────────────────────────────────────────
    if verbose:
        print_flush(f"Loading training data from {train_xyz}...")
    train_raw = parse_xyz_file(train_xyz, verbose=verbose)
    if verbose:
        print_flush(f"  Loaded {len(train_raw):,} training structures")

    if verbose:
        print_flush(f"Loading test data from {test_xyz}...")
    test_raw = parse_xyz_file(test_xyz, verbose=verbose)
    if verbose:
        print_flush(f"  Loaded {len(test_raw):,} test structures")

    if les_readout is None:
        # 'edge_basis' is the LES default; short-range runs take 'sum' so the
        # model carries no unused charge-head parameters (which would also
        # break DDP's find_unused_parameters=False).
        les_readout = 'edge_basis' if use_les else 'sum'
    # ── Split train → train + val ─────────────────────────────────────────
    idx = np.random.permutation(len(train_raw))
    n_val_actual = min(n_val, len(train_raw) // 10)
    val_raw   = [train_raw[i] for i in idx[:n_val_actual]]
    train_use = [train_raw[i] for i in idx[n_val_actual:]]
    if n_train is not None:
        train_use = train_use[:n_train]
    if n_test is not None:
        test_raw = test_raw[:n_test]

    if verbose:
        n_atoms_list = [s['n_atoms'] for s in train_use]
        print_flush(f"Train: {len(train_use):,} | Val: {len(val_raw):,} | Test: {len(test_raw):,}")
        print_flush(f"Train atom count: min={min(n_atoms_list)}, "
                    f"max={max(n_atoms_list)}, avg={np.mean(n_atoms_list):.1f}")
        print_flush(f"Device: {device}")

    # ── Per-element energy reference ─────────────────────────────────────
    if verbose:
        print_flush("Computing per-element energy reference...")
    e_ref = compute_energy_reference(train_use)
    if verbose:
        for t, name in enumerate(TYPE_NAMES):
            print_flush(f"  {name}: {e_ref[t]:.4f} eV/atom")

    # Re-seed numpy with rank so epoch-level sampling differs across ranks.
    np.random.seed(seed + rank)

    # ── Convert to tensors ────────────────────────────────────────────────
    if verbose:
        print_flush("Converting to tensors...")
    pos_train, frc_train, eng_train, typ_train = to_device_tensors(train_use, e_ref, dtype, device)
    pos_val,   frc_val,   eng_val,   typ_val   = to_device_tensors(val_raw,   e_ref, dtype, device)
    pos_test,  frc_test,  eng_test,  typ_test  = to_device_tensors(test_raw,  e_ref, dtype, device)

    # ── Model ─────────────────────────────────────────────────────────────
    model = ECENet(
        n_types=N_TYPES,
        r_cut_edge=r_cut_edge,
        r_cut_neighbor=r_cut_neighbor,
        l_max=l_max,
        n_max=n_max,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_max_d=n_max_d,
        m_max=m_max,
        cutoff_type=cutoff_type,
        activation=activation,
        use_nonlinearity=use_nonlinearity,
        n_grid=n_grid,
        output_hidden_dims=output_hidden_dims,
        analytic_ace_basis=analytic_ace_basis,
        bottleneck_dim=bottleneck_dim,
        n_mp=n_mp,
        mp_type=mp_type,
        mp_dim=mp_dim,
        mp_n_heads=mp_n_heads,
        mp_msg_envelope=mp_msg_envelope,
        mp_l_attention=mp_l_attention,
        element_film=element_film, film_embed_dim=film_embed_dim,
        film_n_rbf=film_n_rbf, film_hidden=film_hidden,
        film_per_m=film_per_m, film_shift=film_shift,
        les_readout=les_readout,
        les_charge_scale=les_charge_scale,
        les_dipole=les_dipole,
        les_charges=les_charges,
        les_alpha=les_alpha,
    )
    if dtype == torch.float64:
        model = model.double()
    model = model.to(device)

    raw_model = model   # unwrapped reference for eval + checkpointing

    if is_ddp and les_readout != 'sum' and not use_les:
        # The read-out's parameters (les_score / les_edge_charge) would never
        # enter the forward graph (nothing calls return_embeddings during
        # training), and a parameter that never receives a gradient breaks
        # DDP's find_unused_parameters=False collective. Refuse loudly rather
        # than hang/error mid-epoch.
        raise ValueError(f"les_readout={les_readout!r} without use_les=True "
                         "is not supported under DDP: its read-out parameters "
                         "would be unused in every forward.")

    # ── LES long-range module (optional) ──────────────────────────────────
    # Upstream builds its charge head lazily on the first forward (it infers
    # the descriptor width then), so run one throwaway forward NOW: the DDP
    # wrap, the optimizer, and any checkpoint restore below all need the
    # parameters to exist. Each rank materialises with its own RNG state;
    # DDP's construction-time broadcast then syncs rank 0's init to all.
    les_module = None
    if use_les:
        from ecenet.les import LESLongRange
        les_module = LESLongRange(les_arguments)
        with torch.no_grad():
            _, l0_list = model.forward_batch_multi(
                [pos_train[0]], [typ_train[0]],
                return_embeddings=True, l0_only=True)
            les_module(l0_list[0], pos_train[0],
                       **model.les_flags)
        les_module = les_module.to(device=device, dtype=dtype)

    all_params = list(model.parameters())
    if les_module is not None:
        all_params += list(les_module.parameters())

    if is_ddp:
        wrapper = _MultiForwardWrapper(model, les_module)
        train_model = DDP(wrapper, device_ids=[local_rank], find_unused_parameters=False)
        # create_graph=True in force training can produce non-contiguous grads,
        # causing DDP bucket-view stride mismatches. Make them contiguous first.
        for p in all_params:
            if p.requires_grad:
                p.register_hook(lambda g: g.contiguous())
    else:
        train_model = _MultiForwardWrapper(model, les_module)

    # Plain (non-DDP) forward for evaluation — rank 0 calls it alone, so it
    # must not be the DDP module. Includes E_lr when LES is on, so the val/test
    # MAEs measure the same total energy the loss trains.
    eval_fwd = _MultiForwardWrapper(model, les_module)

    # Precompute per-structure neighbour lists once (fixed training positions)
    # so the training step skips the O(N²) dist_mat + per-structure nonzero
    # syncs. Training only: evaluation rebuilds topology on the fly.
    topo_train = None
    if precompute_topology:
        topo_train = raw_model.build_topology(pos_train)
        if verbose:
            n_edges = sum(int(t[0].numel()) for t in topo_train)
            print_flush(f"  [precompute_topology] built neighbour lists for "
                        f"{len(topo_train):,} structures ({n_edges:,} edges total); "
                        f"per-step nonzero syncs skipped")

    n_params = sum(p.numel() for p in all_params if p.requires_grad)
    if verbose:
        model_name = "ECENet"
        m_max_eff = m_max if m_max is not None else l_max
        print_flush(f"\n{model_name}: {n_layers} layers, l_max={l_max}, "
                    f"m_max={m_max_eff}, n_max={n_max}, "
                    f"embed_dim={embed_dim}, n_max_d={n_max_d}")
        print_flush(f"  n_features_per_m: {model.n_features_per_m}")
        print_flush(f"  r_cut_edge={r_cut_edge}, r_cut_neighbor={r_cut_neighbor}")
        print_flush(f"  Trainable parameters: {n_params:,}"
                    + (" (incl. LES charge head)" if use_les else ""))

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
    # LR schedule. 'plateau' (default): ReduceLROnPlateau on the val metric.
    # 'cosine': linear warmup over warmup_epochs, then cosine decay to
    # lr*lr_min_factor. 'multistep': the same warmup, then lr *= lr_gamma at each
    # epoch in lr_milestones. The two open-loop schedules are pure functions of
    # the epoch index, so they carry no state, resume exactly from start_epoch,
    # and need nothing in the checkpoint — unlike torch's stateful MultiStepLR,
    # which counts .step() calls and would replay from the initial LR on resume.
    # Being a function of the epoch also means every DDP rank computes the same
    # LR independently, with nothing to keep in sync.
    if lr_schedule not in ('plateau', 'cosine', 'multistep'):
        raise ValueError("lr_schedule must be 'plateau', 'cosine' or 'multistep', "
                         f"got {lr_schedule!r}")
    milestones = sorted(int(m) for m in (lr_milestones or []))
    if lr_schedule == 'multistep' and not milestones:
        raise ValueError("lr_schedule='multistep' requires lr_milestones "
                         "(epochs at which to multiply the lr by lr_gamma).")
    if lr_schedule == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=scheduler_patience)
    else:
        scheduler = None
        if is_main:
            if lr_schedule == 'multistep':
                unreached = [m for m in milestones if m >= n_epochs]
                if unreached:
                    print_flush(f"  WARNING: milestones {unreached} are >= n_epochs="
                                f"{n_epochs} and will never fire.")
                steps = ', '.join(f"epoch {m}: lr→{lr * lr_gamma ** (i + 1):.2e}"
                                  for i, m in enumerate(milestones) if m < n_epochs)
                print_flush(f"  LR schedule: multistep (gamma={lr_gamma}) — {steps}")
            else:
                print_flush(f"  LR schedule: cosine (warmup={warmup_epochs}, "
                            f"floor={lr * lr_min_factor:.2e})")
            if warmup_epochs >= n_epochs:
                print_flush(f"  WARNING: warmup_epochs={warmup_epochs} >= n_epochs="
                            f"{n_epochs}; the decay phase never runs.")

    def open_loop_lr(epoch):
        """LR for a given (0-based) epoch under the cosine / multistep schedules."""
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return lr * (epoch + 1) / warmup_epochs            # linear warmup → lr
        if lr_schedule == 'multistep':
            # Counted from the epoch index, not accumulated, so a resumed run
            # lands on exactly the LR a fresh run would have at this epoch.
            return lr * (lr_gamma ** sum(1 for m in milestones if epoch >= m))
        # cosine: decay so the LAST epoch (n_epochs-1) reaches the floor exactly.
        progress = (epoch - warmup_epochs) / max(1, n_epochs - 1 - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        lr_min = lr * lr_min_factor
        return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * progress))

    # ── Checkpoint restore ────────────────────────────────────────────────
    start_epoch = 0
    best_val_weighted = float('inf')
    best_test_e_mae = float('nan')
    best_test_f_mae = float('nan')
    best_state = None
    best_les_state = None
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model'], strict=False)
        if use_les != ('les' in ckpt):
            raise ValueError(
                f"Checkpoint at {checkpoint_path} was trained with "
                f"use_les={'les' in ckpt}, but this run has use_les={use_les}.")
        if les_module is not None:
            les_module.load_state_dict(ckpt['les']['state_dict'])
            best_les_state = ckpt['les'].get('best_state')
        optimizer.load_state_dict(ckpt['optimizer'])
        # Open-loop schedules have no state (the LR is a function of the
        # epoch), and saved state may come from a different schedule.
        if scheduler is not None and ckpt.get('scheduler') is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        # Back-compat: older checkpoints stored 'best_val_force_mae'.
        best_val_weighted = ckpt.get('best_val_weighted',
                                     ckpt.get('best_val_force_mae', float('inf')))
        best_state = ckpt['best_state']
        best_test_e_mae = ckpt.get('best_test_e_mae', float('nan'))
        best_test_f_mae = ckpt.get('best_test_f_mae', float('nan'))
        if verbose:
            print_flush(f"Resumed from checkpoint: epoch {ckpt['epoch']}, "
                        f"best val [weighted]={best_val_weighted:.4f}, "
                        f"best test E={best_test_e_mae:.4f} F={best_test_f_mae:.4f}")

    def save_checkpoint(epoch):
        if checkpoint_path is None or not is_main:
            return
        print_flush("  Saving checkpoint...")
        out_les = (None if les_module is None else {
            'arguments': les_arguments,
            'state_dict': les_module.state_dict(),
            'best_state': best_les_state,
        })
        torch.save({
            **({'les': out_les} if out_les is not None else {}),
            'epoch': epoch,
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'best_val_weighted': best_val_weighted,
            'best_test_e_mae': best_test_e_mae,
            'best_test_f_mae': best_test_f_mae,
            'best_state': best_state,
            'hparams': dict(
                n_types=N_TYPES,
                r_cut_edge=r_cut_edge, r_cut_neighbor=r_cut_neighbor,
                l_max=l_max, n_max=n_max, embed_dim=embed_dim,
                n_layers=n_layers, n_max_d=n_max_d, m_max=m_max, n_grid=n_grid,
                cutoff_type=cutoff_type, activation=activation,
                use_nonlinearity=use_nonlinearity,
                output_hidden_dims=output_hidden_dims,
                analytic_ace_basis=analytic_ace_basis,
                bottleneck_dim=bottleneck_dim,
                n_mp=n_mp,
                mp_type=mp_type, mp_dim=mp_dim,
                mp_n_heads=mp_n_heads,
                mp_msg_envelope=mp_msg_envelope,
                mp_l_attention=mp_l_attention,
                element_film=element_film, film_embed_dim=film_embed_dim,
                film_n_rbf=film_n_rbf, film_hidden=film_hidden,
                film_per_m=film_per_m, film_shift=film_shift,
                les_readout=les_readout,
                les_charge_scale=les_charge_scale,
                les_dipole=les_dipole,
                les_charges=les_charges,
                les_alpha=les_alpha,
            ),
            'e_ref': e_ref,  # per-element reference energies (eV/atom)
            # Self-describing metadata for the calculator (no dataset coupling).
            'element_to_type': ELEMENT_TO_TYPE,   # {symbol: type_idx}
            'energy_units': 'eV',
        }, checkpoint_path)
        print_flush("  Checkpoint saved.")

    # ── Evaluation (rank 0 only) ──────────────────────────────────────────
    # eval_metric='mae': mean |err|. 'rmse': sqrt(mean err²) — same accumulation
    # with |err| swapped for err², and a sqrt at the end.
    def _err(diff):
        return (diff ** 2 if eval_metric == 'rmse' else diff.abs())

    def _final(total, count):
        if not count:
            return float('nan')
        m = total / count
        return math.sqrt(m) if eval_metric == 'rmse' else m

    def evaluate(pos_list, frc_list, eng_target, typ_list, max_samples=None):
        eval_fwd.eval()          # toggles the model AND the LES head
        indices = list(range(len(pos_list)))
        if max_samples is not None and max_samples < len(indices):
            indices = list(np.random.choice(indices, max_samples, replace=False))

        energy_acc = 0.0
        force_acc  = 0.0
        force_count = 0

        for start in range(0, len(indices), eval_batch_size):
            batch = indices[start:start + eval_batch_size]
            pos_b = [pos_list[i].detach().clone().requires_grad_(True) for i in batch]
            typ_b = [typ_list[i] for i in batch]

            with torch.enable_grad():
                eng_b = eval_fwd(pos_b, typ_b)
                # allow_unused: see training loop — a zero-edge structure
                # (e.g. a lone atom) has no positional dependence in its
                # predicted energy → forces are 0.
                grads = torch.autograd.grad(eng_b.sum(), pos_b, allow_unused=True)
                grads = tuple(
                    g if g is not None else torch.zeros_like(pos_b[k])
                    for k, g in enumerate(grads)
                )

            for k, i in enumerate(batch):
                n = pos_list[i].shape[0]
                energy_acc  += _err((eng_b[k] - eng_target[i]) / n).item()
                force_acc   += _err(-grads[k] - frc_list[i]).sum().item()
                force_count += frc_list[i].numel()

        eval_fwd.train()
        return _final(energy_acc, len(indices)), _final(force_acc, force_count)

    # ── Training loop ─────────────────────────────────────────────────────
    n_train_actual = len(pos_train)
    epoch_size = n_per_epoch if n_per_epoch is not None else n_train_actual

    if verbose:
        loss_desc = f"loss={loss}" + (f" (δ={huber_delta})" if loss == 'huber' else '')
        print_flush(f"\nTraining for {n_epochs} epochs "
                    f"(batch={batch_size}, epoch_size={epoch_size:,}, world_size={world_size}, "
                    f"lr={lr}, E-weight={energy_weight}, F-weight={force_weight}, {loss_desc})")

    # Size-aware batching (see size_aware_batches). Off unless requested.
    n_atoms_train = np.array([p.shape[0] for p in pos_train], dtype=np.int64)
    if verbose and bucket and max_atoms_per_batch is None:
        print_flush(f"  Size-bucketed batching: fixed batch_size={batch_size}, "
                    f"sorted by atom count and aligned across {world_size} rank(s)")
    if max_atoms_per_batch is not None:
        _largest = int(n_atoms_train.max())
        if _largest > max_atoms_per_batch:
            raise ValueError(
                f"max_atoms_per_batch={max_atoms_per_batch} is smaller than the "
                f"largest training structure ({_largest} atoms); it could never "
                "be placed in a batch.")
        if verbose:
            print_flush(f"  Atom-budget batching: <={max_atoms_per_batch} atoms/batch"
                        + (f", <={max_batch_count} structures" if max_batch_count else "")
                        + f" (mean {n_atoms_train.mean():.1f} atoms/structure), "
                        + ("size-sorted" if bucket_sort else
                           "greedy on the shuffled order (diverse batches)"))

    epochs_without_improvement = 0
    t_start = time.time()

    for epoch in range(start_epoch, n_epochs):
        # Open-loop schedules set this epoch's LR up front; 'plateau' instead
        # adjusts it below, after the val step.
        if scheduler is None:
            for pg in optimizer.param_groups:
                pg['lr'] = open_loop_lr(epoch)
        raw_model.train()
        epoch_loss = 0.0

        # Partition epoch across ranks: each rank handles epoch_size / world_size structures.
        # Together all ranks cover epoch_size structures per epoch → ~world_size× speedup.
        rank_epoch_size = (epoch_size + world_size - 1) // world_size  # ceil div
        if cycle_data and epoch_size < n_train_actual:
            # Cycle through full dataset in order: re-shuffle once per full pass,
            # then hand out consecutive chunks so every molecule appears exactly
            # once per (n_train_actual // epoch_size) epochs.
            chunks_per_cycle = n_train_actual // epoch_size
            cycle_num  = epoch // chunks_per_cycle
            chunk_idx  = epoch % chunks_per_cycle
            cycle_rng  = np.random.RandomState(seed + cycle_num)
            all_idx    = cycle_rng.permutation(n_train_actual)[:chunks_per_cycle * epoch_size]
            all_idx    = all_idx[chunk_idx * epoch_size:(chunk_idx + 1) * epoch_size]
        else:
            rng     = np.random.RandomState(seed + epoch)
            all_idx = rng.choice(n_train_actual, epoch_size, replace=(epoch_size > n_train_actual))
        if bucket or max_atoms_per_batch is not None:
            rank_batches = size_aware_batches(
                all_idx, n_atoms_train, world_size, rank, epoch, seed,
                batch_size=batch_size, max_atoms_per_batch=max_atoms_per_batch,
                max_batch_count=max_batch_count, bucket_sort=bucket_sort,
                verbose=verbose)
        else:
            rank_idx = all_idx[rank * rank_epoch_size:(rank + 1) * rank_epoch_size]
            n_b = (len(rank_idx) + batch_size - 1) // batch_size
            rank_batches = [rank_idx[b * batch_size:(b + 1) * batch_size]
                            for b in range(n_b)]
        n_batches = max(1, len(rank_batches))

        for batch_indices in rank_batches:
            optimizer.zero_grad()

            pos_rg   = [pos_train[i].detach().clone().requires_grad_(True) for i in batch_indices]
            typ_b    = [typ_train[i] for i in batch_indices]
            # Precomputed neighbour lists for this batch (positions are fixed,
            # so topo_train[i] matches the freshly re-leafed pos_rg).
            topo_b = ([topo_train[i] for i in batch_indices]
                      if topo_train is not None else None)
            eng_pred = train_model(pos_rg, typ_b, topology=topo_b)   # DDP syncs gradients here
            eng_tgt  = torch.stack([eng_train[i] for i in batch_indices])

            # Per-element loss according to --loss
            def _elem_loss(diff):
                if loss == 'mse':
                    return diff ** 2
                if loss == 'l1':
                    return diff.abs()
                # huber: L2 for |d| <= delta, L1 (linear) beyond
                abs_d = diff.abs()
                quad  = 0.5 * diff ** 2
                lin   = huber_delta * (abs_d - 0.5 * huber_delta)
                return torch.where(abs_d <= huber_delta, quad, lin)

            n_atoms_b = torch.tensor(
                [pos_train[i].shape[0] for i in batch_indices],
                dtype=dtype, device=device)
            energy_loss = _elem_loss((eng_pred - eng_tgt) / n_atoms_b).mean()

            if force_weight > 0:
                # allow_unused: a structure that ends up with zero edges (e.g. a
                # lone atom with no neighbour within r_cut_edge) contributes only
                # the constant per-element atomic_energy to eng_pred, so its
                # position leaf doesn't enter the graph. Forces are exactly zero
                # there, so substitute zeros for None grads.
                frc_grads = torch.autograd.grad(eng_pred.sum(), pos_rg,
                                                create_graph=True,
                                                allow_unused=True)
                frc_grads = tuple(
                    g if g is not None else torch.zeros_like(pos_rg[k])
                    for k, g in enumerate(frc_grads)
                )
                force_loss = sum(
                    _elem_loss(-frc_grads[k] - frc_train[batch_indices[k]]).mean()
                    for k in range(len(batch_indices))
                ) / len(batch_indices)
            else:
                force_loss = 0.0

            total_loss = energy_weight * energy_loss + force_weight * force_loss
            total_loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            epoch_loss += total_loss.item()


        epoch_loss /= n_batches

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            if is_ddp:
                dist.barrier()

            # Only rank 0 evaluates and logs
            val_weighted_tensor = torch.tensor(float('inf'), device=device)
            if is_main:
                print_flush("  Evaluating train...")
                train_e_mae, train_f_mae = evaluate(
                    pos_train, frc_train, eng_train, typ_train, max_samples=200)
                print_flush("  Evaluating val...")
                val_e_mae, val_f_mae = evaluate(
                    pos_val, frc_val, eng_val, typ_val)
                # Weighted selection metric (mirrors the training-loss weighting).
                val_weighted = energy_weight * val_e_mae + force_weight * val_f_mae
                val_weighted_tensor = torch.tensor(val_weighted, device=device)

            # Broadcast the weighted val metric to all ranks so the scheduler stays in sync
            if is_ddp:
                dist.broadcast(val_weighted_tensor, src=0)
            if scheduler is not None:
                scheduler.step(val_weighted_tensor.item())

            # Determine should_stop and exchange stop signal BEFORE test evaluation,
            # so non-main ranks don't time out waiting for rank 0's slow test eval.
            _do_test_eval = False
            if is_main:
                if val_weighted < best_val_weighted:
                    best_val_weighted = val_weighted
                    best_state = {k: v.clone() for k, v in raw_model.state_dict().items()}
                    if les_module is not None:
                        best_les_state = {k: v.clone()
                                          for k, v in les_module.state_dict().items()}
                    epochs_without_improvement = 0
                    _do_test_eval = True
                else:
                    epochs_without_improvement += 1

                should_stop = (early_stopping_patience is not None
                              and epochs_without_improvement >= early_stopping_patience)
                if is_ddp:
                    # Broadcast stop signal now — before the slow test evaluation.
                    stop = torch.tensor(1 if should_stop else 0, device=device)
                    dist.broadcast(stop, src=0)
            elif is_ddp:
                # Non-main ranks receive the stop signal and are now free to proceed.
                stop = torch.tensor(0, device=device)
                dist.broadcast(stop, src=0)
                if stop.item() == 1:
                    break

            # Rank 0 only: test evaluation, checkpoint, logging.
            # Other ranks are already in the next epoch (or stopped) — no NCCL ops here.
            if is_main:
                if _do_test_eval:
                    print_flush("  Evaluating test...")
                    best_test_e_mae, best_test_f_mae = evaluate(
                        pos_test, frc_test, eng_test, typ_test)

                save_checkpoint(epoch)

                elapsed = time.time() - t_start
                lr_now = optimizer.param_groups[0]['lr']
                print_flush(
                    f"  Epoch {epoch+1:3d}: loss={epoch_loss:.4f} | [{eval_metric}] "
                    f"train E={train_e_mae:.4f} F={train_f_mae:.4f} | "
                    f"val E={val_e_mae:.4f} F={val_f_mae:.4f} | "
                    f"lr={lr_now:.1e} | {elapsed:.0f}s | "
                    f"best val [weighted]={best_val_weighted:.4f} "
                    f"[test E={best_test_e_mae:.4f} F={best_test_f_mae:.4f}]")

                if should_stop:
                    print_flush(f"  Early stopping at epoch {epoch+1}")
                    break

    # ── Final evaluation (rank 0 only) ───────────────────────────────────
    results = {}
    if is_main:
        if best_state is not None:
            raw_model.load_state_dict(best_state, strict=False)
            if les_module is not None and best_les_state is not None:
                les_module.load_state_dict(best_les_state)

        train_e_mae, train_f_mae = evaluate(pos_train, frc_train, eng_train, typ_train, max_samples=500)
        val_e_mae,   val_f_mae   = evaluate(pos_val,   frc_val,   eng_val,   typ_val)
        test_e_mae,  test_f_mae  = evaluate(pos_test,  frc_test,  eng_test,  typ_test)
        total_time = time.time() - t_start

        print_flush(f"\nFinal Results ({eval_metric.upper()}):")
        print_flush(f"  Train: E={train_e_mae:.4f} eV/atom, F={train_f_mae:.4f} eV/Å")
        print_flush(f"  Val:   E={val_e_mae:.4f} eV/atom, F={val_f_mae:.4f} eV/Å")
        print_flush(f"  Test:  E={test_e_mae:.4f} eV/atom, F={test_f_mae:.4f} eV/Å")
        print_flush(f"Total time: {total_time:.1f}s")

        results = {
            f'train_energy_{eval_metric}': train_e_mae, f'train_force_{eval_metric}': train_f_mae,
            f'val_energy_{eval_metric}':   val_e_mae,   f'val_force_{eval_metric}':   val_f_mae,
            f'test_energy_{eval_metric}':  test_e_mae,  f'test_force_{eval_metric}':  test_f_mae,
            'n_params': n_params, 'time': total_time,
            'les_module': les_module,   # None unless use_les
        }

    if is_ddp:
        dist.destroy_process_group()

    return raw_model, results


# ---------------------------------------------------------------------------
# Entry point — torchrun-compatible (multi-GPU DDP)
# ---------------------------------------------------------------------------
# torchrun sets LOCAL_RANK / RANK / WORLD_SIZE in the environment; we read them
# here and hand them to train_ecenet_spice for DDP setup. Set hyperparameters by
# editing the call below (or import train_ecenet_spice from your own driver).
#
#     python scripts/train_ecenet_spice.py                 # single process
#     torchrun --nproc_per_node=4 scripts/train_ecenet_spice.py   # multi-GPU

if __name__ == "__main__":
    local_rank  = int(os.environ.get('LOCAL_RANK', 0))
    rank        = int(os.environ.get('RANK', 0))
    world_size  = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    train_ecenet_spice(rank=rank, world_size=world_size, local_rank=local_rank)
