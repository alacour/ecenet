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


import re
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ecenet import ECENet


class _MultiForwardWrapper(nn.Module):
    """Thin wrapper so DDP intercepts forward_batch_multi for gradient sync."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, positions_list, types_list):
        return self.model.forward_batch_multi(positions_list, types_list)


def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


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
    edge_type_nonlin=False,
    edge_type_linear=False,
    edge_type_output=False,
    analytic_ace_basis=True,
    n_dist_embed=0,
    # Message passing
    n_mp=1,
    n_dist_basis=8,
    # Optimiser
    lr=1e-3,
    weight_decay=1e-5,
    grad_clip=None,
    scheduler_patience=10,
    early_stopping_patience=None,
    # Training
    n_epochs=100,
    batch_size=8,
    energy_weight=1.0,
    force_weight=1.0,
    loss='mse',
    huber_delta=0.01,
    eval_every=1,
    eval_batch_size=32,
    seed=42,
    dtype=torch.float64,
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

    if device is None:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{local_rank}')
        else:
            device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)

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
        n_dist_embed=n_dist_embed,
        edge_type_nonlin=edge_type_nonlin,
        edge_type_linear=edge_type_linear,
        edge_type_output=edge_type_output,
        n_mp=n_mp,
        n_dist_basis=n_dist_basis,
    )
    if dtype == torch.float64:
        model = model.double()
    model = model.to(device)

    raw_model = model   # unwrapped reference for eval + checkpointing

    if is_ddp:
        wrapper = _MultiForwardWrapper(model)
        train_model = DDP(wrapper, device_ids=[local_rank], find_unused_parameters=False)
        # create_graph=True in force training can produce non-contiguous grads,
        # causing DDP bucket-view stride mismatches. Make them contiguous first.
        for p in model.parameters():
            if p.requires_grad:
                p.register_hook(lambda g: g.contiguous())
    else:
        train_model = _MultiForwardWrapper(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        model_name = "ECENet"
        m_max_eff = m_max if m_max is not None else l_max
        print_flush(f"\n{model_name}: {n_layers} layers, l_max={l_max}, "
                    f"m_max={m_max_eff}, n_max={n_max}, "
                    f"embed_dim={embed_dim}, n_max_d={n_max_d}")
        print_flush(f"  n_features_per_m: {model.n_features_per_m}")
        print_flush(f"  r_cut_edge={r_cut_edge}, r_cut_neighbor={r_cut_neighbor}")
        print_flush(f"  Trainable parameters: {n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience)

    # ── Checkpoint restore ────────────────────────────────────────────────
    start_epoch = 0
    best_val_weighted = float('inf')
    best_test_e_mae = float('nan')
    best_test_f_mae = float('nan')
    best_state = None
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
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
        torch.save({
            'epoch': epoch,
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
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
                n_dist_embed=n_dist_embed,
                edge_type_nonlin=edge_type_nonlin,
                edge_type_linear=edge_type_linear,
                edge_type_output=edge_type_output,
                n_mp=n_mp, n_dist_basis=n_dist_basis,
            ),
            'e_ref': e_ref,  # per-element reference energies (eV/atom)
            # Self-describing metadata for the calculator (no dataset coupling).
            'element_to_type': ELEMENT_TO_TYPE,   # {symbol: type_idx}
            'energy_units': 'eV',
        }, checkpoint_path)
        print_flush("  Checkpoint saved.")

    # ── Evaluation (rank 0 only) ──────────────────────────────────────────
    def evaluate(pos_list, frc_list, eng_target, typ_list, max_samples=None):
        raw_model.eval()
        indices = list(range(len(pos_list)))
        if max_samples is not None and max_samples < len(indices):
            indices = list(np.random.choice(indices, max_samples, replace=False))

        energy_abs = 0.0
        force_abs  = 0.0
        force_count = 0

        for start in range(0, len(indices), eval_batch_size):
            batch = indices[start:start + eval_batch_size]
            pos_b = [pos_list[i].detach().clone().requires_grad_(True) for i in batch]
            typ_b = [typ_list[i] for i in batch]

            with torch.enable_grad():
                eng_b = raw_model.forward_batch_multi(pos_b, typ_b)
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
                energy_abs  += (eng_b[k] - eng_target[i]).abs().item() / n
                force_abs   += (-grads[k] - frc_list[i]).abs().sum().item()
                force_count += frc_list[i].numel()

        raw_model.train()
        n = len(indices)
        return energy_abs / n, force_abs / force_count

    # ── Training loop ─────────────────────────────────────────────────────
    n_train_actual = len(pos_train)
    epoch_size = n_per_epoch if n_per_epoch is not None else n_train_actual

    if verbose:
        loss_desc = f"loss={loss}" + (f" (δ={huber_delta})" if loss == 'huber' else '')
        print_flush(f"\nTraining for {n_epochs} epochs "
                    f"(batch={batch_size}, epoch_size={epoch_size:,}, world_size={world_size}, "
                    f"lr={lr}, E-weight={energy_weight}, F-weight={force_weight}, {loss_desc})")

    epochs_without_improvement = 0
    t_start = time.time()

    for epoch in range(start_epoch, n_epochs):
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
        rank_idx = all_idx[rank * rank_epoch_size:(rank + 1) * rank_epoch_size]
        n_batches = (len(rank_idx) + batch_size - 1) // batch_size

        for b in range(n_batches):
            batch_indices = rank_idx[b * batch_size:(b + 1) * batch_size]
            optimizer.zero_grad()

            pos_rg   = [pos_train[i].detach().clone().requires_grad_(True) for i in batch_indices]
            typ_b    = [typ_train[i] for i in batch_indices]
            eng_pred = train_model(pos_rg, typ_b)   # DDP syncs gradients here
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
            scheduler.step(val_weighted_tensor.item())

            # Determine should_stop and exchange stop signal BEFORE test evaluation,
            # so non-main ranks don't time out waiting for rank 0's slow test eval.
            _do_test_eval = False
            if is_main:
                if val_weighted < best_val_weighted:
                    best_val_weighted = val_weighted
                    best_state = {k: v.clone() for k, v in raw_model.state_dict().items()}
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
                    f"  Epoch {epoch+1:3d}: loss={epoch_loss:.4f} | "
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

        train_e_mae, train_f_mae = evaluate(pos_train, frc_train, eng_train, typ_train, max_samples=500)
        val_e_mae,   val_f_mae   = evaluate(pos_val,   frc_val,   eng_val,   typ_val)
        test_e_mae,  test_f_mae  = evaluate(pos_test,  frc_test,  eng_test,  typ_test)
        total_time = time.time() - t_start

        print_flush("\nFinal Results (MAE):")
        print_flush(f"  Train: E={train_e_mae:.4f} eV/atom, F={train_f_mae:.4f} eV/Å")
        print_flush(f"  Val:   E={val_e_mae:.4f} eV/atom, F={val_f_mae:.4f} eV/Å")
        print_flush(f"  Test:  E={test_e_mae:.4f} eV/atom, F={test_f_mae:.4f} eV/Å")
        print_flush(f"Total time: {total_time:.1f}s")

        results = {
            'train_energy_mae': train_e_mae, 'train_force_mae': train_f_mae,
            'val_energy_mae':   val_e_mae,   'val_force_mae':   val_f_mae,
            'test_energy_mae':  test_e_mae,  'test_force_mae':  test_f_mae,
            'n_params': n_params, 'time': total_time,
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
