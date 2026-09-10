"""Training script for ECENet on small ASE-readable datasets (extxyz etc.).

The lightweight, single-process analogue of ``train_ecenet_mptrj.py`` for
datasets that fit comfortably in memory (hundreds to a few thousand frames):
liquid water, electrolytes, molten salts, and the other LES benchmark sets.
Reuses the MPtrj trainer's loader / split / topology machinery; drops DDP,
prepared shards and CPU-offload, and adds the one thing the big trainers do
not have yet: **joint LES long-range training**.

With ``use_les=True`` the model's rotation-invariant per-atom embedding ``l0``
feeds ``ecenet.les.LESLongRange`` (a wrapper around the inventors' ``les``
package — optional install, see ecenet/les.py for the licensing note), and

    E = E_sr + E_lr

is minimised on one autograd graph, so forces (and stress, via the strain
trick) need no extra code. The LES charge head is built lazily by upstream on
its first forward, so the trainer materialises it with one throwaway forward
before creating the optimiser / restoring a checkpoint.

Checkpoints carry the usual self-describing keys (``hparams``,
``element_to_type``, ``e_ref``) plus, for LES runs, a top-level ``les`` dict
with the wrapper's state. ``ECENetCalculator.from_checkpoint`` refuses such
checkpoints rather than silently dropping the long-range term (pass
``ignore_les=True`` there to load the short-range part deliberately).

Stress: ASE ``info['stress']`` is already in eV/Å³ (Voigt or 3×3), so the
default ``stress_conv=1.0`` — unlike the MPtrj trainer, whose raw kBar input
needs converting.

Usage (import-and-call; every option is a keyword argument):

    from scripts.train_ecenet_xyz import train_ecenet_xyz
    model, les_module, results = train_ecenet_xyz(
        train_path='data/train-H2O_RPBE-D3.xyz',
        test_path='data/test-H2O_RPBE-D3.xyz',
        n_epochs=200, use_les=True)
"""

import os
import sys  # repo root + scripts/ on path for `import ecenet` / the mptrj helpers

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import math
import time

import numpy as np
import torch
from train_ecenet_mptrj import (
    build_topology,
    compute_energy_reference,
    load_mptrj,
    print_flush,
    split_by_frame,
)

from ecenet import ECENet, elements

# ---------------------------------------------------------------------------
# Structures → on-device tensor dicts (with topology and cell)
# ---------------------------------------------------------------------------
# Like the MPtrj trainer's to_device_tensors, but also keeps the cell tensor:
# the LES reciprocal-space path needs it, and under stress training it must be
# strained alongside positions and shifts.

def tensorize(structures, type_map, e_ref, r_cut_edge, r_cut_nb,
              stress_conv, dtype, device):
    out = []
    for s in structures:
        types_np = np.array([type_map[int(z)] for z in s['numbers']], dtype=np.int64)
        ref = sum(e_ref[type_map[int(z)]] for z in s['numbers'])

        ei, ej, shift_e, ni, nj, shift_nb = build_topology(
            s['positions'], s['cell'], s['pbc'], r_cut_edge, r_cut_nb, device, dtype)

        periodic = s['pbc'] and s['cell'] is not None
        cell_t = (torch.tensor(s['cell'], dtype=dtype, device=device)
                  if periodic else None)
        volume = abs(np.linalg.det(s['cell'])) if periodic else 0.0

        stress_t = None
        if s['stress'] is not None and volume > 0:
            stress_t = torch.tensor(np.asarray(s['stress']) * stress_conv,
                                    dtype=dtype, device=device)

        out.append({
            'pos':     torch.tensor(s['positions'], dtype=dtype, device=device),
            'types':   torch.tensor(types_np, dtype=torch.long, device=device),
            'energy':  torch.tensor(s['energy'] - ref, dtype=dtype, device=device),
            'forces':  torch.tensor(s['forces'], dtype=dtype, device=device),
            'stress':  stress_t,
            'cell':    cell_t,
            'volume':  volume,
            'edge_i':  ei, 'edge_j': ej, 'shift_e': shift_e,
            'nb_src':  ni, 'nb_dst': nj, 'shift_nb': shift_nb,
            'n_atoms': s['n_atoms'],
        })
    return out


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_ecenet_xyz(
    train_path='train.xyz',
    test_path=None,
    data_format='auto',
    energy_key='energy',
    # Pre-loaded structure dicts (bypass file loading; used by tests)
    train_structures=None,
    test_structures=None,
    max_load=None,
    # Splits
    n_train=None,
    val_frac=0.1,
    n_val=None,
    n_test=None,
    # Long-range (LES)
    use_les=False,
    les_arguments=None,      # extra kwargs for upstream les.Les (see ecenet/les.py)
    les_readout=None,       # None -> 'edge_basis' if use_les else 'sum'
    les_charge_scale=1.0,    # fixed multiplier on the edge-mode latent charge (MACELES: 0.1)
    les_dipole=False,        # edge head also emits bond dipoles; l0 packed [q | u]
    les_charges=True,        # False (needs les_dipole): dipoles-only — q hard zero, standard-init dipole head
    les_alpha=None,          # None | 'iso' | 'aniso': edge head also emits polarizabilities (induced dipoles)
    # Geometry
    r_cut_edge=5.0,
    r_cut_neighbor=4.0,
    l_max=3,
    n_max=4,
    cutoff_type='cosine',
    # Architecture (mirrors train_ecenet_mptrj.py)
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
    # Optimiser
    lr=1e-3,
    weight_decay=1e-5,
    grad_clip=None,
    scheduler_patience=10,      # 'plateau' only
    lr_schedule='plateau',      # 'plateau' | 'cosine' | 'multistep'
    warmup_epochs=0,
    lr_min_factor=0.0,
    lr_milestones=None,
    lr_gamma=0.1,
    early_stopping_patience=None,
    # Training
    n_epochs=100,
    batch_size=4,
    energy_weight=1.0,
    force_weight=1.0,
    stress_weight=0.0,
    stress_conv=1.0,          # ASE info['stress'] is already eV/Å³
    loss='mse',
    loss_type=None,           # alias for `loss` (train_ecenet.py's name); wins if set
    huber_delta=0.01,
    eval_metric='mae',        # 'mae' | 'rmse' — reported metrics AND the best-val
                              # selection value (the training loss itself is unchanged)
    best_metric='weighted',   # 'force' | 'energy' | 'weighted' — which val metric
                              # selects the best checkpoint (and drives 'plateau')
    eval_every=1,
    eval_batch_size=16,
    seed=42,
    dtype=torch.float64,
    tf32=False,
    device=None,
    checkpoint_path=None,
    verbose=True,
):
    use_stress = stress_weight > 0
    if loss_type is not None:
        loss = loss_type
    if loss not in ('mse', 'l1', 'huber'):
        raise ValueError(f"loss must be 'mse', 'l1' or 'huber', got {loss!r}")
    if eval_metric not in ('mae', 'rmse'):
        raise ValueError(f"eval_metric must be 'mae' or 'rmse', got {eval_metric!r}")
    if best_metric not in ('force', 'energy', 'weighted'):
        raise ValueError("best_metric must be 'force', 'energy' or 'weighted', "
                         f"got {best_metric!r}")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

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

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── Load data ─────────────────────────────────────────────────────────
    if train_structures is not None:
        train_raw = train_structures
    else:
        train_raw = load_mptrj(train_path, data_format, energy_key,
                               max_structures=max_load, verbose=verbose)
    if test_structures is not None:
        test_raw = test_structures
    elif test_path is not None:
        test_raw = load_mptrj(test_path, data_format, energy_key,
                              max_structures=max_load, verbose=verbose)
    else:
        test_raw = []

    train_use, val_raw = split_by_frame(train_raw, val_frac, seed)
    if n_train is not None:
        train_use = train_use[:n_train]
    if n_val is not None:
        val_raw = val_raw[:n_val]
    if n_test is not None:
        test_raw = test_raw[:n_test]

    type_map = elements.build_type_map(
        z for s in (train_raw + test_raw) for z in s['numbers'])
    if les_readout is None:
        # 'edge_basis' is the LES default; short-range runs take 'sum' so the
        # model carries no unused charge-head parameters (which would also
        # break DDP's find_unused_parameters=False).
        les_readout = 'edge_basis' if use_les else 'sum'
    n_types = len(type_map)
    if verbose:
        n_atoms_list = [s['n_atoms'] for s in train_use]
        elems = ' '.join(elements.symbol(z) for z in sorted(type_map))
        print_flush(f"Train: {len(train_use):,} | Val: {len(val_raw):,} | "
                    f"Test: {len(test_raw):,} frames")
        print_flush(f"Atoms/struct: min={min(n_atoms_list)} max={max(n_atoms_list)} "
                    f"avg={np.mean(n_atoms_list):.1f}")
        print_flush(f"n_types={n_types}: {elems}")
        print_flush(f"Device: {device} | stress={'on' if use_stress else 'off'} | "
                    f"LES={'on' if use_les else 'off'}")

    e_ref = compute_energy_reference(train_use, type_map)

    train_data = tensorize(train_use, type_map, e_ref, r_cut_edge,
                           r_cut_neighbor, stress_conv, dtype, device)
    val_data   = tensorize(val_raw,   type_map, e_ref, r_cut_edge,
                           r_cut_neighbor, stress_conv, dtype, device)
    test_data  = tensorize(test_raw,  type_map, e_ref, r_cut_edge,
                           r_cut_neighbor, stress_conv, dtype, device)

    # ── Model ─────────────────────────────────────────────────────────────
    model = ECENet(
        n_types=n_types,
        r_cut_edge=r_cut_edge, r_cut_neighbor=r_cut_neighbor,
        l_max=l_max, n_max=n_max, embed_dim=embed_dim, n_layers=n_layers,
        n_max_d=n_max_d, m_max=m_max, cutoff_type=cutoff_type,
        activation=activation, use_nonlinearity=use_nonlinearity, n_grid=n_grid,
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
    )
    if dtype == torch.float64:
        model = model.double()
    model = model.to(device)

    # ── LES long-range module (optional) ──────────────────────────────────
    # Upstream builds its charge MLP lazily on the first forward (it infers
    # the descriptor width then), so run one throwaway forward NOW: the
    # optimiser and any checkpoint restore below need the parameters to exist.
    les_module = None
    if use_les:
        from ecenet.les import LESLongRange
        les_module = LESLongRange(les_arguments)
        d0 = train_data[0]
        with torch.no_grad():
            _, l0 = model.forward_pbc(
                d0['pos'], d0['types'], d0['edge_i'], d0['edge_j'], d0['shift_e'],
                d0['nb_src'], d0['nb_dst'], d0['shift_nb'],
                return_embeddings=True, l0_only=True)
            les_module(l0, d0['pos'], cell=d0['cell'], **model.les_flags)
        les_module = les_module.to(device=device, dtype=dtype)

    params = list(model.parameters())
    if les_module is not None:
        params += list(les_module.parameters())
    n_params = sum(p.numel() for p in params if p.requires_grad)
    if verbose:
        print_flush(f"\nECENet: {n_layers} layers, l_max={l_max}, n_max={n_max}, "
                    f"embed_dim={embed_dim}, n_types={n_types}")
        print_flush(f"  Trainable parameters: {n_params:,}"
                    + (" (incl. LES charge head)" if use_les else ""))

    # ── Optimiser / LR schedule (same semantics as the other trainers) ────
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
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

    def open_loop_lr(epoch):
        """LR at a (0-based) epoch under cosine / multistep — a pure function of
        the epoch index (resume-exact; nothing to checkpoint)."""
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return lr * (epoch + 1) / warmup_epochs
        if lr_schedule == 'multistep':
            return lr * (lr_gamma ** sum(1 for m in milestones if epoch >= m))
        progress = (epoch - warmup_epochs) / max(1, n_epochs - 1 - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        lr_min = lr * lr_min_factor
        return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * progress))

    # ── Checkpoint restore ────────────────────────────────────────────────
    start_epoch = 0
    best_val = float('inf')
    best_test = (float('nan'), float('nan'), float('nan'))
    best_state = None
    best_les_state = None
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])          # strict: resume must match
        if use_les != ('les' in ckpt):
            raise ValueError(
                f"Checkpoint at {checkpoint_path} was trained with "
                f"use_les={'les' in ckpt}, but this run has use_les={use_les}.")
        if les_module is not None:
            les_module.load_state_dict(ckpt['les']['state_dict'])
            best_les_state = ckpt['les'].get('best_state')
        optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler is not None and ckpt.get('scheduler') is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        saved_metric = ckpt.get('best_metric', 'weighted')
        best_val = ckpt.get('best_val', ckpt.get('best_val_weighted', float('inf')))
        if saved_metric != best_metric:
            # Values under different metrics are not comparable — restart the
            # best-checkpoint selection rather than compare apples to oranges.
            best_val = float('inf')
        best_state = ckpt['best_state']
        best_test = ckpt.get('best_test', best_test)
        if verbose:
            print_flush(f"Resumed from epoch {ckpt['epoch']}, "
                        f"best val [{best_metric}]={best_val:.4f}")

    def save_checkpoint(epoch):
        if checkpoint_path is None:
            return
        out = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'best_val': best_val,
            'best_metric': best_metric,
            'best_test': best_test,
            'best_state': best_state,
            'hparams': dict(
                n_types=n_types,
                r_cut_edge=r_cut_edge, r_cut_neighbor=r_cut_neighbor,
                l_max=l_max, n_max=n_max, embed_dim=embed_dim, n_layers=n_layers,
                n_max_d=n_max_d, m_max=m_max, n_grid=n_grid, cutoff_type=cutoff_type,
                activation=activation, use_nonlinearity=use_nonlinearity,
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
            'element_to_type': elements.to_element_to_type(type_map),
            'e_ref': e_ref,
            'stress_conv': stress_conv,
        }
        if les_module is not None:
            out['les'] = {
                'arguments': les_arguments,
                'state_dict': les_module.state_dict(),
                'best_state': best_les_state,
            }
        torch.save(out, checkpoint_path)

    # ── Loss helper ───────────────────────────────────────────────────────
    def elem_loss(diff):
        if loss == 'mse':
            return diff ** 2
        if loss == 'l1':
            return diff.abs()
        abs_d = diff.abs()
        return torch.where(abs_d <= huber_delta, 0.5 * diff ** 2,
                           huber_delta * (abs_d - 0.5 * huber_delta))

    # ── Forward over a batch, with strain leaves for stress ──────────────
    def predict(batch, create_graph):
        """Energies (SR + optional LES) with force/stress autograd.

        Everything a structure's energy depends on — positions, PBC shift
        vectors, and (for LES's Ewald part) the cell — is strain-transformed
        with one ε leaf per structure, so σ = (1/V)·dE/dε covers the
        long-range term too.
        """
        energies = []
        pos_leaf, strain_leaf = [], []
        for d in batch:
            p = d['pos'].detach().clone().requires_grad_(True)
            pos_leaf.append(p)
            cell_in = d['cell']
            if use_stress:
                eps = torch.zeros(3, 3, dtype=p.dtype, device=p.device,
                                  requires_grad=True)
                strain_leaf.append(eps)
                pos_in = p + p @ eps
                shift_e_in = d['shift_e'] + d['shift_e'] @ eps
                shift_nb_in = d['shift_nb'] + d['shift_nb'] @ eps
                if cell_in is not None:
                    cell_in = cell_in + cell_in @ eps
            else:
                pos_in, shift_e_in, shift_nb_in = p, d['shift_e'], d['shift_nb']

            if use_les:
                e_sr, l0 = model.forward_pbc(
                    pos_in, d['types'], d['edge_i'], d['edge_j'], shift_e_in,
                    d['nb_src'], d['nb_dst'], shift_nb_in,
                    return_embeddings=True, l0_only=True)
                e_lr = les_module(l0, pos_in, cell=cell_in,
                                  **model.les_flags)
                energies.append(e_sr + e_lr.sum())
            else:
                energies.append(model.forward_pbc(
                    pos_in, d['types'], d['edge_i'], d['edge_j'], shift_e_in,
                    d['nb_src'], d['nb_dst'], shift_nb_in))
        energies = torch.stack(energies)

        forces_list = stress_list = None
        if force_weight > 0 or use_stress:
            grad_inputs = pos_leaf + strain_leaf
            # allow_unused: a zero-edge structure never puts its position leaf
            # into the SR graph (LES always does, so this only fires with
            # use_les=False); the physical gradient is exactly zero there.
            grads = torch.autograd.grad(energies.sum(), grad_inputs,
                                        create_graph=create_graph,
                                        allow_unused=True)
            B = len(batch)
            forces_list = [
                -grads[k] if grads[k] is not None else torch.zeros_like(pos_leaf[k])
                for k in range(B)
            ]
            if use_stress:
                stress_list = [
                    (grads[B + k] if grads[B + k] is not None
                     else torch.zeros_like(strain_leaf[k])) / batch[k]['volume']
                    for k in range(B)
                ]
        return energies, forces_list, stress_list

    def _train_mode(train):
        model.train(train)
        if les_module is not None:
            les_module.train(train)

    # ── Evaluation ────────────────────────────────────────────────────────
    # eval_metric='mae': mean |err|. 'rmse': sqrt(mean err²) — same accumulation
    # with the per-element |err| swapped for err², and a sqrt at the end.
    def _err(diff):
        return (diff ** 2 if eval_metric == 'rmse' else diff.abs())

    def _final(total, count):
        if not count:
            return float('nan')
        m = total / count
        return math.sqrt(m) if eval_metric == 'rmse' else m

    def evaluate(data, max_samples=None):
        _train_mode(False)
        if max_samples is not None and max_samples < len(data):
            idx = np.random.choice(len(data), max_samples, replace=False)
            data = [data[int(i)] for i in idx]
        e_acc = f_acc = s_acc = 0.0
        f_count = s_count = n = 0
        for start in range(0, len(data), eval_batch_size):
            batch = data[start:start + eval_batch_size]
            with torch.enable_grad():
                energies, forces_list, stress_list = predict(batch, create_graph=False)
            for k, d in enumerate(batch):
                e_acc += _err((energies[k] - d['energy']) / d['n_atoms']).item()
                if forces_list is not None:
                    f_acc += _err(forces_list[k] - d['forces']).sum().item()
                    f_count += d['forces'].numel()
                if stress_list is not None and d['stress'] is not None:
                    s_acc += _err(stress_list[k] - d['stress']).sum().item()
                    s_count += d['stress'].numel()
            n += len(batch)
        _train_mode(True)
        return _final(e_acc, n), _final(f_acc, f_count), _final(s_acc, s_count)

    # ── Training loop ─────────────────────────────────────────────────────
    if verbose:
        sloss = f" S-weight={stress_weight}" if use_stress else ""
        print_flush(f"\nTraining {n_epochs} epochs (batch={batch_size}, "
                    f"n_train={len(train_data)}, lr={lr}, E-weight={energy_weight}, "
                    f"F-weight={force_weight}{sloss}, loss={loss})")

    epochs_without_improvement = 0
    t_start = time.time()

    for epoch in range(start_epoch, n_epochs):
        if scheduler is None:
            for pg in optimizer.param_groups:
                pg['lr'] = open_loop_lr(epoch)
        _train_mode(True)
        epoch_loss = 0.0

        rng = np.random.RandomState(seed + epoch)
        perm = rng.permutation(len(train_data))
        n_batches = 0
        for b in range(0, len(perm), batch_size):
            batch = [train_data[i] for i in perm[b:b + batch_size]]
            n_batches += 1
            optimizer.zero_grad()

            energies, forces_list, stress_list = predict(batch, create_graph=True)
            eng_tgt = torch.stack([d['energy'] for d in batch])
            n_atoms_b = torch.tensor([d['n_atoms'] for d in batch],
                                     dtype=dtype, device=device)
            energy_loss = elem_loss((energies - eng_tgt) / n_atoms_b).mean()

            force_loss = energies.new_zeros(())
            if force_weight > 0:
                force_loss = sum(elem_loss(forces_list[k] - batch[k]['forces']).mean()
                                 for k in range(len(batch))) / len(batch)

            stress_loss = energies.new_zeros(())
            if use_stress:
                terms = [elem_loss(stress_list[k] - batch[k]['stress']).mean()
                         for k in range(len(batch)) if batch[k]['stress'] is not None]
                if terms:
                    stress_loss = sum(terms) / len(terms)

            total_loss = (energy_weight * energy_loss + force_weight * force_loss
                          + stress_weight * stress_loss)
            total_loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            optimizer.step()
            epoch_loss += total_loss.item()

        epoch_loss /= max(1, n_batches)

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            tr_e, tr_f, tr_s = evaluate(train_data, max_samples=200)
            va_e, va_f, va_s = evaluate(val_data)
            va_weighted = energy_weight * va_e + force_weight * va_f
            if use_stress:
                va_weighted += stress_weight * va_s
            va_sel = {'force': va_f, 'energy': va_e,
                      'weighted': va_weighted}[best_metric]
            if scheduler is not None:
                scheduler.step(va_sel)

            if va_sel < best_val:
                best_val = va_sel
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                if les_module is not None:
                    best_les_state = {k: v.clone()
                                      for k, v in les_module.state_dict().items()}
                epochs_without_improvement = 0
                best_test = evaluate(test_data) if test_data else best_test
            else:
                epochs_without_improvement += 1

            save_checkpoint(epoch)
            if verbose:
                lr_now = optimizer.param_groups[0]['lr']
                ssfx = f" S={va_s:.4f}" if use_stress else ""
                # Test metrics are frozen at the last val improvement (mirrors
                # the MPtrj trainer); omitted when there is no test set.
                btsfx = f" S={best_test[2]:.4f}" if use_stress else ""
                bt = (f" [test E={best_test[0]:.4f} F={best_test[1]:.4f}{btsfx}]"
                      if test_data else "")
                print_flush(
                    f"  Epoch {epoch+1:3d}: loss={epoch_loss:.4f} | [{eval_metric}] "
                    f"train E={tr_e:.4f} F={tr_f:.4f} | val E={va_e:.4f} F={va_f:.4f}{ssfx} | "
                    f"lr={lr_now:.1e} | {time.time()-t_start:.0f}s | "
                    f"best val [{best_metric}]={best_val:.4f}{bt}")
            if (early_stopping_patience is not None
                    and epochs_without_improvement >= early_stopping_patience):
                if verbose:
                    print_flush(f"  Early stopping at epoch {epoch+1}")
                break

    # ── Final evaluation (best weights) ───────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
        if les_module is not None and best_les_state is not None:
            les_module.load_state_dict(best_les_state)
    tr = evaluate(train_data, max_samples=500)
    va = evaluate(val_data)
    te = evaluate(test_data) if test_data else (float('nan'),) * 3
    if verbose:
        print_flush(f"\nFinal Results ({eval_metric.upper()}):")
        print_flush(f"  Train: E={tr[0]:.4f} eV/atom F={tr[1]:.4f} eV/Å S={tr[2]:.4e} eV/Å³")
        print_flush(f"  Val:   E={va[0]:.4f} eV/atom F={va[1]:.4f} eV/Å S={va[2]:.4e} eV/Å³")
        print_flush(f"  Test:  E={te[0]:.4f} eV/atom F={te[1]:.4f} eV/Å S={te[2]:.4e} eV/Å³")
        print_flush(f"Total time: {time.time()-t_start:.1f}s")
    m = eval_metric
    results = {
        f'train_energy_{m}': tr[0], f'train_force_{m}': tr[1], f'train_stress_{m}': tr[2],
        f'val_energy_{m}': va[0], f'val_force_{m}': va[1], f'val_stress_{m}': va[2],
        f'test_energy_{m}': te[0], f'test_force_{m}': te[1], f'test_stress_{m}': te[2],
        'n_params': n_params, 'n_types': n_types, 'type_map': type_map,
    }
    return model, les_module, results


if __name__ == "__main__":
    train_ecenet_xyz()
