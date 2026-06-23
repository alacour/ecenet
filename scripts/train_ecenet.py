"""Training for ECENet on rMD17 / MD22 single-molecule datasets.

Import and call ``train_ecenet`` directly — all settings are keyword arguments
(message passing via n_mp>=2; n_mp=1 → plain ECENet baseline):

    from train_ecenet import train_ecenet
    model, results = train_ecenet(molecule='ethanol', n_train=950, n_val=50, n_test=50,
                                  r_cut_edge=5.0, r_cut_neighbor=5.0,
                                  l_max=2, n_max=8, embed_dim=16, n_epochs=200,
                                  n_mp=2)            # n_mp>=2 turns on message passing

See the train_ecenet() signature for the full set of arguments.
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import time

import numpy as np
import torch
import torch.nn as nn

from ecenet import ECENet, elements

# ---------------------------------------------------------------------------
# Topology precomputation (edge / neighbour lists, non-periodic)
# ---------------------------------------------------------------------------

def precompute_topology(positions_list, r_cut_edge, r_cut_neighbor):
    """Precompute directed edge and neighbour index lists per structure (no grad).

    Non-periodic (rMD17 / MD22): an O(N²) distance matrix per frame. Returns a list
    of dicts with 'edge_i'/'edge_j' (edge endpoints) and 'nb_src'/'nb_dst'
    (ACE-basis neighbour pairs). (Periodic systems are handled by train_ecenet_mptrj.py.)
    """
    cache = []
    with torch.no_grad():
        for pos in positions_list:
            p = pos.detach()
            diff = p.unsqueeze(0) - p.unsqueeze(1)               # (N, N, 3)
            dist_mat = torch.sqrt((diff ** 2).sum(-1) + 1e-30)   # (N, N)
            edge_i, edge_j = ((dist_mat < r_cut_edge) & (dist_mat > 1e-10)).nonzero(as_tuple=True)
            nb_src, nb_dst = ((dist_mat < r_cut_neighbor) & (dist_mat > 1e-10)).nonzero(as_tuple=True)
            cache.append({'edge_i': edge_i, 'edge_j': edge_j,
                          'nb_src': nb_src, 'nb_dst': nb_dst})
    return cache


def get_fixed_topology(cache):
    """Return the shared topology dict if all structures share identical topology,
    otherwise return None (signals variable-topology fallback).
    """
    if not cache:
        return None
    ref = cache[0]
    for topo in cache[1:]:
        for k in ('edge_i', 'edge_j', 'nb_src', 'nb_dst'):
            if not torch.equal(topo[k], ref[k]):
                return None
    return ref



def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

RMD17_MOLECULES = ['ethanol', 'malonaldehyde', 'uracil', 'benzene', 'toluene',
                   'naphthalene', 'salicylic', 'aspirin', 'paracetamol', 'azobenzene']
MD22_MOLECULES  = ['Ac-Ala3-NHMe', 'DHA', 'stachyose', 'AT-AT', 'AT-AT-CG-CG',
                   'buckyball-catcher', 'double-walled_nanotube']


def load_data(molecule, data_dir, n_train, n_val, n_test, dtype, device, seed):
    """Load rMD17 or MD22 dataset, split, and return tensors."""
    if molecule in RMD17_MOLECULES:
        data_dir = data_dir or "rmd17/npz_data"
        data = np.load(f"{data_dir}/rmd17_{molecule}.npz")
        positions  = data['coords']
        forces     = data['forces']
        energies   = data['energies']
        atomic_numbers = data['nuclear_charges']
    elif molecule in MD22_MOLECULES:
        data_dir = data_dir or "md22"
        data = np.load(f"{data_dir}/md22_{molecule}.npz")
        positions  = data['R']
        forces     = data['F']
        energies   = data['E'].reshape(-1)   # MD22 stores E as (n, 1)
        atomic_numbers = data['z']
    else:
        raise ValueError(f"Unknown molecule '{molecule}'. "
                         f"rMD17: {RMD17_MOLECULES}  MD22: {MD22_MOLECULES}")

    n_total = len(positions)
    np.random.seed(seed)
    torch.manual_seed(seed)
    idx = np.random.permutation(n_total)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:n_train + n_val + n_test]

    def _to_tensors(indices):
        pos = [torch.tensor(positions[i], dtype=dtype, device=device) for i in indices]
        frc = [torch.tensor(forces[i],    dtype=dtype, device=device) for i in indices]
        eng = torch.tensor(np.array([energies[i] for i in indices]), dtype=dtype, device=device)
        return pos, frc, eng

    pos_train, frc_train, eng_train = _to_tensors(train_idx)
    pos_val,   frc_val,   eng_val   = _to_tensors(val_idx)
    pos_test,  frc_test,  eng_test  = _to_tensors(test_idx)

    # Zero-mean energies using training set
    energy_mean = eng_train.mean()
    eng_train = eng_train - energy_mean
    eng_val   = eng_val   - energy_mean
    eng_test  = eng_test  - energy_mean

    # Atom types — compact dense map over the elements present (shared helper).
    type_to_idx = elements.build_type_map(atomic_numbers)
    atom_types  = torch.tensor([type_to_idx[int(z)] for z in atomic_numbers],
                                dtype=torch.long, device=device)
    type_names  = [elements.symbol(z) for z in type_to_idx]
    n_types     = len(type_to_idx)

    return (pos_train, frc_train, eng_train,
            pos_val,   frc_val,   eng_val,
            pos_test,  frc_test,  eng_test,
            atom_types, n_types, type_names, len(atomic_numbers), type_to_idx,
            energy_mean.item())


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_ecenet(
    molecule='Ac-Ala3-NHMe',
    data_dir=None,
    n_train=950,
    n_val=50,
    n_test=100,
    # Geometry
    r_cut_edge=5.0,
    r_cut_neighbor=4.0,
    l_max=3,
    n_max=4,
    # Architecture
    embed_dim=16,
    n_layers=2,
    n_max_d=8,
    cutoff_type='cosine',
    activation='silu',
    use_nonlinearity=True,
    n_grid=None,
    output_hidden_dims=None,
    analytic_ace_basis=True,
    n_dist_embed=0,
    m_max=None,
    edge_type_nonlin=False,
    edge_type_linear=False,
    edge_type_output=False,
    # Message passing
    n_mp=1,
    n_dist_basis=8,
    # Batching
    use_graph_batch=True,
    # Optimiser
    lr=1e-3,
    weight_decay=1e-5,
    grad_clip=None,
    scheduler_patience=10,
    early_stopping_patience=None,
    # Training
    n_epochs=200,
    batch_size=8,
    energy_weight=1.0,
    force_weight=1.0,
    eval_every=5,
    eval_batch_size=32,
    seed=42,
    dtype=torch.float64,
    device=None,
    checkpoint_path=None,
    reset_optimizer=False,
    best_metric='weighted',
    optimizer_type='adamw',
    loss_type='mse',
    huber_delta=0.01,
    verbose=True,
):
    # Validate best_metric early so a bad value fails fast.
    if best_metric not in ('force', 'energy', 'weighted'):
        raise ValueError(f"best_metric must be 'force', 'energy', or 'weighted'; got {best_metric!r}")
    if loss_type not in ('mse', 'huber'):
        raise ValueError(f"loss_type must be 'mse' or 'huber'; got {loss_type!r}")

    # Per-element reduced loss applied to both the per-atom energy error and
    # the force error (the 'weighted' val metric uses the same function, so it
    # stays consistent with the training objective).
    def loss_fn(pred, tgt):
        if loss_type == 'huber':
            return nn.functional.huber_loss(pred, tgt, delta=huber_delta)
        return ((pred - tgt) ** 2).mean()
    if optimizer_type not in ('adamw', 'adam', 'sgd'):
        raise ValueError(f"optimizer_type must be 'adamw', 'adam', or 'sgd'; got {optimizer_type!r}")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    # ── Data ──────────────────────────────────────────────────────────────
    (pos_train, frc_train, eng_train,
     pos_val,   frc_val,   eng_val,
     pos_test,  frc_test,  eng_test,
     atom_types, n_types, type_names, n_atoms, type_to_idx,
     energy_mean) = load_data(
        molecule, data_dir, n_train, n_val, n_test, dtype, device, seed)

    if verbose:
        print_flush(f"Dataset: {molecule} ({n_atoms} atoms, types: {type_names})")
        print_flush(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
        print_flush(f"Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = ECENet(
        n_types=n_types,
        r_cut_edge=r_cut_edge,
        r_cut_neighbor=r_cut_neighbor,
        l_max=l_max,
        n_max=n_max,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_max_d=n_max_d,
        cutoff_type=cutoff_type,
        activation=activation,
        use_nonlinearity=use_nonlinearity,
        n_grid=n_grid,
        output_hidden_dims=output_hidden_dims,
        analytic_ace_basis=analytic_ace_basis,
        n_dist_embed=n_dist_embed,
        m_max=m_max,
        edge_type_nonlin=edge_type_nonlin,
        edge_type_linear=edge_type_linear,
        edge_type_output=edge_type_output,
        n_mp=n_mp,
        n_dist_basis=n_dist_basis,
    )
    if dtype == torch.float64:
        model = model.double()
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        model_name = "ECENet"
        print_flush(f"\n{model_name}: n_mp={n_mp}, {n_layers} layers/stage, l_max={l_max}, n_max={n_max}, "
                    f"embed_dim={embed_dim}, n_max_d={n_max_d}")
        print_flush(f"  n_features_per_m: {model.n_features_per_m}")
        print_flush(f"  r_cut_edge={r_cut_edge}, r_cut_neighbor={r_cut_neighbor}")
        print_flush(f"  Trainable parameters: {n_params}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:   # 'sgd'
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                                    momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience)

    # ── Checkpoint restore ────────────────────────────────────────────────
    start_epoch = 0
    best_val_metric = float('inf')
    best_test_e_mae = float('nan')
    best_test_f_mae = float('nan')
    best_state = None
    if checkpoint_path is not None and __import__('os').path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        if not reset_optimizer:
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        # Backward compat: old checkpoints only have 'best_val_force_mae' and
        # implicitly used force-MAE as the best criterion. Newer checkpoints
        # also store 'best_metric' and 'best_val_metric'.
        saved_best_metric = ckpt.get('best_metric', 'force')
        saved_best_value  = ckpt.get('best_val_metric', ckpt['best_val_force_mae'])
        if saved_best_metric == best_metric:
            best_val_metric = saved_best_value
            best_state = ckpt['best_state']
            reset_msg = ""
        else:
            # Metric changed since last run: the saved best value is on a
            # different scale, so reset the comparison and let the next
            # improvement under the new metric set a fresh best_state.
            best_val_metric = float('inf')
            best_state = None
            reset_msg = (f" (metric changed: {saved_best_metric!r}→{best_metric!r}, "
                         f"best reset)")
        if verbose:
            opt_msg = " (optimizer reset)" if reset_optimizer else ""
            best_str = (f"{saved_best_value:.4f}" if saved_best_metric == best_metric
                        else "inf")
            print_flush(f"Resumed from checkpoint: epoch {ckpt['epoch']}, "
                        f"best val [{best_metric}]={best_str}{opt_msg}{reset_msg}")

    def save_checkpoint(epoch):
        if checkpoint_path is None:
            return
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            # Legacy key (kept for backward compat with old readers); now
            # stores the best value of whichever metric was tracked.
            'best_val_force_mae': best_val_metric,
            'best_val_metric':    best_val_metric,
            'best_metric':        best_metric,
            'best_state': best_state,
            'hparams': dict(
                n_types=n_types,
                r_cut_edge=r_cut_edge, r_cut_neighbor=r_cut_neighbor,
                l_max=l_max, n_max=n_max, embed_dim=embed_dim,
                n_layers=n_layers, n_max_d=n_max_d, n_grid=n_grid,
                cutoff_type=cutoff_type, activation=activation,
                use_nonlinearity=use_nonlinearity,
                output_hidden_dims=output_hidden_dims,
                analytic_ace_basis=analytic_ace_basis,
                n_dist_embed=n_dist_embed,
                n_mp=n_mp, n_dist_basis=n_dist_basis,
            ),
            # molecule-specific element mapping: {symbol: type_index}
            'element_to_type': elements.to_element_to_type(type_to_idx),
            # rMD17/MD22 data is in kcal/mol — calculator needs this to convert to eV
            'energy_units':   'kcal/mol',
            # mean energy subtracted during training (kcal/mol); add back for absolute energies
            'energy_mean':    energy_mean,
        }, checkpoint_path)

    # ── Topology precomputation ───────────────────────────────────────────
    fixed_topo = None
    train_topo = val_topo = test_topo = None
    if use_graph_batch:
        if verbose:
            print_flush("Precomputing topology...", end=" ")
        all_pos = pos_train + pos_val + pos_test
        topo_cache = precompute_topology(all_pos, r_cut_edge, r_cut_neighbor)

        train_topo = topo_cache[:n_train]
        val_topo   = topo_cache[n_train:n_train + n_val]
        test_topo  = topo_cache[n_train + n_val:]
        fixed_topo = get_fixed_topology(train_topo)
        if verbose:
            if fixed_topo is not None:
                print_flush(f"fixed topology ({fixed_topo['edge_i'].shape[0]} edges) "
                            "— vectorized path enabled")
            else:
                print_flush("variable topology — per-structure loop")

    def pick_topology(topo_split, batch_indices):
        if fixed_topo is not None:
            return fixed_topo
        if topo_split is not None:
            return [topo_split[i] for i in batch_indices]
        return None

    # ── Evaluation ────────────────────────────────────────────────────────
    def evaluate(pos_list, frc_list, eng_target, topo_split=None, max_samples=None):
        """Returns (energy_MAE, force_MAE, loss) on the given set.

        loss matches the training objective exactly:
          energy_weight · mean[((E_pred - E_tgt)/n_atoms)²]
          + force_weight · mean_structures[ mean((-grad - F)²) ]
        (MSE, per-atom energy), so it can be used as the 'weighted' best metric.
        """
        model.eval()
        idx = (list(np.random.choice(len(pos_list), min(max_samples, len(pos_list)), replace=False))
               if max_samples is not None else list(range(len(pos_list))))
        energy_mae = 0.0
        force_mae  = 0.0
        force_count = 0
        energy_se_pa = 0.0   # Σ loss_fn((E_err)/n_atoms)  — per-atom energy loss numerator
        force_se     = 0.0   # Σ loss_fn(F_err)            — per-structure force loss numerator
        for start in range(0, len(idx), eval_batch_size):
            batch = idx[start:start + eval_batch_size]
            pos_b = [pos_list[i].detach().clone().requires_grad_(True) for i in batch]
            topo  = pick_topology(topo_split, batch)
            with torch.enable_grad():
                eng_b = model.forward_batch(pos_b, atom_types, topology=topo)
                grads = torch.autograd.grad(eng_b.sum(), pos_b)
            for k, i in enumerate(batch):
                e_err = eng_b[k] - eng_target[i]
                f_err = -grads[k] - frc_list[i]
                energy_mae   += e_err.abs().item()
                force_mae    += f_err.abs().sum().item()
                force_count  += frc_list[i].numel()
                energy_se_pa += loss_fn(eng_b[k] / n_atoms, eng_target[i] / n_atoms).item()
                force_se     += loss_fn(-grads[k], frc_list[i]).item()
        model.train()
        n = len(idx)
        loss = energy_weight * (energy_se_pa / n) + force_weight * (force_se / n)
        return energy_mae / n, force_mae / force_count, loss

    # ── Training loop ─────────────────────────────────────────────────────
    if verbose:
        print_flush(f"\nTraining for {n_epochs} epochs (batch={batch_size}, "
                    f"lr={lr}, E-weight={energy_weight}, F-weight={force_weight})")

    epochs_without_improvement = 0
    t_start = time.time()

    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0.0
        perm = np.random.permutation(n_train)
        n_batches = (n_train + batch_size - 1) // batch_size

        for b in range(n_batches):
            batch_indices = perm[b * batch_size:(b + 1) * batch_size]
            optimizer.zero_grad()

            pos_b   = [pos_train[i].detach().clone().requires_grad_(True) for i in batch_indices]
            eng_tgt = torch.stack([eng_train[i] for i in batch_indices])
            topo    = pick_topology(train_topo, batch_indices)

            eng_pred = model.forward_batch(pos_b, atom_types, topology=topo)

            energy_loss = loss_fn(eng_pred / n_atoms, eng_tgt / n_atoms)

            if force_weight > 0:
                frc_grads = torch.autograd.grad(eng_pred.sum(), pos_b, create_graph=True)
                force_loss = sum(
                    loss_fn(-g, frc_train[batch_indices[k]])
                    for k, g in enumerate(frc_grads)
                ) / len(batch_indices)
            else:
                force_loss = 0.0

            loss = energy_weight * energy_loss + force_weight * force_loss
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= n_batches

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            train_e_mae, train_f_mae, _        = evaluate(pos_train, frc_train, eng_train,
                                                          topo_split=train_topo, max_samples=100)
            val_e_mae,   val_f_mae,   val_loss = evaluate(pos_val,   frc_val,   eng_val,
                                                          topo_split=val_topo)

            # Compute the chosen comparison metric.
            if best_metric == 'force':
                cur_val_metric = val_f_mae
            elif best_metric == 'energy':
                cur_val_metric = val_e_mae
            else:   # 'weighted' → the actual training loss evaluated on the val set
                cur_val_metric = val_loss

            scheduler.step(cur_val_metric)

            if cur_val_metric < best_val_metric:
                best_val_metric = cur_val_metric
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
                best_test_e_mae, best_test_f_mae, _ = evaluate(pos_test, frc_test, eng_test,
                                                               topo_split=test_topo)
            else:
                epochs_without_improvement += 1

            save_checkpoint(epoch)

            if verbose:
                elapsed = time.time() - t_start
                lr_now = optimizer.param_groups[0]['lr']
                print_flush(f"  Epoch {epoch+1:3d}: loss={epoch_loss:.4f} | "
                            f"train E={train_e_mae:.4f} F={train_f_mae:.4f} | "
                            f"val E={val_e_mae:.4f} F={val_f_mae:.4f} | "
                            f"lr={lr_now:.1e} | {elapsed:.0f}s | "
                            f"best val [{best_metric}]={best_val_metric:.4f} "
                            f"[test E={best_test_e_mae:.4f} F={best_test_f_mae:.4f}]")

            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                if verbose:
                    print_flush(f"  Early stopping at epoch {epoch+1}")
                break

    # ── Final evaluation ──────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    train_e_mae, train_f_mae, _ = evaluate(pos_train, frc_train, eng_train, topo_split=train_topo)
    val_e_mae,   val_f_mae,   _ = evaluate(pos_val,   frc_val,   eng_val,   topo_split=val_topo)
    test_e_mae,  test_f_mae,  _ = evaluate(pos_test,  frc_test,  eng_test,  topo_split=test_topo)
    total_time = time.time() - t_start

    if verbose:
        print_flush("\nFinal Results (MAE):")
        print_flush(f"  Train: E={train_e_mae:.4f} kcal/mol, F={train_f_mae:.4f} kcal/mol/Å")
        print_flush(f"  Val:   E={val_e_mae:.4f} kcal/mol, F={val_f_mae:.4f} kcal/mol/Å")
        print_flush(f"  Test:  E={test_e_mae:.4f} kcal/mol, F={test_f_mae:.4f} kcal/mol/Å")
        print_flush(f"Total time: {total_time:.1f}s")

    return model, {
        'train_energy_mae': train_e_mae, 'train_force_mae': train_f_mae,
        'val_energy_mae':   val_e_mae,   'val_force_mae':   val_f_mae,
        'test_energy_mae':  test_e_mae,  'test_force_mae':  test_f_mae,
        'n_params': n_params, 'time': total_time,
    }
