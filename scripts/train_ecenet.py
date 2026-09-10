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


import math
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
    m_max=None,
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
    # Joint LES long-range training: E = E_sr + E_lr on one autograd graph
    # (isolated pairwise path — rMD17/MD22 molecules have no cell). NOTE on
    # units: these datasets are in kcal/mol while the LES Coulomb constant is
    # eV·Å-based, so the latent charges absorb the scale factor — they are
    # not in units of e (they are latent regardless; on eV datasets they
    # come out closer to physical).
    use_les=False,
    les_arguments=None,    # extra kwargs for upstream les.Les (see ecenet/les.py)
    # Batching
    use_graph_batch=True,
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
    n_epochs=200,
    batch_size=8,
    energy_weight=1.0,
    force_weight=1.0,
    eval_every=5,
    eval_batch_size=32,
    seed=42,
    dtype=torch.float64,
    tf32=False,              # route float32 matmuls to TF32 tensor cores (Ampere+)
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
    if les_readout is None:
        # 'edge_basis' is the LES default; short-range runs take 'sum' so the
        # model carries no unused charge-head parameters (which would also
        # break DDP's find_unused_parameters=False).
        les_readout = 'edge_basis' if use_les else 'sum'
    if optimizer_type not in ('adamw', 'adam', 'sgd'):
        raise ValueError(f"optimizer_type must be 'adamw', 'adam', or 'sgd'; got {optimizer_type!r}")
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    # TF32: on Ampere+ the step is dominated by fp32 matmuls (forward and the
    # force double-backward). Routing those to TF32 tensor cores is the cheapest
    # large speedup, but TF32 keeps only ~10 mantissa bits — A/B the val
    # force/energy MAE before trusting it. No effect under float64 (TF32 is a
    # float32-only mode), so warn rather than silently do nothing. (Same
    # switch as train_ecenet_spice / train_ecenet_xyz.)
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
        m_max=m_max,
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

    # ── LES long-range module (optional) ──────────────────────────────────
    # Upstream builds its charge MLP lazily on the first forward, so it is
    # materialised NOW (synthetic probe via the shared load_les_module dance):
    # the optimiser and any checkpoint restore below need the parameters to
    # exist. Edge-mode read-outs bypass the head (module is parameter-free).
    les_module = None
    if use_les:
        from ecenet.les import load_les_module
        les_module = load_les_module({'arguments': les_arguments}, model,
                                     device, dtype, load_state=False)
        les_module.train()

    all_params = list(model.parameters())
    if les_module is not None:
        all_params += list(les_module.parameters())
    n_params = sum(p.numel() for p in all_params if p.requires_grad)
    if verbose:
        model_name = "ECENet"
        print_flush(f"\n{model_name}: n_mp={n_mp}, {n_layers} layers/stage, l_max={l_max}, n_max={n_max}, "
                    f"embed_dim={embed_dim}, n_max_d={n_max_d}")
        print_flush(f"  n_features_per_m: {model.n_features_per_m}")
        print_flush(f"  r_cut_edge={r_cut_edge}, r_cut_neighbor={r_cut_neighbor}")
        print_flush(f"  Trainable parameters: {n_params}"
                    + (" (incl. LES charge head)" if use_les else ""))

    # ── Optimiser ─────────────────────────────────────────────────────────
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=weight_decay)
    else:   # 'sgd'
        optimizer = torch.optim.SGD(all_params, lr=lr, weight_decay=weight_decay,
                                    momentum=0.9)
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
        if verbose:
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
    best_val_metric = float('inf')
    best_test_e_mae = float('nan')
    best_test_f_mae = float('nan')
    best_state = None
    best_les_state = None
    if checkpoint_path is not None and __import__('os').path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        if use_les != ('les' in ckpt):
            raise ValueError(
                f"Checkpoint at {checkpoint_path} was trained with "
                f"use_les={'les' in ckpt}, but this run has use_les={use_les}.")
        if les_module is not None:
            les_module.load_state_dict(ckpt['les']['state_dict'])
            best_les_state = ckpt['les'].get('best_state')
        if not reset_optimizer:
            optimizer.load_state_dict(ckpt['optimizer'])
            # Open-loop schedules have no state (the LR is a function of the
            # epoch), and saved state may come from a different schedule.
            if scheduler is not None and ckpt.get('scheduler') is not None:
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
        out_les = (None if les_module is None else {
            'arguments': les_arguments,
            'state_dict': les_module.state_dict(),
            'best_state': best_les_state,
        })
        torch.save({
            **({'les': out_les} if out_les is not None else {}),
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
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

    def batch_energies(pos_b, topo):
        """Total energies for a batch: E_sr, plus E_lr under use_les.

        The LES call is batched (SPICE's recipe): one call over the
        concatenated atoms with a structure-index vector, cell=None → the
        vectorized isolated pairwise path. Everything stays on one autograd
        graph, so the force grads below cover E_lr with no extra code.
        """
        if les_module is None:
            return model.forward_batch(pos_b, atom_types, topology=topo)
        e_sr, l0_list = model.forward_batch(pos_b, atom_types, topology=topo,
                                            return_embeddings=True, l0_only=True)
        l0 = torch.cat(l0_list, dim=0)
        pos = torch.cat(pos_b, dim=0)
        batch_vec = torch.cat([
            torch.full((p.shape[0],), b, dtype=torch.long, device=p.device)
            for b, p in enumerate(pos_b)])
        return e_sr + les_module(l0, pos, batch=batch_vec,
                                 n_struct=len(pos_b), **model.les_flags)

    def _train_mode(train):
        model.train(train)
        if les_module is not None:
            les_module.train(train)

    # ── Evaluation ────────────────────────────────────────────────────────
    def evaluate(pos_list, frc_list, eng_target, topo_split=None, max_samples=None):
        """Returns (energy_MAE, force_MAE, loss) on the given set.

        loss matches the training objective exactly:
          energy_weight · mean[((E_pred - E_tgt)/n_atoms)²]
          + force_weight · mean_structures[ mean((-grad - F)²) ]
        (MSE, per-atom energy), so it can be used as the 'weighted' best metric.
        """
        _train_mode(False)
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
                eng_b = batch_energies(pos_b, topo)
                grads = torch.autograd.grad(eng_b.sum(), pos_b)
            for k, i in enumerate(batch):
                e_err = eng_b[k] - eng_target[i]
                f_err = -grads[k] - frc_list[i]
                energy_mae   += e_err.abs().item()
                force_mae    += f_err.abs().sum().item()
                force_count  += frc_list[i].numel()
                energy_se_pa += loss_fn(eng_b[k] / n_atoms, eng_target[i] / n_atoms).item()
                force_se     += loss_fn(-grads[k], frc_list[i]).item()
        _train_mode(True)
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
        # Open-loop schedules set this epoch's LR up front; 'plateau' instead
        # adjusts it below, after the val step.
        if scheduler is None:
            for pg in optimizer.param_groups:
                pg['lr'] = open_loop_lr(epoch)
        _train_mode(True)
        epoch_loss = 0.0
        perm = np.random.permutation(n_train)
        n_batches = (n_train + batch_size - 1) // batch_size

        for b in range(n_batches):
            batch_indices = perm[b * batch_size:(b + 1) * batch_size]
            optimizer.zero_grad()

            pos_b   = [pos_train[i].detach().clone().requires_grad_(True) for i in batch_indices]
            eng_tgt = torch.stack([eng_train[i] for i in batch_indices])
            topo    = pick_topology(train_topo, batch_indices)

            eng_pred = batch_energies(pos_b, topo)

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
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip)
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

            if scheduler is not None:

                scheduler.step(cur_val_metric)

            if cur_val_metric < best_val_metric:
                best_val_metric = cur_val_metric
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                if les_module is not None:
                    best_les_state = {k: v.clone()
                                      for k, v in les_module.state_dict().items()}
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
        if les_module is not None and best_les_state is not None:
            les_module.load_state_dict(best_les_state)

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
        'les_module': les_module,   # None unless use_les
    }
