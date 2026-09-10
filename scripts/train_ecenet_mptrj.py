"""Training script for ECENet on the Materials Project trajectory (MPtrj) dataset.

MPtrj (CHGNet, 2022.9) is ~1.58M periodic DFT frames spanning ~89 elements with
per-frame energy (eV), forces (eV/Å) and stress (raw VASP, kBar). This trainer is
the **periodic + stress** analogue of ``train_ecenet_spice.py``:

  * periodic boundary conditions — uses ``model.forward_pbc`` with ASE neighbor
    lists that enumerate *all* periodic images within the cutoff (correct when
    r_cut > L/2, common for small MPtrj cells — unlike the minimum-image
    shortcut in ``ecenet.calculator._gpu_neighbor_list``).
  * stress targets — strain-autograd, identical convention to the calculator /
    ``test_stress_fd.py``: σ = (1/V)·dE/dε in eV/Å³.
  * many elements — a dynamic Z→type map built from the data.

Periodic systems routinely produce edges pointing *exactly* along a Cartesian
axis — most of all the self-image edges (i==j, separation = a lattice vector),
plus axis-aligned cells with atoms at special fractional coords. Those are the
poles of the Wigner frame. ``build_D1_from_rhat`` (features/ece_sphere.py) is
pole-safe in both forward and backward (safe-sqrt in each Gram-Schmidt chart),
so Wigner rotation trains correctly on crystals (FD-verified on MPtrj to ~1e-10).

Data: set ``train_path`` (and optional ``test_path``) in the call at the bottom
of this file. The format is auto-detected by extension (override with
``data_format``):

  *.json[.gz]   CHGNet figshare ``MPtrj_2022.9_full.json`` (pymatgen Structure
                dicts; parsed without a pymatgen dependency).
  *.parquet     MPContribs bulk parquet (best-effort; needs ``pyarrow``).
  *.xyz/.extxyz/.traj/.db (or ``data_format='ase'``) any ASE-readable periodic
                file with energy/forces/stress attached.

Usage:
    Set hyperparameters in the ``train_ecenet_mptrj(...)`` call at the bottom of
    this file (or import the function from your own driver), then launch:

        # single process
        python scripts/train_ecenet_mptrj.py

        # multi-GPU data-parallel (DDP) via torchrun
        torchrun --nproc_per_node=4 scripts/train_ecenet_mptrj.py

    Every training/model option is a keyword argument of ``train_ecenet_mptrj``.

NOTE on scaling: like the SPICE trainer this holds the whole (sub)set in memory,
including the precomputed neighbor lists. That is fine for a dev subset
(``n_train``); the full 1.5M-frame run needs lazy/on-disk loading (a follow-up).
"""

import os
import sys  # repo root on path for `import ecenet` when run as a script

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import gc
import gzip
import itertools
import json
import math
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from ecenet import ECENet, elements
from ecenet.datasets.mptrj import (
    MPtrjShardDataset,
    collate_keep_list,
    ensure_atom_counts,
    load_manifest,
    split_shards,
)

# Raw MPtrj stress is VASP stress in kBar. CHGNet's training stress (GPa) is
# -0.1 × σ_kBar (kBar→GPa with a sign flip); GPa→eV/Å³ divides by 160.21766208.
# Combined: σ[eV/Å³] = -σ_kBar / 1602.1766208. This yields the (1/V)·dE/dε sign
# convention used by the model / ecenet.calculator (no further flip).
STRESS_KBAR_TO_EVA3 = -1.0 / 1602.1766208


def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# DDP forward wrapper
# ---------------------------------------------------------------------------

class _PBCMultiForwardWrapper(nn.Module):
    """Loops model.forward_pbc over a batch of (different) periodic structures.

    DDP intercepts this module's forward so the subsequent loss.backward syncs
    gradients. Inputs are already strain-transformed by the caller (so stress
    can be obtained by differentiating the returned energies w.r.t. the strain
    leaves — including the CELLS, which the LES Ewald term depends on
    explicitly); this module only evaluates energies.

    With a LES module attached, also computes the long-range energy from the
    per-structure ``l0`` embeddings and returns ``E_sr + E_lr``. Registering
    the LES module HERE (not just in the optimizer) puts its parameters
    inside DDP's bucket reduction, and the head runs on every step, so
    ``find_unused_parameters=False`` stays valid (same pattern as the SPICE
    trainer). The LES call is batched: one call over the concatenated atoms
    with a structure-index vector and the stacked ``(B, 3, 3)`` cells →
    upstream's reciprocal-space Ewald path, per structure.
    """
    def __init__(self, model, les_module=None):
        super().__init__()
        self.model = model
        self.les = les_module
        # l0 convention (l0_is_charge / les_dipole) read off the model.
        self.les_flags = model.les_flags

    def forward(self, pos_list, types_list, edge_i_list, edge_j_list,
                shift_e_list, nb_src_list, nb_dst_list, shift_nb_list,
                cell_list=None):
        energies, l0_list = [], []
        for k in range(len(pos_list)):
            args = (pos_list[k], types_list[k],
                    edge_i_list[k], edge_j_list[k], shift_e_list[k],
                    nb_src_list[k], nb_dst_list[k], shift_nb_list[k])
            if self.les is None:
                energies.append(self.model.forward_pbc(*args))
            else:
                e, l0 = self.model.forward_pbc(*args, return_embeddings=True,
                                               l0_only=True)
                energies.append(e)
                l0_list.append(l0)
        energies = torch.stack(energies)
        if self.les is None:
            return energies
        l0 = torch.cat(l0_list, dim=0)
        pos = torch.cat(pos_list, dim=0)
        batch = torch.cat([
            torch.full((p.shape[0],), b, dtype=torch.long, device=p.device)
            for b, p in enumerate(pos_list)])
        cells = torch.stack(cell_list)          # (B, 3, 3); guarded in predict()
        return energies + self.les(l0, pos, cell=cells, batch=batch,
                                   **self.les_flags)


# ---------------------------------------------------------------------------
# Dataset loading — produces a list of plain structure dicts
# ---------------------------------------------------------------------------
# Each dict:
#   numbers   : (N,)  int   atomic numbers Z
#   positions : (N,3) float Cartesian Å
#   cell      : (3,3) float lattice vectors as rows (Å), or None
#   pbc       : bool
#   energy    : float total energy (eV)
#   forces    : (N,3) float eV/Å
#   stress    : (3,3) float raw VASP stress (kBar), or None
#   n_atoms   : int

def _infer_format(path, data_format):
    if data_format != 'auto':
        return data_format
    low = path.lower()
    if low.endswith('.json') or low.endswith('.json.gz'):
        return 'json'
    if low.endswith('.parquet'):
        return 'parquet'
    return 'ase'   # .xyz/.extxyz/.traj/.db/... handled by ase.io


def _open_maybe_gz(path):
    return gzip.open(path, 'rt') if path.lower().endswith('.gz') else open(path, 'r')


def _structure_dict_to_arrays(struct):
    """Convert a pymatgen Structure as_dict to (numbers, positions, cell).

    Handles both Cartesian ('xyz') and fractional ('abc') site coordinates
    without requiring pymatgen to be installed.
    """
    cell = np.asarray(struct['lattice']['matrix'], dtype=np.float64)  # rows = a,b,c
    sites = struct['sites']
    n = len(sites)
    numbers = np.empty(n, dtype=np.int64)
    positions = np.empty((n, 3), dtype=np.float64)
    for i, site in enumerate(sites):
        sp = site['species'][0]['element']          # e.g. 'Fe'
        numbers[i] = elements.number(sp)
        if 'xyz' in site:
            positions[i] = site['xyz']
        else:                                        # fractional → Cartesian
            positions[i] = np.asarray(site['abc'], dtype=np.float64) @ cell
    return numbers, positions, cell


def _skip_ws(f, buf, i, chunk_size, chars=' \t\r\n,'):
    """Advance i past `chars`, refilling buf from f. Returns (buf, i)."""
    while True:
        while i < len(buf) and buf[i] in chars:
            i += 1
        if i < len(buf):
            return buf, i
        more = f.read(chunk_size)
        if not more:
            return buf, i           # EOF
        buf = buf[i:] + more; i = 0  # compact + refill


def _raw_decode_grow(dec, f, buf, i, chunk_size):
    """raw_decode a JSON value at buf[i:], growing buf from f until it parses.
    Returns (obj, buf, end_index)."""
    while True:
        try:
            obj, j = dec.raw_decode(buf, i)
            return obj, buf, j
        except json.JSONDecodeError:
            more = f.read(chunk_size)
            if not more:
                raise               # genuinely malformed / truncated
            buf = buf + more


def _stream_json_materials(path, chunk_size=1 << 22):
    """Yield (mp_id, frames_dict) for each top-level entry of the MPtrj JSON,
    parsing ONE material at a time.

    The 12 GB file does not fit in RAM as parsed Python objects, so we scan the
    outer ``{key: value, ...}`` object incrementally and json-decode each value
    on its own (C-accelerated via JSONDecoder.raw_decode). Memory stays bounded
    by the largest single material; ``max_structures`` callers can stop early.
    """
    dec = json.JSONDecoder()
    with _open_maybe_gz(path) as f:
        buf, i = _skip_ws(f, '', 0, chunk_size, ' \t\r\n')
        if i >= len(buf) or buf[i] != '{':
            return
        i += 1
        while True:
            buf, i = _skip_ws(f, buf, i, chunk_size)           # ws + commas
            if i >= len(buf) or buf[i] == '}':
                return
            key, buf, i = _raw_decode_grow(dec, f, buf, i, chunk_size)   # "mp-id"
            buf, i = _skip_ws(f, buf, i, chunk_size, ' \t\r\n')
            if i < len(buf) and buf[i] == ':':
                i += 1
            buf, i = _skip_ws(f, buf, i, chunk_size, ' \t\r\n')
            val, buf, i = _raw_decode_grow(dec, f, buf, i, chunk_size)   # frames dict
            yield key, val
            buf = buf[i:]; i = 0                                # free processed text


def _load_json(path, energy_key, max_structures, verbose):
    """Parse the CHGNet figshare MPtrj JSON: {mp_id: {frame_key: frame}} (streamed)."""
    structures = []
    t0 = time.time()
    for mp_id, frames in _stream_json_materials(path):
        for frame_key, fr in frames.items():
            struct = fr.get('structure')
            if struct is None:
                continue
            numbers, positions, cell = _structure_dict_to_arrays(struct)

            energy = fr.get(energy_key)
            if energy is None:   # fall back across the known energy keys
                for k in ('corrected_total_energy', 'uncorrected_total_energy', 'energy'):
                    if fr.get(k) is not None:
                        energy = fr[k]
                        break
            if energy is None:
                continue

            forces = fr.get('force', fr.get('forces'))
            if forces is None:
                continue
            forces = np.asarray(forces, dtype=np.float64)

            stress = fr.get('stress')
            stress = np.asarray(stress, dtype=np.float64) if stress is not None else None

            structures.append({
                'numbers': numbers, 'positions': positions, 'cell': cell,
                'pbc': True, 'energy': float(energy), 'forces': forces,
                'stress': stress, 'n_atoms': len(numbers),
                'mp_id': mp_id,   # top-level key: groups a material's trajectory frames
            })
            if max_structures is not None and len(structures) >= max_structures:
                if verbose:
                    print_flush(f"  Parsed {len(structures):,} frames "
                                f"({time.time()-t0:.0f}s); stopping at max_structures")
                return structures
            if verbose and len(structures) % 100000 == 0:
                print_flush(f"  Parsed {len(structures):,} frames ({time.time()-t0:.0f}s)...")
    return structures


def _load_parquet(path, energy_key, max_structures, verbose):
    """Best-effort reader for the MPContribs MPtrj parquet bulk file.

    Column names vary by export; we try the common ones. If the structure is
    stored as a JSON string of a pymatgen dict, it is parsed like the JSON path.
    Adjust here once the exact schema of the downloaded file is known.
    """
    import pandas as pd  # pyarrow-backed

    df = pd.read_parquet(path)
    cols = {c.lower(): c for c in df.columns}

    def col(*names):
        for nm in names:
            if nm in cols:
                return cols[nm]
        return None

    c_struct = col('structure', 'atoms')
    c_energy = col(energy_key, 'corrected_total_energy', 'uncorrected_total_energy', 'energy')
    c_force  = col('force', 'forces')
    c_stress = col('stress')
    if c_struct is None or c_energy is None or c_force is None:
        raise ValueError(
            f"Could not locate required columns in {path}. Found: {list(df.columns)}. "
            f"Pass a converted file or extend _load_parquet for this schema.")

    c_mpid = col('mp_id', 'material_id', 'mpid')
    structures = []
    for _, row in df.iterrows():
        struct = row[c_struct]
        if isinstance(struct, (str, bytes)):
            struct = json.loads(struct)
        numbers, positions, cell = _structure_dict_to_arrays(struct)
        forces = np.asarray(row[c_force], dtype=np.float64).reshape(-1, 3)
        stress = np.asarray(row[c_stress], dtype=np.float64) if c_stress else None
        if stress is not None:
            stress = stress.reshape(3, 3)
        structures.append({
            'numbers': numbers, 'positions': positions, 'cell': cell,
            'pbc': True, 'energy': float(row[c_energy]), 'forces': forces,
            'stress': stress, 'n_atoms': len(numbers),
            'mp_id': row[c_mpid] if c_mpid else None,
        })
        if max_structures is not None and len(structures) >= max_structures:
            break
    return structures


def _load_ase(path, max_structures, verbose):
    """Read any ASE-supported file; pull energy/forces/stress defensively."""
    from ase.io import iread

    structures = []
    for atoms in iread(path):
        numbers = atoms.get_atomic_numbers().astype(np.int64)
        positions = atoms.get_positions().astype(np.float64)
        has_cell = bool(atoms.cell.any())
        cell = np.asarray(atoms.get_cell()).astype(np.float64) if has_cell else None

        info = atoms.info
        energy = None
        for k in ('energy', 'REF_energy', 'DFT_energy', 'TotEnergy'):
            if k in info:
                energy = float(info[k]); break
        if energy is None:
            try:
                energy = float(atoms.get_potential_energy())
            except Exception:
                continue

        forces = None
        for k in ('forces', 'REF_forces', 'DFT_forces'):
            if k in atoms.arrays:
                forces = np.asarray(atoms.arrays[k], dtype=np.float64); break
        if forces is None:
            try:
                forces = atoms.get_forces().astype(np.float64)
            except Exception:
                continue

        stress = None
        for k in ('stress', 'REF_stress', 'DFT_stress'):
            if k in info:
                s = np.asarray(info[k], dtype=np.float64)
                stress = _voigt_to_3x3(s) if s.shape == (6,) else s.reshape(3, 3)
                break

        structures.append({
            'numbers': numbers, 'positions': positions, 'cell': cell,
            'pbc': bool(atoms.pbc.any()), 'energy': energy, 'forces': forces,
            'stress': stress, 'n_atoms': len(numbers),
            'mp_id': info.get('mp_id', info.get('material_id')),
        })
        if max_structures is not None and len(structures) >= max_structures:
            break
    return structures


def _voigt_to_3x3(v):
    return np.array([[v[0], v[5], v[4]],
                     [v[5], v[1], v[3]],
                     [v[4], v[3], v[2]]], dtype=np.float64)


def load_mptrj(path, data_format='auto', energy_key='corrected_total_energy',
               max_structures=None, verbose=True):
    fmt = _infer_format(path, data_format)
    if verbose:
        print_flush(f"Loading {path}  (format={fmt})...")
    if fmt == 'json':
        return _load_json(path, energy_key, max_structures, verbose)
    if fmt == 'parquet':
        return _load_parquet(path, energy_key, max_structures, verbose)
    return _load_ase(path, max_structures, verbose)


# ---------------------------------------------------------------------------
# Train/val split, grouped by material (mp_id) to avoid trajectory leakage
# ---------------------------------------------------------------------------

def split_by_material(structures, val_frac, seed):
    """Split `structures` into (train, val) by holding out a fraction of whole
    materials, so all frames of a given mp_id stay in one split.

    Guarantees every element present in the data appears in TRAIN at least once
    (otherwise its embedding / atomic-energy / energy-reference parameters never
    get a training gradient, and the train-built type map would miss it): any val
    material holding the sole copy of an element is pulled back into train.

    Deterministic (sorted material keys + seeded permutation) → all DDP ranks
    agree. Structures without an 'mp_id' fall back to their own index as the
    group key (i.e. a per-frame split) — fine for non-trajectory data.
    """
    from collections import defaultdict
    groups, elems_of = defaultdict(list), {}
    for i, s in enumerate(structures):
        key = s.get('mp_id')
        groups[key if key is not None else f'__idx_{i}'].append(i)

    mat_keys = sorted(groups.keys(), key=str)
    for key in mat_keys:
        elems_of[key] = {int(z) for i in groups[key] for z in structures[i]['numbers']}
    all_elems = set().union(*elems_of.values()) if elems_of else set()

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(mat_keys))
    n_val_mat = int(round(val_frac * len(mat_keys)))
    n_val_mat = min(max(n_val_mat, 1), len(mat_keys) - 1) if len(mat_keys) > 1 else 0
    val_keys = {mat_keys[perm[k]] for k in range(n_val_mat)}

    # Ensure train covers all elements: pull back val materials that hold a
    # currently-uncovered element (deterministic order over mat_keys).
    train_elems = set().union(*(elems_of[k] for k in mat_keys if k not in val_keys),
                              set())
    missing = all_elems - train_elems
    for key in mat_keys:
        if not missing:
            break
        if key in val_keys and (elems_of[key] & missing):
            val_keys.discard(key)
            train_elems |= elems_of[key]
            missing = all_elems - train_elems

    train, val = [], []
    for key in mat_keys:
        (val if key in val_keys else train).extend(structures[i] for i in groups[key])
    return train, val


def split_by_frame(structures, val_frac, seed):
    """Random frame-level train/val split (the default).

    Trains on ALL materials — every compound contributes frames to train — with
    val just a small random hold-out of frames for early-stopping / LR. The val
    set is optimistically biased (correlated frames of a material can land in
    both splits), which is fine since it's operational only; the real benchmark
    is external (WBM). Still guarantees every element appears in train at least
    once (so its parameters get trained and the type map covers it).

    Deterministic (seeded) → all DDP ranks agree.
    """
    n = len(structures)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    n_val = min(max(n_val, 1), n - 1) if n > 1 else 0
    val_idx = set(perm[:n_val].tolist())

    # Guarantee every element appears in train (pull val frames back as needed).
    all_elems = {int(z) for s in structures for z in s['numbers']}
    train_elems = {int(z) for i, s in enumerate(structures)
                   if i not in val_idx for z in s['numbers']}
    missing = all_elems - train_elems
    for i in perm[:n_val]:                      # deterministic order
        if not missing:
            break
        es = {int(z) for z in structures[i]['numbers']}
        if es & missing:
            val_idx.discard(int(i))
            train_elems |= es
            missing = all_elems - train_elems

    train = [s for i, s in enumerate(structures) if i not in val_idx]
    val   = [structures[i] for i in sorted(val_idx)]
    return train, val


# ---------------------------------------------------------------------------
# Element → type mapping (dynamic, dense over elements present)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-element energy reference (least squares on composition)
# ---------------------------------------------------------------------------

def compute_energy_reference(structures, type_map):
    n_types = len(type_map)
    n = len(structures)
    A = np.zeros((n, n_types), dtype=np.float64)
    E = np.zeros(n, dtype=np.float64)
    for i, s in enumerate(structures):
        for z in s['numbers']:
            A[i, type_map[int(z)]] += 1
        E[i] = s['energy']
    e_ref, _, _, _ = np.linalg.lstsq(A, E, rcond=None)
    return e_ref


# ---------------------------------------------------------------------------
# Topology (PBC neighbor lists with Cartesian shift vectors)
# ---------------------------------------------------------------------------

# torch_neighbor_list now lives in ecenet.radial (the calculator needs it too
# for small cells); re-exported here so existing imports keep working.
from ecenet.radial import torch_neighbor_list  # noqa: E402, F401


def build_topology(positions, cell, pbc, r_cut_edge, r_cut_nb, device, dtype):
    """Directed edge/neighbor indices + Cartesian PBC shift vectors (torch).

    Returns device tensors: edge_i, edge_j, shift_e (E,3), nb_src, nb_dst,
    shift_nb (NB,3). Convention matches forward_pbc:
        diff_ij = positions[j] - positions[i] + shift   (shift = S @ cell)
    """
    pos = torch.as_tensor(positions, dtype=dtype, device=device)
    cell_t = (torch.as_tensor(cell, dtype=dtype, device=device)
              if (pbc and cell is not None) else None)
    ei, ej, shift_e  = torch_neighbor_list(pos, cell_t, float(r_cut_edge))
    ni, nj, shift_nb = torch_neighbor_list(pos, cell_t, float(r_cut_nb))
    return ei, ej, shift_e, ni, nj, shift_nb


# ---------------------------------------------------------------------------
# Convert structures → list of on-device tensor dicts (with topology)
# ---------------------------------------------------------------------------

def to_device_tensors(structures, type_map, e_ref, r_cut_edge, r_cut_nb,
                      stress_conv, dtype, device, verbose=False,
                      storage_device=None, consume=False):
    """Build per-structure tensor dicts: positions, types, energy (ref-subtracted),
    forces, stress (eV/Å³ or None), volume, and precomputed PBC topology.

    storage_device: if set (e.g. ``torch.device('cpu')``), tensors are built on
    ``device`` (fast GPU neighbor-list construction) then moved to
    ``storage_device`` for long-term storage. Use this for large datasets that
    do not fit in GPU memory — 1.5M MPtrj frames at ~30 KB/frame is ~45 GB,
    well over a 40 GB A100. Per-batch transfer to GPU happens in the training
    loop via ``_batch_to_device``.

    consume: if True, set ``structures[i] = None`` after each frame is
    tensorized so the raw dict (and its numpy arrays) can be GC'd before the
    next frame is processed. Cuts peak host memory roughly in half during the
    build (~40 GB → ~20 GB per rank for the full MPtrj run). Requires the
    caller to be the sole owner of these dicts (drop ``train_raw`` first).
    """
    sdev = storage_device if storage_device is not None else device
    move = sdev != device
    out = []
    t0 = time.time()
    for i, s in enumerate(structures):
        numbers = s['numbers']
        types_np = np.array([type_map[int(z)] for z in numbers], dtype=np.int64)

        ref = sum(e_ref[type_map[int(z)]] for z in numbers)

        # Topology built with torch on the fast device (GPU); moved to sdev below.
        ei, ej, shift_e, ni, nj, shift_nb = build_topology(
            s['positions'], s['cell'], s['pbc'], r_cut_edge, r_cut_nb, device, dtype)
        if move:
            ei, ej, shift_e = ei.to(sdev), ej.to(sdev), shift_e.to(sdev)
            ni, nj, shift_nb = ni.to(sdev), nj.to(sdev), shift_nb.to(sdev)

        cell = s['cell']
        periodic = s['pbc'] and cell is not None
        volume = abs(np.linalg.det(cell)) if periodic else 0.0
        cell_t = torch.tensor(cell, dtype=dtype, device=sdev) if periodic else None

        stress_t = None
        if s['stress'] is not None and volume > 0:
            stress_t = torch.tensor(np.asarray(s['stress']) * stress_conv,
                                    dtype=dtype, device=sdev)   # (3,3) eV/Å³

        out.append({
            'pos':     torch.tensor(s['positions'], dtype=dtype, device=sdev),
            'cell':    cell_t,   # (3,3) or None; LES Ewald needs it explicitly
            'types':   torch.tensor(types_np, dtype=torch.long, device=sdev),
            'energy':  torch.tensor(s['energy'] - ref, dtype=dtype, device=sdev),
            'forces':  torch.tensor(s['forces'], dtype=dtype, device=sdev),
            'stress':  stress_t,
            'volume':  volume,
            'edge_i':  ei, 'edge_j': ej, 'shift_e': shift_e,
            'nb_src':  ni, 'nb_dst': nj, 'shift_nb': shift_nb,
            'n_atoms': s['n_atoms'],
        })
        if consume:
            structures[i] = None      # drop the sole external ref → numpy arrays freed
        if verbose and len(out) % 50000 == 0:
            print_flush(f"  Built topology for {len(out):,} structures ({time.time()-t0:.0f}s)...")
    return out


# Tensor fields we ship to the compute device each batch. Volume/n_atoms are
# Python scalars and stay as-is.
_BATCH_TENSOR_KEYS = ('pos', 'types', 'energy', 'forces', 'stress', 'cell',
                      'edge_i', 'edge_j', 'shift_e',
                      'nb_src', 'nb_dst', 'shift_nb')


def _batch_to_device(batch, device, non_blocking=True):
    """Return a copy of ``batch`` (list of per-frame dicts) with all tensor
    fields moved to ``device``. No-op if already there. Used to keep training
    data resident on CPU and transfer only the active batch to GPU."""
    out = []
    for d in batch:
        nd = {'n_atoms': d['n_atoms'], 'volume': d['volume']}
        for k in _BATCH_TENSOR_KEYS:
            v = d.get(k)
            if isinstance(v, torch.Tensor) and v.device != device:
                v = v.to(device, non_blocking=non_blocking)
            nd[k] = v
        out.append(nd)
    return out


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_ecenet_mptrj(
    train_path='MPtrj_2022.9_full.json',
    test_path=None,
    data_format='auto',
    energy_key='corrected_total_energy',
    # Pre-tensorized shards (bypasses load + topology build; see prepare_mptrj.py).
    # When set, train_path/test_path/data_format/energy_key are ignored and
    # type_map/e_ref/r_cut_*/dtype come from the prepared manifest.
    prepared_dir=None,
    num_workers=0,           # DataLoader workers (prepared mode only)
    # Size-aware batching (prepared mode only): pack each shard's frames into
    # batches of at most this many total atoms, so memory/compute per step is
    # roughly uniform and several large crystals can no longer OOM one batch.
    # DDP-safe: per-round min-truncation keeps the batch count identical on
    # every rank (see MPtrjShardDataset). batch_size is ignored for training
    # when set (eval keeps eval_batch_size).
    max_atoms_per_batch=None,
    max_batch_count=None,    # optional cap on frames per packed batch
    bucket_sort=True,        # sort each shard by atom count before packing
    # Pre-loaded structures (bypass file loading; used by tests)
    train_structures=None,
    test_structures=None,
    # Cap frames read from disk (essential for the 12GB JSON on limited RAM;
    # the full set only fits in memory on a big-RAM cluster). None = read all.
    max_load=None,
    # Data splits (MPtrj = train+val only; test is external — WBM/Matbench).
    n_train=None,
    val_split='frame',  # 'frame': train on ALL materials (default) | 'material': hold out whole materials
    val_frac=0.05,      # fraction held out for val (of frames if 'frame', of materials if 'material')
    n_val=None,         # optional cap on number of val frames evaluated
    n_test=None,
    n_per_epoch=None,
    cycle_data=False,
    # Geometry
    r_cut_edge=5.0,
    r_cut_neighbor=4.0,
    l_max=3,
    n_max=4,
    cutoff_type='cosine',
    # Architecture (mirrors train_ecenet_spice.py)
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
    # Joint LES long-range training: E = E_sr + E_lr on one autograd graph.
    # Periodic structures use upstream's reciprocal-space Ewald, so every
    # frame needs its cell tensor — prepared shards written before cells were
    # stored must be re-prepared. Stress covers E_lr too (the strain pass
    # transforms the cell alongside positions and shifts).
    use_les=False,
    les_arguments=None,      # extra kwargs for upstream les.Les (see ecenet/les.py)
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
    stress_weight=0.0,
    stress_conv=STRESS_KBAR_TO_EVA3,
    loss='mse',
    huber_delta=0.01,
    eval_every=1,
    eval_batch_size=32,
    seed=42,
    dtype=torch.float64,
    tf32=False,              # route float32 matmuls to TF32 tensor cores (Ampere+)
    device=None,
    cpu_data=False,           # store the precomputed batch tensors on CPU and
                              # transfer per-batch to GPU; required for the
                              # full 1.5M MPtrj run (~45 GB on GPU otherwise).
    checkpoint_path=None,
    verbose=True,
    # DDP
    rank=0,
    world_size=1,
    local_rank=0,
):
    if les_readout is None:
        # 'edge_basis' is the LES default; short-range runs take 'sum' so the
        # model carries no unused charge-head parameters (which would also
        # break DDP's find_unused_parameters=False).
        les_readout = 'edge_basis' if use_les else 'sum'
    is_ddp = world_size > 1
    is_main = (rank == 0)
    verbose = verbose and is_main
    use_stress = stress_weight > 0

    if max_atoms_per_batch is not None and prepared_dir is None:
        raise ValueError(
            "max_atoms_per_batch is only supported with prepared_dir (per-shard "
            "size-aware batching); the in-memory path still batches by batch_size")

    # Under DDP every parameter must receive a gradient every step
    # (find_unused_parameters=False); a non-'sum' les_readout without use_les
    # would leave its read-out parameters unused and hang the reduction.
    if is_ddp and les_readout != 'sum' and not use_les:
        raise ValueError(f"les_readout={les_readout!r} without use_les=True "
                         "is not supported under DDP: its read-out parameters "
                         "would never receive gradients.")

    if device is None:
        device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')
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

    np.random.seed(seed)
    torch.manual_seed(seed + rank)

    # ── Prepared-shard branch (approach #3): skip load/split/tensorize ────
    use_prepared = prepared_dir is not None
    if use_prepared:
        if verbose:
            print_flush(f"Loading prepared shards from {prepared_dir}...")
        manifest, type_map, e_ref_np, all_shard_paths = load_manifest(prepared_dir)
        # Verify config compatibility — these are baked into the prepared data.
        prep_dtype = 'float32' if dtype == torch.float32 else 'float64'
        if manifest['dtype'] != prep_dtype:
            raise ValueError(f"prepared dtype={manifest['dtype']} but trainer dtype={prep_dtype} "
                             f"(re-prepare with --float32 / drop --float32 to match)")
        if abs(manifest['r_cut_edge'] - r_cut_edge) > 1e-9:
            raise ValueError(f"prepared r_cut_edge={manifest['r_cut_edge']} != "
                             f"trainer r_cut_edge={r_cut_edge}")
        if abs(manifest['r_cut_neighbor'] - r_cut_neighbor) > 1e-9:
            raise ValueError(f"prepared r_cut_neighbor={manifest['r_cut_neighbor']} != "
                             f"trainer r_cut_neighbor={r_cut_neighbor}")
        # Split shards into train/val (frame-level random; frames were globally
        # shuffled at prepare time so a shard-suffix is a uniform random sample).
        train_shard_paths, val_shard_paths = split_shards(
            all_shard_paths, val_frac=val_frac, seed=seed)
        # Cap train via n_train (round up to whole shards).
        ssz = manifest['shard_size']
        if n_train is not None:
            n_train_shards = max(1, (n_train + ssz - 1) // ssz)
            train_shard_paths = train_shard_paths[:n_train_shards]
        # n_val cap (frames): round up to whole shards likewise.
        if n_val is not None:
            n_val_shards = max(1, (n_val + ssz - 1) // ssz)
            val_shard_paths = val_shard_paths[:n_val_shards]
        n_types = len(type_map)
        n_train_actual = len(train_shard_paths) * ssz
        n_val_actual   = len(val_shard_paths) * ssz
        train_atom_counts = None
        if max_atoms_per_batch is not None:
            # Per-frame atom counts (metadata sidecar): every rank derives every
            # shard's packing — hence the aligned per-round batch counts — from
            # these alone, without loading shards. Rank 0 back-fills the file
            # for prepared dirs that predate it; the others wait, then read.
            if is_main:
                counts_by_shard = ensure_atom_counts(prepared_dir)
            if is_ddp:
                dist.barrier()
            if not is_main:
                counts_by_shard = ensure_atom_counts(prepared_dir)
            train_atom_counts = [counts_by_shard[os.path.basename(p)]
                                 for p in train_shard_paths]
        train_data = MPtrjShardDataset(train_shard_paths, rank=rank,
                                       world_size=world_size, seed=seed, shuffle=True,
                                       max_atoms_per_batch=max_atoms_per_batch,
                                       max_batch_count=max_batch_count,
                                       bucket_sort=bucket_sort,
                                       atom_counts=train_atom_counts)
        val_data   = MPtrjShardDataset(val_shard_paths,   rank=0,
                                       world_size=1, seed=seed, shuffle=False)
        test_data  = []   # external benchmark (WBM); not stored in prepared dir
        # Build a tensor e_ref aligned with predict()'s usage (subtracted into
        # 'energy' field at prepare time — nothing else to do here).
        e_ref = e_ref_np
        if verbose:
            elems = ' '.join(elements.symbol(z) for z in sorted(type_map))
            print_flush(f"Train: ~{n_train_actual:,} frames / {len(train_shard_paths)} shards"
                        f" | Val: ~{n_val_actual:,} frames / {len(val_shard_paths)} shards"
                        f" | Test: 0 frames (external)")
            print_flush(f"n_types={n_types}: {elems}")
            print_flush(f"Device: {device} | stress={'on' if use_stress else 'off'}")
            if max_atoms_per_batch is not None:
                cap = f", ≤{max_batch_count} frames" if max_batch_count else ""
                print_flush(f"Size-aware batching: ≤{max_atoms_per_batch} atoms/batch"
                            f"{cap}, bucket_sort={bucket_sort} → "
                            f"{len(train_data)} batches/rank at epoch 0")
        # Skip the entire load/split/tensorize block below.

    # ── Load data ─────────────────────────────────────────────────────────
    if use_prepared:
        pass    # branched above
    elif train_structures is None:
        train_raw = load_mptrj(train_path, data_format, energy_key,
                               max_structures=max_load, verbose=verbose)
        if verbose:
            print_flush(f"  Loaded {len(train_raw):,} training frames")
    else:
        train_raw = train_structures
        if verbose:
            print_flush(f"  Loaded {len(train_raw):,} training frames")

    if not use_prepared:
        # Optional external test set (e.g. WBM); MPtrj itself is only train+val,
        # since the real benchmark is out-of-distribution (Matbench-Discovery / WBM).
        if test_structures is not None:
            test_raw = test_structures
        elif test_path is not None:
            test_raw = load_mptrj(test_path, data_format, energy_key,
                                  max_structures=max_load, verbose=verbose)
        else:
            test_raw = []

    if not use_prepared:
        # ── Split train → train + val ─────────────────────────────────────
        # Default 'frame': random frame split → trains on ALL materials, val
        # is just an operational early-stop/LR signal (real test is external
        # WBM). 'material': hold out whole materials. Both guarantee every
        # element is represented in the train split.
        if val_split == 'material':
            train_use, val_raw = split_by_material(train_raw, val_frac, seed)
        else:
            train_use, val_raw = split_by_frame(train_raw, val_frac, seed)
        if n_train is not None:
            capped = train_use[:n_train]
            # The split's "every element appears in train" guarantee is undone
            # by the cap; rescue evicted elements from the discarded tail.
            kept = {int(z) for s in capped for z in s['numbers']}
            missing = {int(z) for s in train_use for z in s['numbers']} - kept
            for s in itertools.islice(train_use, n_train, None):
                if not missing:
                    break
                es = {int(z) for z in s['numbers']}
                if es & missing:
                    capped.append(s)
                    missing -= es
            train_use = capped
        if n_val is not None:
            val_raw = val_raw[:n_val]
        if n_test is not None:
            test_raw = test_raw[:n_test]

        # Type map built over the full loaded pool so val/test atoms always
        # have a type slot; cap-rescue ensures every type sees training grads.
        type_map = elements.build_type_map(
            z for s in (train_raw + test_raw) for z in s['numbers'])
        n_types = len(type_map)
        if verbose:
            n_atoms_list = [s['n_atoms'] for s in train_use]
            n_mat_tr = len({s.get('mp_id', id(s)) for s in train_use})
            n_mat_va = len({s.get('mp_id', id(s)) for s in val_raw})
            elems = ' '.join(elements.symbol(z) for z in sorted(type_map))
            print_flush(f"Train: {len(train_use):,} frames / {n_mat_tr:,} materials | "
                        f"Val: {len(val_raw):,} frames / {n_mat_va:,} materials | "
                        f"Test: {len(test_raw):,} frames (external)")
            print_flush(f"Atoms/struct: min={min(n_atoms_list)} max={max(n_atoms_list)} "
                        f"avg={np.mean(n_atoms_list):.1f}")
            print_flush(f"n_types={n_types}: {elems}")
            print_flush(f"Device: {device} | stress={'on' if use_stress else 'off'}")

        if verbose:
            print_flush("Computing per-element energy reference...")
        e_ref = compute_energy_reference(train_use, type_map)

        np.random.seed(seed + rank)

        if verbose:
            kind = "CPU (per-batch GPU transfer)" if cpu_data else f"compute device ({device})"
            print_flush(f"Building tensors + PBC topology... [storage={kind}]")
        storage_device = torch.device('cpu') if cpu_data else device
        del train_raw
        train_data = to_device_tensors(train_use, type_map, e_ref, r_cut_edge,
                                       r_cut_neighbor, stress_conv, dtype, device, verbose,
                                       storage_device=storage_device, consume=True)
        del train_use
        val_data   = to_device_tensors(val_raw,   type_map, e_ref, r_cut_edge,
                                       r_cut_neighbor, stress_conv, dtype, device,
                                       storage_device=storage_device, consume=True)
        del val_raw
        test_data  = to_device_tensors(test_raw,  type_map, e_ref, r_cut_edge,
                                       r_cut_neighbor, stress_conv, dtype, device,
                                       storage_device=storage_device, consume=True)
        del test_raw
        gc.collect()

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
    raw_model = model

    # ── LES long-range module (optional) ──────────────────────────────────
    # Upstream builds its charge MLP lazily on the first forward, so it is
    # materialised BEFORE the DDP wrap / optimiser / checkpoint restore via
    # the shared load_les_module dance (synthetic probe, no data needed).
    les_module = None
    if use_les:
        from ecenet.les import load_les_module
        les_module = load_les_module({'arguments': les_arguments}, model,
                                     device, dtype, load_state=False)
        les_module.train()

    all_params = list(model.parameters())
    if les_module is not None:
        all_params += list(les_module.parameters())

    if is_ddp:
        train_model = DDP(_PBCMultiForwardWrapper(model, les_module),
                          device_ids=[local_rank],
                          find_unused_parameters=False)
        # create_graph=True force/stress training yields non-contiguous grads →
        # DDP bucket-view stride mismatch. Make them contiguous (as in SPICE).
        for p in all_params:
            if p.requires_grad:
                p.register_hook(lambda g: g.contiguous())
    else:
        train_model = _PBCMultiForwardWrapper(model, les_module)

    # Plain (non-DDP) wrapper for evaluation — rank 0 calls it alone, so it must
    # not be the DDP module (whose forward expects all ranks to participate).
    eval_fwd = _PBCMultiForwardWrapper(raw_model, les_module)

    n_params = sum(p.numel() for p in all_params if p.requires_grad)
    if verbose:
        mname = "ECENet"
        print_flush(f"\n{mname}: {n_layers} layers, l_max={l_max}, n_max={n_max}, "
                    f"embed_dim={embed_dim}, n_types={n_types}"
                    + (" | LES=on" if use_les else ""))
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
    best_test = (float('nan'), float('nan'), float('nan'))
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
        best_test = ckpt.get('best_test', best_test)
        if verbose:
            print_flush(f"Resumed from epoch {ckpt['epoch']}, "
                        f"best val [weighted]={best_val_weighted:.4f}")

    def save_checkpoint(epoch):
        if checkpoint_path is None or not is_main:
            return
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
            'element_to_type': elements.to_element_to_type(type_map),  # {symbol: type_idx}
            'e_ref': e_ref,             # per-element reference energies (eV)
            'stress_conv': stress_conv,
        }, checkpoint_path)

    # ── Loss helper ───────────────────────────────────────────────────────
    def elem_loss(diff):
        if loss == 'mse':
            return diff ** 2
        if loss == 'l1':
            return diff.abs()
        abs_d = diff.abs()
        return torch.where(abs_d <= huber_delta, 0.5 * diff ** 2,
                           huber_delta * (abs_d - 0.5 * huber_delta))

    # ── Build a forward over a list of structures, returning predictions ──
    def predict(batch, create_graph, fwd):
        """Run forward_pbc over `batch` (list of data dicts) with strain leaves.

        `fwd` is the forward module to call (DDP-wrapped train_model during
        training; the plain eval_fwd during evaluation). Returns (energies,
        forces_list, stress_list) where forces/stress are None when not
        requested. `create_graph` keeps the graph for the loss backward
        (training); set False for evaluation.
        """
        pos_leaf, strain_leaf = [], []
        pos_in, shift_e_in, shift_nb_in, cell_in = [], [], [], []
        types_b, ei_b, ej_b, ni_b, nj_b = [], [], [], [], []
        for d in batch:
            cell = d.get('cell')
            if use_les and cell is None:
                raise ValueError(
                    "use_les=True needs every structure's cell tensor (the "
                    "Ewald term depends on it explicitly), but this frame has "
                    "none. Prepared shards written before cells were stored "
                    "must be re-prepared (prepare_mptrj.py now includes them).")
            p = d['pos'].detach().clone().requires_grad_(True)
            pos_leaf.append(p)
            if use_stress:
                eps = torch.zeros(3, 3, dtype=d['pos'].dtype, device=d['pos'].device,
                                  requires_grad=True)
                strain_leaf.append(eps)
                pos_in.append(p + p @ eps)
                shift_e_in.append(d['shift_e'] + d['shift_e'] @ eps)
                shift_nb_in.append(d['shift_nb'] + d['shift_nb'] @ eps)
                cell_in.append(cell + cell @ eps if cell is not None else None)
            else:
                pos_in.append(p)
                shift_e_in.append(d['shift_e'])
                shift_nb_in.append(d['shift_nb'])
                cell_in.append(cell)
            types_b.append(d['types'])
            ei_b.append(d['edge_i']); ej_b.append(d['edge_j'])
            ni_b.append(d['nb_src']); nj_b.append(d['nb_dst'])

        energies = fwd(pos_in, types_b, ei_b, ej_b, shift_e_in,
                       ni_b, nj_b, shift_nb_in, cell_in)

        forces_list = stress_list = None
        if force_weight > 0 or use_stress:
            grad_inputs = list(pos_in)
            if use_stress:
                grad_inputs = grad_inputs + strain_leaf
            # allow_unused: a structure with zero edges (lone atom in a cell
            # whose self-images all sit beyond r_cut_edge) never puts its
            # position leaf into the forward graph; strain similarly when no
            # edge crosses a periodic boundary. The physical gradient is
            # exactly zero in those cases, so substitute zeros for None.
            grads = torch.autograd.grad(energies.sum(), grad_inputs,
                                        create_graph=create_graph,
                                        allow_unused=True)
            B = len(batch)
            forces_list = [
                -grads[k] if grads[k] is not None else torch.zeros_like(pos_in[k])
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
        raw_model.train(train)
        if les_module is not None:
            les_module.train(train)

    # ── Evaluation (rank 0) ───────────────────────────────────────────────
    def _eval_batches(data):
        """Yield (already-on-device) eval batches.
        - List-of-dicts: iterate sequentially in eval_batch_size chunks.
        - MPtrjShardDataset: build a non-shuffled, non-DDP-sharded view so
          rank 0 sees the full eval set (consistent MAE across runs)."""
        if isinstance(data, MPtrjShardDataset):
            eval_ds = MPtrjShardDataset(data.shard_paths, rank=0, world_size=1,
                                        seed=data.seed, shuffle=False)
            loader = DataLoader(eval_ds, batch_size=eval_batch_size,
                                collate_fn=collate_keep_list, num_workers=0)
            for batch in loader:
                yield _batch_to_device(batch, device)
        else:
            for start in range(0, len(data), eval_batch_size):
                batch = data[start:start + eval_batch_size]
                if cpu_data:
                    batch = _batch_to_device(batch, device)
                yield batch

    def evaluate(data, max_samples=None):
        _train_mode(False)
        # For list-of-dicts we can subsample randomly (cheap random access).
        # For streaming we just truncate to the first max_samples frames.
        if not isinstance(data, MPtrjShardDataset) and max_samples is not None \
                and max_samples < len(data):
            idx = np.random.choice(len(data), max_samples, replace=False)
            data = [data[int(i)] for i in idx]

        e_abs = f_abs = s_abs = 0.0
        f_count = s_count = n = 0
        for batch in _eval_batches(data):
            if max_samples is not None and n >= max_samples:
                break
            if max_samples is not None and n + len(batch) > max_samples:
                batch = batch[:max_samples - n]
            with torch.enable_grad():
                energies, forces_list, stress_list = predict(batch, create_graph=False, fwd=eval_fwd)
            for k, d in enumerate(batch):
                e_abs += (energies[k] - d['energy']).abs().item() / d['n_atoms']
                if forces_list is not None:
                    f_abs += (forces_list[k] - d['forces']).abs().sum().item()
                    f_count += d['forces'].numel()
                if stress_list is not None and d['stress'] is not None:
                    s_abs += (stress_list[k] - d['stress']).abs().sum().item()
                    s_count += d['stress'].numel()
            n += len(batch)
        _train_mode(True)
        f_mae = f_abs / f_count if f_count else float('nan')
        s_mae = s_abs / s_count if s_count else float('nan')
        return (e_abs / n if n else float('nan')), f_mae, s_mae

    # ── Training loop ─────────────────────────────────────────────────────
    n_train_actual = (n_train_actual if use_prepared else len(train_data))
    epoch_size = n_per_epoch if n_per_epoch is not None else n_train_actual
    if verbose:
        sloss = f" S-weight={stress_weight}" if use_stress else ""
        print_flush(f"\nTraining {n_epochs} epochs (batch={batch_size}, "
                    f"epoch_size={epoch_size:,}, world_size={world_size}, lr={lr}, "
                    f"E-weight={energy_weight}, F-weight={force_weight}{sloss}, loss={loss})")

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

        rank_epoch_size = (epoch_size + world_size - 1) // world_size

        # Per-mode batch source. Prepared mode streams via DataLoader; legacy
        # mode samples indices into the in-memory list. Both yield lists of
        # per-frame dicts already on the compute device.
        if use_prepared:
            train_data.set_epoch(epoch)
            if max_atoms_per_batch is not None:
                # The dataset yields ready-packed batches whose per-rank count
                # is already DDP-aligned; batch_size does not apply.
                loader = DataLoader(train_data, batch_size=None,
                                    num_workers=num_workers, pin_memory=True)
                if n_per_epoch is not None:
                    # Approximate frame budget → batch cap. ANY shared cap is
                    # DDP-safe (per-rank totals are equal, so every rank stops
                    # at the same batch index); only the frames/epoch is
                    # approximate.
                    mean_atoms = float(np.mean([c.mean() for c in train_atom_counts]))
                    est = max(1.0, max_atoms_per_batch / mean_atoms)
                    if max_batch_count:
                        est = min(est, float(max_batch_count))
                    n_batches_target = max(1, int(round(rank_epoch_size / est)))
                else:
                    n_batches_target = None
            else:
                loader = DataLoader(train_data, batch_size=batch_size,
                                    collate_fn=collate_keep_list, num_workers=num_workers,
                                    pin_memory=True)
                n_batches_target = (rank_epoch_size + batch_size - 1) // batch_size
            def _batches():
                for b, raw in enumerate(loader):
                    if n_batches_target is not None and b >= n_batches_target:
                        return
                    yield _batch_to_device(raw, device)
        else:
            if cycle_data and epoch_size < n_train_actual:
                chunks_per_cycle = n_train_actual // epoch_size
                cycle_rng = np.random.RandomState(seed + epoch // chunks_per_cycle)
                all_idx = cycle_rng.permutation(n_train_actual)[:chunks_per_cycle * epoch_size]
                ci = epoch % chunks_per_cycle
                all_idx = all_idx[ci * epoch_size:(ci + 1) * epoch_size]
            else:
                rng = np.random.RandomState(seed + epoch)
                all_idx = rng.choice(n_train_actual, epoch_size, replace=(epoch_size > n_train_actual))
            rank_idx = all_idx[rank * rank_epoch_size:(rank + 1) * rank_epoch_size]
            n_batches_target = max(1, (len(rank_idx) + batch_size - 1) // batch_size)
            def _batches():
                for b in range(n_batches_target):
                    sel = rank_idx[b * batch_size:(b + 1) * batch_size]
                    if len(sel) == 0:
                        continue
                    batch = [train_data[i] for i in sel]
                    if cpu_data:
                        batch = _batch_to_device(batch, device)
                    yield batch

        n_batches = 0
        for batch in _batches():
            n_batches += 1
            optimizer.zero_grad()

            energies, forces_list, stress_list = predict(batch, create_graph=True, fwd=train_model)
            eng_tgt = torch.stack([d['energy'] for d in batch])
            n_atoms_b = torch.tensor([d['n_atoms'] for d in batch], dtype=dtype, device=device)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            epoch_loss += total_loss.item()

        epoch_loss /= max(1, n_batches)

        if (epoch + 1) % eval_every == 0 or epoch == 0:
            if is_ddp:
                dist.barrier()
            val_weighted_tensor = torch.tensor(float('inf'), device=device)
            if is_main:
                tr_e, tr_f, tr_s = evaluate(train_data, max_samples=200)
                # max_samples makes n_val exact in prepared mode too (the shard
                # cap only rounds up to whole shards); legacy val_data is
                # already ≤ n_val, so it's a no-op there.
                va_e, va_f, va_s = evaluate(val_data, max_samples=n_val)
                # Weighted selection metric (mirrors the training-loss weighting);
                # stress only contributes when it's part of the loss (else va_s may be NaN).
                va_weighted = energy_weight * va_e + force_weight * va_f
                if use_stress:
                    va_weighted += stress_weight * va_s
                val_weighted_tensor = torch.tensor(va_weighted, device=device)
            if is_ddp:
                dist.broadcast(val_weighted_tensor, src=0)
            if scheduler is not None:
                scheduler.step(val_weighted_tensor.item())

            if is_main:
                if va_weighted < best_val_weighted:
                    best_val_weighted = va_weighted
                    best_state = {k: v.clone() for k, v in raw_model.state_dict().items()}
                    if les_module is not None:
                        best_les_state = {k: v.clone()
                                          for k, v in les_module.state_dict().items()}
                    epochs_without_improvement = 0
                    best_test = evaluate(test_data) if test_data else best_test
                else:
                    epochs_without_improvement += 1
                should_stop = (early_stopping_patience is not None
                               and epochs_without_improvement >= early_stopping_patience)
                if is_ddp:
                    dist.broadcast(torch.tensor(1 if should_stop else 0, device=device), src=0)
            elif is_ddp:
                stop = torch.tensor(0, device=device)
                dist.broadcast(stop, src=0)
                if stop.item() == 1:
                    break

            if is_main:
                save_checkpoint(epoch)
                lr_now = optimizer.param_groups[0]['lr']
                ssfx = f" S={va_s:.4f}" if use_stress else ""
                tr_ssfx = f" S={tr_s:.4f}" if use_stress else ""
                print_flush(
                    f"  Epoch {epoch+1:3d}: loss={epoch_loss:.4f} | "
                    f"train E={tr_e:.4f} F={tr_f:.4f}{tr_ssfx} | "
                    f"val E={va_e:.4f} F={va_f:.4f}{ssfx} | "
                    f"lr={lr_now:.1e} | {time.time()-t_start:.0f}s | "
                    f"best val [weighted]={best_val_weighted:.4f} "
                    f"[test E={best_test[0]:.4f} F={best_test[1]:.4f} S={best_test[2]:.4f}]")
                if should_stop:
                    print_flush(f"  Early stopping at epoch {epoch+1}")
                    break

    # ── Final evaluation (rank 0) ─────────────────────────────────────────
    results = {}
    if is_main:
        if best_state is not None:
            raw_model.load_state_dict(best_state, strict=False)
            if les_module is not None and best_les_state is not None:
                les_module.load_state_dict(best_les_state)
        tr = evaluate(train_data, max_samples=500)
        va = evaluate(val_data, max_samples=n_val)
        te = evaluate(test_data) if test_data else (float('nan'),)*3
        print_flush("\nFinal Results (MAE):")
        print_flush(f"  Train: E={tr[0]:.4f} eV/atom F={tr[1]:.4f} eV/Å S={tr[2]:.4e} eV/Å³")
        print_flush(f"  Val:   E={va[0]:.4f} eV/atom F={va[1]:.4f} eV/Å S={va[2]:.4e} eV/Å³")
        print_flush(f"  Test:  E={te[0]:.4f} eV/atom F={te[1]:.4f} eV/Å S={te[2]:.4e} eV/Å³")
        print_flush(f"Total time: {time.time()-t_start:.1f}s")
        results = {
            'train_energy_mae': tr[0], 'train_force_mae': tr[1], 'train_stress_mae': tr[2],
            'val_energy_mae': va[0], 'val_force_mae': va[1], 'val_stress_mae': va[2],
            'test_energy_mae': te[0], 'test_force_mae': te[1], 'test_stress_mae': te[2],
            'n_params': n_params, 'n_types': n_types, 'type_map': type_map,
            'les_module': les_module,   # None unless use_les
        }

    if is_ddp:
        dist.destroy_process_group()
    return raw_model, results


# ---------------------------------------------------------------------------
# Entry point — torchrun-compatible (multi-GPU DDP)
# ---------------------------------------------------------------------------
# torchrun sets LOCAL_RANK / RANK / WORLD_SIZE in the environment; we read them
# here and hand them to train_ecenet_mptrj for DDP setup. Set hyperparameters by
# editing the call below (or import train_ecenet_mptrj from your own driver).
#
#     python scripts/train_ecenet_mptrj.py                 # single process
#     torchrun --nproc_per_node=4 scripts/train_ecenet_mptrj.py   # multi-GPU

if __name__ == "__main__":
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank       = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    train_ecenet_mptrj(rank=rank, world_size=world_size, local_rank=local_rank)
