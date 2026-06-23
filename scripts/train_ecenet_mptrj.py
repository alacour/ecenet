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
    leaves); this module only evaluates energies.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pos_list, types_list, edge_i_list, edge_j_list,
                shift_e_list, nb_src_list, nb_dst_list, shift_nb_list):
        energies = []
        for k in range(len(pos_list)):
            energies.append(self.model.forward_pbc(
                pos_list[k], types_list[k],
                edge_i_list[k], edge_j_list[k], shift_e_list[k],
                nb_src_list[k], nb_dst_list[k], shift_nb_list[k]))
        return torch.stack(energies)


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

def torch_neighbor_list(pos, cell, r_cut):
    """Vectorized all-images neighbor list (matches ASE neighbor_list('ijS')).

    Fast torch O(N²·n_shift) replacement for ASE's per-pair Python loop, runs on
    whatever device `pos` is on (GPU on the cluster). Unlike the minimum-image
    list in ecenet.calculator._gpu_neighbor_list, this enumerates *every* periodic
    image within the cutoff and includes self-image edges (i==j, S≠0) — essential
    for MPtrj's small cells where r_cut > L/2 is the norm.

    Args:
        pos:   (N, 3) tensor of Cartesian positions
        cell:  (3, 3) tensor (rows = lattice vectors), or None for non-periodic
        r_cut: cutoff radius (Å)
    Returns:
        i, j:  (M,) long tensors (directed pairs)
        shift: (M, 3) tensor of Cartesian shifts, with
               diff = pos[j] - pos[i] + shift,  0 < |diff| < r_cut
    """
    device, dtype = pos.device, pos.dtype
    if cell is None:                              # non-periodic → only the zero shift
        shift_cart = torch.zeros(1, 3, dtype=dtype, device=device)
    else:
        # cells to replicate per axis: ceil-ish bound from reciprocal widths.
        recip = torch.linalg.inv(cell).transpose(0, 1)            # rows = reciprocal vecs b_i
        n_rep = (torch.floor(r_cut * recip.norm(dim=1)).to(torch.long) + 1).tolist()
        rng = [torch.arange(-n, n + 1, device=device) for n in n_rep]
        S = torch.cartesian_prod(*rng).to(dtype).reshape(-1, 3)   # (n_shift, 3) integer shifts
        shift_cart = S @ cell                                      # (n_shift, 3) Cartesian

    rel = pos.unsqueeze(0) - pos.unsqueeze(1)                      # rel[i,j] = pos[j] - pos[i]  (N,N,3)
    D   = rel.unsqueeze(2) + shift_cart.view(1, 1, -1, 3)         # (N, N, n_shift, 3)
    d2  = (D * D).sum(-1)                                          # (N, N, n_shift)
    mask = (d2 < r_cut * r_cut) & (d2 > 1e-10)                     # drops the S=0 self term
    i, j, sidx = mask.nonzero(as_tuple=True)
    return i, j, shift_cart[sidx]


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
        volume = abs(np.linalg.det(cell)) if (s['pbc'] and cell is not None) else 0.0

        stress_t = None
        if s['stress'] is not None and volume > 0:
            stress_t = torch.tensor(np.asarray(s['stress']) * stress_conv,
                                    dtype=dtype, device=sdev)   # (3,3) eV/Å³

        out.append({
            'pos':     torch.tensor(s['positions'], dtype=dtype, device=sdev),
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
_BATCH_TENSOR_KEYS = ('pos', 'types', 'energy', 'forces', 'stress',
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
    stress_weight=0.0,
    stress_conv=STRESS_KBAR_TO_EVA3,
    loss='mse',
    huber_delta=0.01,
    eval_every=1,
    eval_batch_size=32,
    seed=42,
    dtype=torch.float64,
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
    is_ddp = world_size > 1
    is_main = (rank == 0)
    verbose = verbose and is_main
    use_stress = stress_weight > 0

    if device is None:
        device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)

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
        train_data = MPtrjShardDataset(train_shard_paths, rank=rank,
                                       world_size=world_size, seed=seed, shuffle=True)
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
        n_dist_embed=n_dist_embed,
        edge_type_nonlin=edge_type_nonlin, edge_type_linear=edge_type_linear,
        edge_type_output=edge_type_output,
        n_mp=n_mp, n_dist_basis=n_dist_basis,
    )
    if dtype == torch.float64:
        model = model.double()
    model = model.to(device)
    raw_model = model

    if is_ddp:
        train_model = DDP(_PBCMultiForwardWrapper(model), device_ids=[local_rank],
                          find_unused_parameters=False)
        # create_graph=True force/stress training yields non-contiguous grads →
        # DDP bucket-view stride mismatch. Make them contiguous (as in SPICE).
        for p in model.parameters():
            if p.requires_grad:
                p.register_hook(lambda g: g.contiguous())
    else:
        train_model = _PBCMultiForwardWrapper(model)

    # Plain (non-DDP) wrapper for evaluation — rank 0 calls it alone, so it must
    # not be the DDP module (whose forward expects all ranks to participate).
    eval_fwd = _PBCMultiForwardWrapper(raw_model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        mname = "ECENet"
        print_flush(f"\n{mname}: {n_layers} layers, l_max={l_max}, n_max={n_max}, "
                    f"embed_dim={embed_dim}, n_types={n_types}")
        print_flush(f"  Trainable parameters: {n_params:,}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=scheduler_patience)

    # ── Checkpoint restore ────────────────────────────────────────────────
    start_epoch = 0
    best_val_weighted = float('inf')
    best_test = (float('nan'), float('nan'), float('nan'))
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
        best_test = ckpt.get('best_test', best_test)
        if verbose:
            print_flush(f"Resumed from epoch {ckpt['epoch']}, "
                        f"best val [weighted]={best_val_weighted:.4f}")

    def save_checkpoint(epoch):
        if checkpoint_path is None or not is_main:
            return
        torch.save({
            'epoch': epoch,
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
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
                n_dist_embed=n_dist_embed,
                edge_type_nonlin=edge_type_nonlin, edge_type_linear=edge_type_linear,
                edge_type_output=edge_type_output,
                n_mp=n_mp, n_dist_basis=n_dist_basis,
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
        pos_in, shift_e_in, shift_nb_in = [], [], []
        types_b, ei_b, ej_b, ni_b, nj_b = [], [], [], [], []
        for d in batch:
            p = d['pos'].detach().clone().requires_grad_(True)
            pos_leaf.append(p)
            if use_stress:
                eps = torch.zeros(3, 3, dtype=d['pos'].dtype, device=d['pos'].device,
                                  requires_grad=True)
                strain_leaf.append(eps)
                pos_in.append(p + p @ eps)
                shift_e_in.append(d['shift_e'] + d['shift_e'] @ eps)
                shift_nb_in.append(d['shift_nb'] + d['shift_nb'] @ eps)
            else:
                pos_in.append(p)
                shift_e_in.append(d['shift_e'])
                shift_nb_in.append(d['shift_nb'])
            types_b.append(d['types'])
            ei_b.append(d['edge_i']); ej_b.append(d['edge_j'])
            ni_b.append(d['nb_src']); nj_b.append(d['nb_dst'])

        energies = fwd(pos_in, types_b, ei_b, ej_b, shift_e_in,
                       ni_b, nj_b, shift_nb_in)

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
        raw_model.eval()
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
        raw_model.train()
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
        raw_model.train()
        epoch_loss = 0.0

        rank_epoch_size = (epoch_size + world_size - 1) // world_size

        # Per-mode batch source. Prepared mode streams via DataLoader; legacy
        # mode samples indices into the in-memory list. Both yield lists of
        # per-frame dicts already on the compute device.
        if use_prepared:
            train_data.set_epoch(epoch)
            loader = DataLoader(train_data, batch_size=batch_size,
                                collate_fn=collate_keep_list, num_workers=num_workers,
                                pin_memory=True)
            n_batches_target = (rank_epoch_size + batch_size - 1) // batch_size
            def _batches():
                for b, raw in enumerate(loader):
                    if b >= n_batches_target:
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
                va_e, va_f, va_s = evaluate(val_data)
                # Weighted selection metric (mirrors the training-loss weighting);
                # stress only contributes when it's part of the loss (else va_s may be NaN).
                va_weighted = energy_weight * va_e + force_weight * va_f
                if use_stress:
                    va_weighted += stress_weight * va_s
                val_weighted_tensor = torch.tensor(va_weighted, device=device)
            if is_ddp:
                dist.broadcast(val_weighted_tensor, src=0)
            scheduler.step(val_weighted_tensor.item())

            if is_main:
                if va_weighted < best_val_weighted:
                    best_val_weighted = va_weighted
                    best_state = {k: v.clone() for k, v in raw_model.state_dict().items()}
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
                print_flush(
                    f"  Epoch {epoch+1:3d}: loss={epoch_loss:.4f} | "
                    f"train E={tr_e:.4f} F={tr_f:.4f} | val E={va_e:.4f} F={va_f:.4f}{ssfx} | "
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
        tr = evaluate(train_data, max_samples=500)
        va = evaluate(val_data)
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
