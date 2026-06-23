"""IterableDataset over pre-tensorized MPtrj shards.

Pairs with ``prepare_mptrj.py``: that script writes ``shard_NNNNN.pt`` (each
holding a list of per-frame tensor dicts), plus ``manifest.json``,
``type_map.pt``, and ``e_ref.pt``. This module loads those shards on demand,
yielding per-frame dicts in a DDP-aware, shuffled, per-worker-disjoint order.

Sharding strategy (per epoch, with shuffle=True):
  1. permute the full shard index list (seed = base_seed + epoch).
  2. take the rank's slice via stride: my_shards = perm[rank::world_size].
  3. if a DataLoader worker is attached, sub-slice again by worker id.
  4. for each assigned shard: load it from disk, permute frame order
     (seed = base_seed + epoch + shard_idx), yield each frame.

This gives every frame on the node exactly once per epoch with cheap
shard-level locality (one disk read per ~10k frames). Cross-epoch entropy
comes from the (shard permutation × intra-shard permutation) per epoch.
"""

import json
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


def load_manifest(prepared_dir):
    """Return (manifest_dict, type_map_dict, e_ref_np, abs_shard_paths)."""
    prepared_dir = Path(prepared_dir)
    with open(prepared_dir / 'manifest.json') as f:
        manifest = json.load(f)
    type_map = torch.load(prepared_dir / 'type_map.pt', weights_only=False)
    e_ref = torch.load(prepared_dir / 'e_ref.pt', weights_only=False).numpy()
    shard_paths = [str(prepared_dir / s) for s in manifest['shards']]
    return manifest, type_map, e_ref, shard_paths


class MPtrjShardDataset(IterableDataset):
    """Streams per-frame dicts from a list of .pt shard files.

    Parameters
    ----------
    shard_paths : sequence of str
        Absolute paths to ``shard_*.pt`` files. Each shard is a ``list[dict]``.
    rank, world_size : int
        DDP rank info. Each rank receives a disjoint subset of shards per epoch
        (round-robin over a shard permutation).
    seed : int
        Base seed; per-epoch shuffles are deterministic from (seed, epoch).
    shuffle : bool
        Shuffle shards across epochs and frames within each shard. Default True.
        Set False for eval to walk shards in their on-disk order.
    """

    def __init__(self, shard_paths: Sequence[str], rank: int = 0,
                 world_size: int = 1, seed: int = 0, shuffle: bool = True):
        super().__init__()
        self.shard_paths: List[str] = list(shard_paths)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Call once per epoch (mirrors torch.utils.data.DistributedSampler)."""
        self._epoch = int(epoch)

    # Approximate length: full dataset divided across ranks. Used by trainer
    # progress accounting; an off-by-one due to non-full last shard is fine.
    def __len__(self) -> int:
        return len(self.shard_paths) // max(1, self.world_size)

    def __iter__(self) -> Iterator[dict]:
        rng = np.random.RandomState(self.seed + self._epoch)
        if self.shuffle:
            order = rng.permutation(len(self.shard_paths))
        else:
            order = np.arange(len(self.shard_paths))

        # Rank-disjoint slice.
        my_shards = order[self.rank::self.world_size]

        # Sub-slice across DataLoader workers (if any).
        wi = get_worker_info()
        if wi is not None:
            my_shards = my_shards[wi.id::wi.num_workers]

        for si in my_shards:
            shard = torch.load(self.shard_paths[int(si)], map_location='cpu',
                               weights_only=False)
            if self.shuffle:
                inner_rng = np.random.RandomState(self.seed + self._epoch * 100003
                                                  + int(si))
                idx = inner_rng.permutation(len(shard))
            else:
                idx = np.arange(len(shard))
            for fi in idx:
                yield shard[int(fi)]


def split_shards(shard_paths: Sequence[str], val_frac: float,
                 seed: int = 0) -> tuple:
    """Split shard list into (train_shards, val_shards) by holding out a
    fraction of shards. Frames were pre-shuffled at prepare time, so a random
    shard-level holdout is a uniform random frame-level holdout."""
    shard_paths = list(shard_paths)
    n = len(shard_paths)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n))) if n > 1 else 0
    val_idx = set(perm[:n_val].tolist())
    train = [s for i, s in enumerate(shard_paths) if i not in val_idx]
    val = [shard_paths[i] for i in sorted(val_idx)]
    return train, val


def collate_keep_list(batch):
    """No-op collate: the trainer's predict() expects a list of per-frame
    dicts, not a stacked batch tensor. Each DataLoader iteration yields a
    list of dicts."""
    return batch
