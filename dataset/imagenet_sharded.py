"""Sharded code dataset: reads .npz shards produced by extract_codes_c2i_sharded.py.

Each shard file: codes uint16 [N, num_aug, H*W], labels int32 [N].
Designed for dop-fuse / NFS: O(1000) shard files instead of O(1.28M) per-image npys.
"""
import bisect
import glob
import os
from typing import List, Optional

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.distributed as dist

class ShardedCodeDataset(Dataset):
    def __init__(self, code_dir: str, glob_pattern: str = "shard_*.npz"):
        self.code_dir = code_dir
        self.shard_files: List[str] = sorted(
            glob.glob(os.path.join(code_dir, glob_pattern))
        )
        if not self.shard_files:
            raise FileNotFoundError(f"no shards matching {glob_pattern} under {code_dir}")

        # Read just the header of each shard for size; load on demand.
        self.sizes: List[int] = []
        for f in self.shard_files:
            with np.load(f, mmap_mode="r") as z:
                self.sizes.append(int(z["labels"].shape[0]))
        self.cum = np.cumsum([0] + self.sizes).tolist()
        self.total = self.cum[-1]

        # Per-instance LRU of size 1: most accesses are sequential within a shard.
        self._cache_idx: Optional[int] = None
        self._cache_codes: Optional[np.ndarray] = None
        self._cache_labels: Optional[np.ndarray] = None

        # Compatibility shims for utilities that probe these attributes.
        self.flip = True  # flip aug is baked into num_aug=2
        self.feature_dir = code_dir
        self.aug_feature_dir = None

    def __len__(self) -> int:
        return self.total

    def _load_shard(self, shard_idx: int):
        if self._cache_idx == shard_idx:
            return
        z = np.load(self.shard_files[shard_idx])
        self._cache_codes = z["codes"]    # [N, num_aug, H*W] uint16
        self._cache_labels = z["labels"]  # [N] int32
        self._cache_idx = shard_idx

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        shard_idx = bisect.bisect_right(self.cum, idx) - 1
        local = idx - self.cum[shard_idx]
        self._load_shard(shard_idx)
        codes = self._cache_codes[local]   # [num_aug, H*W] uint16
        label = int(self._cache_labels[local])
        # Match PAR's CustomDataset return contract: tensor codes + tensor label.
        return torch.from_numpy(codes.astype(np.int64)), torch.tensor([label], dtype=torch.long)


class ShardedCodeDataseInRAM(Dataset):
    def __init__(self, code_dir: str, glob_pattern: str = "shard_*.npz"):
        self.code_dir = code_dir
        self.shard_files: List[str] = sorted(
            glob.glob(os.path.join(code_dir, glob_pattern))
        )
        if not self.shard_files:
            raise FileNotFoundError(f"no shards matching {glob_pattern} under {code_dir}")

        codes_list = []
        labels_list = []
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0

        iterator = tqdm(self.shard_files, desc=f"Loading Codes to RAM") if is_main_process else self.shard_files
        for f in iterator:
            with np.load(f) as z:
                codes_list.append(z["codes"])
                labels_list.append(z["labels"])
        
        self.all_codes = np.concatenate(codes_list, axis=0)
        self.all_labels = np.concatenate(labels_list, axis=0)
        self.total = self.all_labels.shape[0]
        
        # Compatibility shims for utilities that probe these attributes.
        self.flip = True  # flip aug is baked into num_aug=2
        self.feature_dir = code_dir
        self.aug_feature_dir = None

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        codes = self.all_codes[idx]   # [num_aug, H*W] uint16
        label = int(self.all_labels[idx])
        # Match PAR's CustomDataset return contract: tensor codes + tensor label.
        return torch.from_numpy(codes.astype(np.int64)), torch.tensor([label], dtype=torch.long)



def build_imagenet_sharded(args, **kwargs):
    code_dir = f"{args.code_path}/imagenet{args.image_size}_codes_sharded"
    return ShardedCodeDataset(code_dir)
