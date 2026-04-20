"""Read HuggingFace `imagenet-1k` parquet shards directly.

Schema: image: struct<bytes: binary, path: string>, label: int64
Avoids unpacking 1.28M JPEGs to disk (bad on dop-fuse / NFS-class FS).
"""
import io
import glob
import os
from typing import List

import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info


def _list_train_parquets(data_path: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_path, "train-*.parquet")))
    if not files:
        raise FileNotFoundError(f"no train-*.parquet under {data_path}")
    return files


def _split_for_worker(items: List[str], rank: int, world_size: int,
                      worker_id: int, num_workers: int) -> List[str]:
    # rank-major, worker-minor partition; deterministic, no overlap.
    bucket = rank * num_workers + worker_id
    total_buckets = world_size * num_workers
    return items[bucket::total_buckets]


class ImageNetParquetIterable(IterableDataset):
    """One pass over HF imagenet-1k train parquets.

    Yields (image_tensor, label_int) where image_tensor has the user-supplied
    transform already applied. Designed for the extract_codes job: ordered,
    DDP-partitioned by parquet file, no shuffling.
    """

    def __init__(self, data_path: str, transform, rank: int = 0,
                 world_size: int = 1, row_group_batch: int = 1):
        super().__init__()
        self.files = _list_train_parquets(data_path)
        self.transform = transform
        self.rank = rank
        self.world_size = world_size
        self.row_group_batch = row_group_batch

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        my_files = _split_for_worker(self.files, self.rank, self.world_size,
                                     worker_id, num_workers)
        for path in my_files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
                imgs = batch.column("image").to_pylist()
                labels = batch.column("label").to_pylist()
                for img_struct, label in zip(imgs, labels):
                    raw = img_struct["bytes"]
                    try:
                        pil = Image.open(io.BytesIO(raw)).convert("RGB")
                    except Exception:
                        # Skip corrupt rows rather than crash an 8h extraction.
                        continue
                    yield self.transform(pil), int(label)

    def approx_len(self) -> int:
        # Sum row-group metadata; cheap enough.
        n = 0
        for f in self.files:
            n += pq.ParquetFile(f).metadata.num_rows
        return n
