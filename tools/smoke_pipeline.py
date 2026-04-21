"""End-to-end smoke for trainer + sampler paths without DDP.

- Builds a tiny synthetic ShardedCodeDataset (no remote data needed).
- Runs one training step.
- Runs one sampling step.
- Verifies sampler→VQ-decode shape pipeline.
"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from autoregressive.models.rowar import RowAR_models
from autoregressive.sample.sample_c2i_rowar import generate_rowar
from dataset.imagenet_sharded import ShardedCodeDataset


def make_fake_shards(root, n_shards=2, per_shard=64, H=24, W=24, V=16384):
    os.makedirs(root, exist_ok=True)
    for s in range(n_shards):
        codes = np.random.randint(0, V, size=(per_shard, 2, H * W), dtype=np.int64).astype(np.uint16)
        labels = np.random.randint(0, 1000, size=(per_shard,), dtype=np.int32)
        np.savez(os.path.join(root, f"shard_rank000_{s:06d}.npz"),
                 codes=codes, labels=labels)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = W = 24

    with tempfile.TemporaryDirectory() as td:
        make_fake_shards(td, n_shards=2, per_shard=64)
        ds = ShardedCodeDataset(td)
        print(f"dataset size: {len(ds)}")
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        x, y = next(iter(loader))
        print(f"batch: x={tuple(x.shape)} dtype={x.dtype}, y={tuple(y.shape)} dtype={y.dtype}")
        assert x.shape == (4, 2, H * W)
        assert y.shape == (4, 1)

        # ---- trainer step ----
        model = RowAR_models["RowAR-B"](
            vocab_size=16384, num_classes=1000, grid_h=H, grid_w=W
        ).to(device).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = x.to(device).long()
        y = y.to(device).reshape(-1).long()
        B, num_aug, N = x.shape
        aug_idx = torch.randint(0, num_aug, (B,), device=device)
        tokens = x[torch.arange(B, device=device), aug_idx].reshape(B, H, W)
        _, loss = model(tokens=tokens, class_idx=y)
        loss.backward()
        opt.step()
        print(f"train step OK, loss={loss.item():.3f}")

        # ---- sampler step (no CFG, fast) ----
        model.eval()
        cls = torch.tensor([207, 360], device=device)
        codes, step_times = generate_rowar(
            model, cls, H=H, W=W, cfg_scale=1.0, temperature=1.0, top_k=100, top_p=1.0
        )
        assert codes.shape == (2, H * W)
        assert codes.dtype == torch.long
        print(f"sample OK: codes={tuple(codes.shape)}, "
              f"steps={len(step_times)}, mean step={1000*sum(step_times)/len(step_times):.1f}ms")
        # range check
        assert codes.min() >= 0 and codes.max() < 16384

        # ---- KV-cache correctness: incremental cached forward vs training forward ----
        # Feed ground-truth rows one-by-one and check the per-row logits match the
        # all-at-once training forward at the matching query positions.
        with torch.no_grad():
            tok_grid = codes.view(2, H, W).long()
            train_logits, _ = model(tokens=tok_grid, class_idx=cls)   # [2, H*W, V]
            train_logits = train_logits.view(2, H, W, -1)

            model.reset_kv_cache()
            prev = None
            max_abs = 0.0
            for r in range(H):
                step_logits = model(tokens=None, class_idx=cls, prev_rows=prev)  # [2, W, V]
                diff = (step_logits.float() - train_logits[:, r].float()).abs().max().item()
                max_abs = max(max_abs, diff)
                row = tok_grid[:, r, :]
                prev = row.unsqueeze(1) if prev is None else torch.cat([prev, row.unsqueeze(1)], dim=1)
            print(f"KV-cache correctness: max|cached - train| = {max_abs:.2e}")
            assert max_abs < 1e-3, f"cached logits diverge from training forward: {max_abs}"


if __name__ == "__main__":
    main()
