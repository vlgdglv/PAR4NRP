"""Tiny smoke test for RowARTransformer.

- Build RowAR-B
- Forward + backward on random tokens
- Run a sampling step
- Check mask invariants
- Try loading a fake LlamaGen-shaped state dict
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from autoregressive.models.rowar import (
    RowAR_models, build_row_attn_mask, build_row_freqs_cis,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = W = 24
    V = 16384
    B = 2

    # ----- mask sanity -----
    m = build_row_attn_mask(H, W)
    S = 1 + 2 * H * W
    assert m.shape == (S, S)
    P0, Q0 = 1, 1 + H * W
    # CLS row attends only to itself
    assert m[0, 0].item() and m[0, 1:].sum().item() == 0, "CLS row leak"
    # Prefix is causal among itself + sees CLS
    p_block = m[P0:Q0, P0:Q0]
    assert torch.equal(p_block, torch.tril(torch.ones_like(p_block))), "prefix not causal"
    assert m[P0:Q0, 0].all(), "prefix should see CLS"
    # Queries at row r see CLS + prefix rows 0..r-1, not their own row, not other queries
    for r in [0, 1, 5, 23]:
        q_idx = Q0 + r * W + W // 2
        # CLS visible
        assert m[q_idx, 0].item()
        # Prefix rows < r visible, row >= r invisible
        for rr in range(H):
            slc = m[q_idx, P0 + rr * W : P0 + (rr + 1) * W]
            if rr < r:
                assert slc.all().item(), f"row {r} should see prefix row {rr}"
            else:
                assert (~slc).all().item(), f"row {r} must NOT see prefix row {rr}"
        # No query attends to any query
        assert (~m[q_idx, Q0:S]).all().item(), f"query leak at row {r}"
    print("mask sanity OK")

    # ----- RoPE shape -----
    head_dim = 64
    fc = build_row_freqs_cis(H, W, head_dim)
    assert fc.shape == (1 + 2 * H * W, head_dim // 2, 2)
    # CLS row should be zeros
    assert fc[0].abs().sum().item() == 0.0
    # Prefix and query slot for same (r,c) should have identical RoPE
    for r, c in [(0, 0), (3, 7), (23, 23)]:
        i_prefix = 1 + r * W + c
        i_query  = 1 + H * W + r * W + c
        assert torch.allclose(fc[i_prefix], fc[i_query]), f"RoPE mismatch at ({r},{c})"
    print("RoPE sanity OK")

    # ----- model forward + backward -----
    model = RowAR_models["RowAR-B"](
        vocab_size=V, num_classes=1000, grid_h=H, grid_w=W
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RowAR-B params: {n_params/1e6:.1f}M")

    tokens = torch.randint(0, V, (B, H, W), device=device)
    cls = torch.randint(0, 1000, (B,), device=device)

    model.train()
    logits, loss = model(tokens=tokens, class_idx=cls)
    print(f"train logits: {tuple(logits.shape)}, loss: {loss.item():.3f} (expect ~{torch.log(torch.tensor(float(V))).item():.2f} at init)")
    loss.backward()
    print("backward OK")

    # ----- one sampling step -----
    model.eval()
    # row 0
    logits0 = model(tokens=None, class_idx=cls, prev_rows=None)
    assert logits0.shape == (B, W, V)
    # row 5
    prev = torch.randint(0, V, (B, 5, W), device=device)
    logits5 = model(tokens=None, class_idx=cls, prev_rows=prev)
    assert logits5.shape == (B, W, V)
    print("sampling step OK")

    # ----- fake LlamaGen warm start -----
    fake = {}
    for k, v in model.state_dict().items():
        if k in ("bos_row", "freqs_cis", "attn_mask"):
            continue
        fake[k] = torch.randn_like(v) * 0.01
    # Add a couple of LlamaGen-specific keys that won't match (should be skipped)
    fake["spe_tok_embeddings.special_embeddings.weight"] = torch.randn(3, 768)
    fake["freqs_cis"] = torch.randn(577, 32, 2)  # PAR's, wrong shape
    fresh_model = RowAR_models["RowAR-B"](
        vocab_size=V, num_classes=1000, grid_h=H, grid_w=W
    ).to(device)
    loaded, skipped = fresh_model.load_llamagen_state_dict(fake, verbose=False)
    print(f"warm-start: loaded={len(loaded)} skipped={len(skipped)}")
    assert len(loaded) >= 80, "expected most weights to load"
    # bos_row should remain freshly initialized
    assert "bos_row" not in [k for k in fake.keys() if k in fresh_model.state_dict()]
    print("LlamaGen warm-start path OK")


if __name__ == "__main__":
    main()
