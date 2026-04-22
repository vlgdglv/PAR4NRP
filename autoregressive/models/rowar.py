"""RowAR V2: single-block row-parallel autoregressive transformer.

Predicts a full row of VQ tokens in parallel. For an HxW grid this is H
sequential steps instead of H*W, while staying byte-compatible with a
LlamaGen c2i raster-AR checkpoint for warm-start.

Sequence layout (S = 1 + H*W):
  pos 0                : CLS embedding (class label, with cfg dropout)
  pos 1 .. W           : block 0 inputs = bos_row replicated W times
                         -> predicts row 0
  pos 1+r*W .. 1+(r+1)*W-1 (r in 1..H-1):
                         block r inputs = tok_emb(x_{r-1, *})
                         -> predicts row r

Attention mask (block-causal, bidirectional within block):
  CLS attends only to itself.
  Position in block r attends to: CLS + all positions in blocks 0..r
  (i.e. self-block is bidirectional; prior blocks fully visible).

2D RoPE keyed by (row, col):
  CLS -> zero rotation.
  Position (r, c) (block r, column c) -> 2D RoPE at (r, c).

Why this beats the V1 dual-block design:
  V1 had [CLS, prefix(H*W), queries(H*W)] (length 1+2HW). The query Q-projection
  was fed `tok_emb(x_{r-1, c})` while LlamaGen-warm-started Q-proj was trained
  for `tok_emb(x_{r, c})` -> active task mismatch, initial loss > ln(V).
  V1 also doubled the sequence length and CFG curve was monotonic in CFG
  (no U-shape) up to CFG=4 -> conditional distribution was weak, unconditional
  near-garbage.
  V2 uses one block, length 1+HW (matches LlamaGen). Q-proj sees tok_emb of the
  PREVIOUS row, exactly mirroring LlamaGen's "predict next from prev token"
  objective, just reorganized into row-blocks via the attention mask. Warm-start
  is now genuinely matched.

Loss: CE on logits at all data positions (block 0..H-1).
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from autoregressive.models.gpt import (
    Attention,
    FeedForward,
    LabelEmbedder,
    RMSNorm,
    apply_rotary_emb,
    find_multiple,
)
from utils.drop_path import DropPath


@dataclass
class RowARArgs:
    dim: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    vocab_size: int = 16384

    grid_h: int = 24
    grid_w: int = 24
    cls_token_num: int = 1


class RowARBlock(nn.Module):
    """Llama transformer block adapted to (x, freqs_cis, mask) signature."""

    def __init__(self, config: RowARArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, freqs_cis, mask):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, None, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


def build_row_freqs_cis(H: int, W: int, head_dim: int, base: float = 10000.0):
    """2D RoPE for the V2 single-block layout, length 1 + H*W.
    Position 0 (CLS) -> zeros (no rotation). Position 1+r*W+c -> 2D RoPE at (r, c).
    Returns tensor of shape (1 + H*W, head_dim // 2, 2).
    """
    assert head_dim % 2 == 0
    half_dim = head_dim // 2
    per_axis = half_dim // 2
    assert per_axis > 0, "head_dim too small for 2D RoPE"

    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[:per_axis].float() / half_dim))
    t_r = torch.arange(H, dtype=torch.float32)
    t_c = torch.arange(W, dtype=torch.float32)
    row_freqs = torch.outer(t_r, freqs)   # (H, per_axis)
    col_freqs = torch.outer(t_c, freqs)   # (W, per_axis)

    row_grid = row_freqs[:, None, :].expand(H, W, per_axis)
    col_grid = col_freqs[None, :, :].expand(H, W, per_axis)
    grid = torch.cat([row_grid, col_grid], dim=-1)             # (H, W, half_dim)
    cache = torch.stack([torch.cos(grid), torch.sin(grid)], dim=-1)  # (H, W, half_dim, 2)
    cache = cache.reshape(H * W, half_dim, 2)

    cls_part = torch.zeros(1, half_dim, 2)
    return torch.cat([cls_part, cache], dim=0)                  # (1 + H*W, half_dim, 2)


def build_row_attn_mask(H: int, W: int) -> torch.Tensor:
    """Block-causal mask, shape (S, S), S = 1 + H*W. True = attend allowed.
      pos 0      : CLS, attends to itself only.
      pos 1..S-1 : block r contains W positions; each attends to CLS and to
                   all positions in blocks 0..r (own block is bidirectional).
    """
    S = 1 + H * W
    mask = torch.zeros(S, S, dtype=torch.bool)

    mask[0, 0] = True
    mask[1:, 0] = True  # all data attends to CLS

    block_idx = torch.arange(H * W) // W                  # [H*W]
    bi = block_idx.view(-1, 1)                             # [H*W, 1]
    bj = block_idx.view(1, -1)                             # [1, H*W]
    mask[1:, 1:] = bj <= bi                                # block-causal, intra-block bidir
    return mask


class RowARTransformer(nn.Module):
    def __init__(self, config: RowARArgs):
        super().__init__()
        self.config = config
        self.H = config.grid_h
        self.W = config.grid_w
        self.cls_token_num = config.cls_token_num

        # Class conditioning (matches LlamaGen / PAR exactly for warm transfer).
        self.cls_embedding = LabelEmbedder(
            config.num_classes, config.dim, config.class_dropout_prob
        )
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # Single learned vector for row-0 inputs (no previous row exists).
        self.bos_row = nn.Parameter(torch.zeros(config.dim))

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = nn.ModuleList([RowARBlock(config, dpr[i]) for i in range(config.n_layer)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        head_dim = config.dim // config.n_head
        self.register_buffer(
            "freqs_cis",
            build_row_freqs_cis(self.H, self.W, head_dim, base=config.rope_base),
            persistent=False,
        )
        self.register_buffer(
            "attn_mask",
            build_row_attn_mask(self.H, self.W),
            persistent=False,
        )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
        # Random init for output head (warm-start usually overwrites it; if it
        # silently doesn't, std=0.02 init is the safe fallback).
        nn.init.normal_(self.bos_row, mean=0.0, std=self.config.initializer_range)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    # ---------------------------------------------------------------- helpers
    def _build_block_inputs(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, H, W] long. Returns [B, H*W, dim] block inputs.
        Block r input: tok_emb(tokens[:, r-1, :]) for r >= 1, bos_row for r = 0.
        """
        B, H, W = tokens.shape
        prev = torch.empty_like(tokens)
        prev[:, 1:, :] = tokens[:, :-1, :]
        prev[:, 0, :] = 0  # placeholder, overwritten below by bos_row
        emb = self.tok_embeddings(prev)              # [B, H, W, dim]
        emb[:, 0, :, :] = self.bos_row.view(1, 1, -1).expand(B, W, -1)
        return emb.reshape(B, H * W, -1)

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        tokens: Optional[torch.Tensor],            # [B, H, W] long, training only
        class_idx: Optional[torch.Tensor],         # [B] long
        prev_rows: Optional[torch.Tensor] = None,  # [B, r, W] long, inference path
        targets: Optional[torch.Tensor] = None,    # [B, H*W] long
    ):
        if self.training or tokens is not None:
            return self._forward_train(tokens, class_idx, targets)
        return self._forward_step(prev_rows, class_idx)

    def _forward_train(self, tokens, class_idx, targets):
        B, H, W = tokens.shape
        assert (H, W) == (self.H, self.W)
        device = tokens.device

        cls_emb = self.cls_embedding(class_idx, train=self.training)[:, :self.cls_token_num]  # [B,1,d]
        block_inputs = self._build_block_inputs(tokens)                                       # [B,H*W,d]

        h = torch.cat([cls_emb, block_inputs], dim=1)                                         # [B,1+H*W,d]
        h = self.tok_dropout(h)

        mask = self.attn_mask.to(device).unsqueeze(0).unsqueeze(0)                            # [1,1,S,S]
        freqs_cis = self.freqs_cis.to(device)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)

        # Logits at data positions only (skip CLS).
        logits = self.output(h[:, 1:, :]).float()                                             # [B,H*W,V]

        if targets is None:
            targets = tokens.reshape(B, H * W)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        return logits, loss

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def reset_kv_cache(self):
        """Clear the per-layer K/V cache used by cached sampling."""
        self._cache_k = [None] * len(self.layers)
        self._cache_v = [None] * len(self.layers)

    def _step_layer_cached(self, layer_idx, layer, x, freqs, intra_chunk_mask):
        """One transformer block forward for a single RowAR row step with KV cache.

        x:            [B, L, d]   new chunk this step (CLS+block_0 if r==0, else block_r).
        freqs:        [L, head_dim//2, 2]   RoPE for the chunk's absolute positions.
        intra_chunk_mask: bool [L, L] or None.
                      None means "attend to everything in cache + self" (used for r >= 1
                      where the chunk is one bidirectional block and cache is fully
                      visible). For r == 0 we pass an explicit mask so CLS does not
                      see the block-0 positions appended in the same chunk.

        Side-effect: appends new K/V to self._cache_{k,v}[layer_idx].
        """
        att = layer.attention
        B, L, _ = x.shape
        n_head = att.n_head
        n_kv = att.n_kv_head
        hd = att.head_dim
        dim = att.dim
        kv_sz = n_kv * hd

        xn = layer.attention_norm(x)
        qkv = att.wqkv(xn)
        q, k, v = qkv.split([dim, kv_sz, kv_sz], dim=-1)
        q = q.view(B, L, n_head, hd)
        k = k.view(B, L, n_kv, hd)
        v = v.view(B, L, n_kv, hd)
        q = apply_rotary_emb(q, freqs)
        k = apply_rotary_emb(k, freqs)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)  # [B, head, L, hd]

        # ---- append new K/V to cache ----
        past_k = self._cache_k[layer_idx]
        past_v = self._cache_v[layer_idx]
        if past_k is None:
            full_k = k
            full_v = v
        else:
            full_k = torch.cat([past_k, k], dim=2)
            full_v = torch.cat([past_v, v], dim=2)
        self._cache_k[layer_idx] = full_k
        self._cache_v[layer_idx] = full_v

        # GQA: repeat KV heads up to n_head if needed.
        rep = n_head // n_kv
        K_all = full_k.repeat_interleave(rep, dim=1) if rep > 1 else full_k
        V_all = full_v.repeat_interleave(rep, dim=1) if rep > 1 else full_v

        L_prev = 0 if past_k is None else past_k.shape[2]
        L_full = L_prev + L

        if intra_chunk_mask is not None:
            full_mask = torch.zeros(L, L_full, dtype=torch.bool, device=x.device)
            full_mask[:, :L_prev] = True
            full_mask[:, L_prev:] = intra_chunk_mask
            attn_mask = full_mask.view(1, 1, L, L_full)
            out = F.scaled_dot_product_attention(q, K_all, V_all, attn_mask=attn_mask)
        else:
            # No mask: attend to all of cache + self (block-r is bidirectional).
            out = F.scaled_dot_product_attention(q, K_all, V_all)

        out = out.transpose(1, 2).contiguous().view(B, L, dim)
        out = att.resid_dropout(att.wo(out))
        x = x + out
        x = x + layer.feed_forward(layer.ffn_norm(x))
        return x

    @torch.no_grad()
    def _forward_step(self, prev_rows: Optional[torch.Tensor], class_idx: torch.Tensor):
        """One row step with persistent KV cache.

        Call order: H times with prev_rows of length 0, 1, ..., H-1.
        Cache grows internally; call reset_kv_cache() before a new sample.
        """
        B = class_idx.shape[0]
        device = class_idx.device
        H, W = self.H, self.W
        if not hasattr(self, "_cache_k") or self._cache_k is None or len(self._cache_k) == 0:
            self.reset_kv_cache()
        r = 0 if prev_rows is None else prev_rows.shape[1]
        assert r < H

        freqs = self.freqs_cis.to(device)

        if r == 0:
            # chunk = [CLS, bos_row * W], absolute positions 0..W
            cls_emb = self.cls_embedding(class_idx, train=False)[:, :self.cls_token_num]   # [B,1,d]
            bos = self.bos_row.view(1, 1, -1).expand(B, W, -1).to(cls_emb.dtype)            # [B,W,d]
            x = torch.cat([cls_emb, bos], dim=1)                                            # [B,1+W,d]
            pos = torch.arange(0, 1 + W, device=device)
            L = 1 + W
            chunk_mask = torch.zeros(L, L, dtype=torch.bool, device=device)
            chunk_mask[0, 0] = True       # CLS sees only itself
            chunk_mask[1:, :] = True      # block-0 positions see CLS + each other
        else:
            last_row = prev_rows[:, -1, :]
            x = self.tok_embeddings(last_row).to(self._cache_k[0].dtype if self._cache_k[0] is not None else self.tok_embeddings.weight.dtype)  # [B,W,d]
            start = 1 + r * W
            pos = torch.arange(start, start + W, device=device)
            chunk_mask = None

        x = self.tok_dropout(x)
        freqs_step = freqs[pos]

        for i, layer in enumerate(self.layers):
            x = self._step_layer_cached(i, layer, x, freqs_step, chunk_mask)

        x = self.norm(x)
        # Logits at block-r positions only (drop CLS slot at r==0).
        logits_part = x[:, 1:, :] if r == 0 else x
        logits = self.output(logits_part).float()                                            # [B, W, V]
        return logits

    # ---------------------------------------------------------------- ckpt I/O
    def load_llamagen_state_dict(self, state_dict: dict, verbose: bool = True):
        """Map a LlamaGen c2i checkpoint into RowAR. Returns (loaded, skipped).

        LlamaGen and RowAR share the llama block module names exactly:
          cls_embedding.embedding_table.weight
          tok_embeddings.weight
          layers.{i}.attention.{wqkv,wo}.weight
          layers.{i}.feed_forward.{w1,w2,w3}.weight
          layers.{i}.attention_norm.weight
          layers.{i}.ffn_norm.weight
          norm.weight
          output.weight

        Skipped (not present or incompatible): freqs_cis (rebuilt), causal_mask
        (rebuilt), spe_tok_embeddings.* (PAR-only), bos_row (new).
        """
        own = self.state_dict()
        loaded, skipped = [], []
        for k, v in state_dict.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                loaded.append(k)
            else:
                skipped.append((k, tuple(v.shape) if hasattr(v, "shape") else None))
        loaded_set = set(loaded)
        fresh = [k for k in own.keys() if k not in loaded_set]

        critical = ["output.weight", "tok_embeddings.weight",
                    "cls_embedding.embedding_table.weight"]
        missing_critical = [k for k in critical if k not in loaded_set]
        if missing_critical:
            raise RuntimeError(
                f"[RowAR.load_llamagen] CRITICAL tensors not warm-started: "
                f"{missing_critical}. Check that the checkpoint contains these keys "
                f"and shapes match. Training with these cold-init will plateau."
            )

        if verbose:
            print(f"[RowAR.load_llamagen] loaded {len(loaded)} tensors, "
                  f"skipped {len(skipped)} from ckpt, fresh-init {len(fresh)} in model")
            for k in critical:
                t = own[k].float()
                print(f"  [warm-start check] {k} std={t.std().item():.4f} "
                      f"abs_max={t.abs().max().item():.4f}")
            if skipped:
                print("  skipped (in ckpt, not used):")
                for k, sh in skipped[:20]:
                    print(f"    {k}  {sh}")
            if fresh:
                print("  fresh-init (in model, not from ckpt):")
                for k in fresh[:20]:
                    print(f"    {k}")
        return loaded, skipped


# --------------------------------------------------------------------- factories
def RowAR_B(**kw):
    return RowARTransformer(RowARArgs(n_layer=12, n_head=12, dim=768, **kw))

def RowAR_L(**kw):
    return RowARTransformer(RowARArgs(n_layer=24, n_head=16, dim=1024, **kw))

def RowAR_XL(**kw):
    return RowARTransformer(RowARArgs(n_layer=36, n_head=20, dim=1280, **kw))


RowAR_models = {
    "RowAR-B": RowAR_B,
    "RowAR-L": RowAR_L,
    "RowAR-XL": RowAR_XL,
}
