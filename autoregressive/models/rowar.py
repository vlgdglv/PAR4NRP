"""RowAR: Next-Row autoregressive transformer.

Predicts a full row of VQ tokens in parallel, conditioned on the class label
and all previously generated rows. For a 24x24 grid this is H=24 sequential
steps instead of 576, while keeping a standard llama-style transformer.

Layout per row r (D2 design: vertical predecessor as query):
  prefix:  [CLS]  prefix tokens are the GT preceding rows
                  positions 1 .. r*W  hold rows 0..r-1 in row-major order
  queries: at positions r*W+1 .. (r+1)*W
           input embedding = tok_emb(x_{r-1, c})  for r>0
                           = BOS_ROW              for r=0
           predicts          x_{r, c}

Training (all-row parallel): build one sequence of length 1 + H*W + H*W
  [CLS] [GT prefix tokens, H*W slots] [query tokens, H*W slots]

Mask:
  prefix part: causal among themselves (positions 0..H*W).
  query at slot (r,c) (global pos = 1+H*W + r*W + c):
    - attends to [CLS]  (position 0)
    - attends to prefix positions [1 .. 1+r*W]  (rows 0..r-1, NOT row r)
    - does NOT attend to any other query

Loss: CE on logits at query positions only (labels = flattened image tokens).

Reuses llama building blocks from autoregressive/models/gpt.py so a
LlamaGen c2i checkpoint can warm-start everything except row queries
and the new positional/mask buffers.
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
    """Same as PAR/LlamaGen TransformerBlock but takes (x, freqs_cis, mask)."""

    def __init__(self, config: RowARArgs, drop_path: float):
        super().__init__()
        # Adapt RowARArgs into the dataclass-like object Attention/FeedForward expect.
        # Attention only reads .dim, .n_head, .n_kv_head, .attn_dropout_p, .resid_dropout_p
        # FeedForward reads .dim, .ffn_dim_multiplier, .multiple_of, .ffn_dropout_p
        # Both work with our RowARArgs duck-typed.
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
    """2D RoPE keyed by (row, col) in row-major order, packed for
         [CLS] [prefix(H*W) row-major] [queries(H*W) row-major]
    Query at slot (r, c) reuses the prefix (r, c) RoPE so it is positionally
    identical to the slot whose token it is predicting. CLS receives zeros
    (no rotation), matching LlamaGen/PAR.

    Returns tensor of shape (1 + 2*H*W, head_dim // 2, 2) with .float dtype.
    """
    assert head_dim % 2 == 0
    half_dim = head_dim // 2          # dims per token's rotary pair
    per_axis = half_dim // 2          # half of half: split between row and col axes
    assert per_axis > 0, "head_dim too small for 2D RoPE"

    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[:per_axis].float() / half_dim))
    # Row and column position indices.
    t_r = torch.arange(H, dtype=torch.float32)
    t_c = torch.arange(W, dtype=torch.float32)
    row_freqs = torch.outer(t_r, freqs)  # (H, per_axis)
    col_freqs = torch.outer(t_c, freqs)  # (W, per_axis)

    # Expand to a (H, W, half_dim) grid by concatenating row-axis and col-axis halves.
    row_grid = row_freqs[:, None, :].expand(H, W, per_axis)   # (H, W, per_axis)
    col_grid = col_freqs[None, :, :].expand(H, W, per_axis)   # (H, W, per_axis)
    grid = torch.cat([row_grid, col_grid], dim=-1)             # (H, W, half_dim)

    cache = torch.stack([torch.cos(grid), torch.sin(grid)], dim=-1)  # (H, W, half_dim, 2)
    cache = cache.reshape(H * W, half_dim, 2)                         # row-major flatten

    cls_part = torch.zeros(1, half_dim, 2)
    query_part = cache.clone()
    return torch.cat([cls_part, cache, query_part], dim=0)            # (1 + 2*H*W, half_dim, 2)


def build_row_attn_mask(H: int, W: int) -> torch.Tensor:
    """Bool mask of shape (S, S), S = 1 + H*W + H*W. True = attend allowed.
    Layout indices:
      0                        : CLS
      1 .. 1+H*W-1             : prefix tokens, row-major
      1+H*W .. 1+2*H*W-1       : query tokens, row-major
    Rules:
      CLS row: nothing attends back to "from CLS to others"; CLS attends only to itself.
      Prefix: standard causal among prefix slots, plus all attend to CLS.
      Queries at slot (r, c): attend to CLS + prefix rows 0..r-1.
                              NOT to prefix row r, NOT to any query.
    """
    S = 1 + 2 * H * W
    P0 = 1                # prefix start
    Q0 = 1 + H * W        # query start
    mask = torch.zeros(S, S, dtype=torch.bool)

    # CLS attends to itself.
    mask[0, 0] = True

    # Prefix block: causal among prefix, all see CLS.
    prefix_idx = torch.arange(H * W)
    # CLS column for prefix:
    mask[P0:Q0, 0] = True
    # Lower-triangular among prefix:
    tri = torch.tril(torch.ones(H * W, H * W, dtype=torch.bool))
    mask[P0:Q0, P0:Q0] = tri

    # Query block: per (r, c), attend to CLS + prefix rows 0..r-1.
    # Build a (H*W, H*W) bool block where row q can see prefix row p iff p < r(q).
    # r(q) = q // W
    rows = (torch.arange(H * W) // W).view(-1, 1)        # (H*W, 1)
    pref_rows = (torch.arange(H * W) // W).view(1, -1)   # (1, H*W)
    q_to_prefix = pref_rows < rows                        # (H*W, H*W) bool

    mask[Q0:S, 0] = True
    mask[Q0:S, P0:Q0] = q_to_prefix
    # Queries do not attend to other queries (block stays False).
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
        # Token embeddings for prefix + queries (queries are vertical predecessors).
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # Single learned vector for row-0 query inputs (no previous row exists).
        self.bos_row = nn.Parameter(torch.zeros(config.dim))

        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = nn.ModuleList([RowARBlock(config, dpr[i]) for i in range(config.n_layer)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        head_dim = config.dim // config.n_head
        # 2D RoPE shared between prefix and query positions (so query (r,c) and prefix (r,c)
        # carry the same positional signal). Shape: (1 + 2*H*W, head_dim//2, 2).
        self.register_buffer(
            "freqs_cis",
            build_row_freqs_cis(self.H, self.W, head_dim, base=config.rope_base),
            persistent=False,
        )
        # Attention mask (bool), shape (S, S).
        self.register_buffer(
            "attn_mask",
            build_row_attn_mask(self.H, self.W),
            persistent=False,
        )

        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
        nn.init.constant_(self.output.weight, 0)
        nn.init.normal_(self.bos_row, mean=0.0, std=self.config.initializer_range)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    # ------------------------------------------------------------------ training
    def _build_query_inputs(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, H, W] long. Returns [B, H*W, dim] query embeddings.
        Query for (r, c):
          r == 0: bos_row (broadcast to W)
          r >  0: tok_emb(tokens[:, r-1, c])
        """
        B, H, W = tokens.shape
        # vertical predecessor: shift down by 1 row, fill row 0 with a sentinel.
        prev = torch.empty_like(tokens)
        prev[:, 1:, :] = tokens[:, :-1, :]
        prev[:, 0, :] = 0  # placeholder, gets overwritten below

        emb = self.tok_embeddings(prev)              # [B, H, W, dim]
        # Replace row 0 with bos_row.
        emb[:, 0, :, :] = self.bos_row.view(1, 1, -1).expand(B, W, -1)
        return emb.reshape(B, H * W, -1)

    def forward(
        self,
        tokens: Optional[torch.Tensor],            # [B, H, W] long, training only
        class_idx: Optional[torch.Tensor],         # [B] long
        # inference path:
        prev_rows: Optional[torch.Tensor] = None,  # [B, r, W] long, r in [0, H)
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
        prefix_emb = self.tok_embeddings(tokens.reshape(B, H * W))                           # [B,H*W,d]
        query_emb = self._build_query_inputs(tokens)                                         # [B,H*W,d]

        h = torch.cat([cls_emb, prefix_emb, query_emb], dim=1)                                # [B,S,d]
        h = self.tok_dropout(h)

        mask = self.attn_mask.to(device).unsqueeze(0).unsqueeze(0)                            # [1,1,S,S]
        freqs_cis = self.freqs_cis.to(device)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)
        h = self.norm(h)

        # Logits only at query positions.
        Q0 = 1 + H * W
        logits = self.output(h[:, Q0:, :]).float()                                            # [B,H*W,V]

        if targets is None:
            targets = tokens.reshape(B, H * W)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        return logits, loss

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def reset_kv_cache(self):
        """Clear the per-layer prefix K/V cache used by cached sampling."""
        self._cache_k = [None] * len(self.layers)
        self._cache_v = [None] * len(self.layers)

    def _step_layer_cached(self, layer_idx, layer, x_p, x_q, freqs_p, freqs_q):
        """One transformer block forward for a single RowAR row step with KV cache.

        x_p: [B, L_p, d] new prefix embeddings this step (L_p = 1 at r=0 for CLS,
              L_p = W at r>=1 for the just-committed row). Goes into cache.
        x_q: [B, W, d]   query embeddings for the row being predicted. NOT cached.
        freqs_p: [L_p, head_dim//2, 2] RoPE for prefix positions (absolute in the
                 precomputed table).
        freqs_q: [W, head_dim//2, 2] RoPE for the query positions of this row,
                 which in our 2D scheme equal the prefix-slot RoPE of row r.

        Returns updated (x_p, x_q). Side-effect: appends new K/V to self._cache_{k,v}.
        """
        att = layer.attention
        B, L_p, _ = x_p.shape
        W = x_q.shape[1]
        n_head = att.n_head
        n_kv = att.n_kv_head
        hd = att.head_dim
        dim = att.dim
        kv_sz = n_kv * hd

        # ---- attention block ----
        xn_p = layer.attention_norm(x_p)
        xn_q = layer.attention_norm(x_q)

        qkv_p = att.wqkv(xn_p)
        qkv_q = att.wqkv(xn_q)
        q_p, k_p, v_p = qkv_p.split([dim, kv_sz, kv_sz], dim=-1)
        q_q, k_q, v_q = qkv_q.split([dim, kv_sz, kv_sz], dim=-1)
        q_p = q_p.view(B, L_p, n_head, hd); k_p = k_p.view(B, L_p, n_kv, hd); v_p = v_p.view(B, L_p, n_kv, hd)
        q_q = q_q.view(B, W,   n_head, hd); k_q = k_q.view(B, W,   n_kv, hd); v_q = v_q.view(B, W,   n_kv, hd)

        q_p = apply_rotary_emb(q_p, freqs_p); k_p = apply_rotary_emb(k_p, freqs_p)
        q_q = apply_rotary_emb(q_q, freqs_q); k_q = apply_rotary_emb(k_q, freqs_q)

        # [B, head, L, D]
        q_p = q_p.transpose(1, 2); k_p = k_p.transpose(1, 2); v_p = v_p.transpose(1, 2)
        q_q = q_q.transpose(1, 2); k_q = k_q.transpose(1, 2); v_q = v_q.transpose(1, 2)

        # ---- append new prefix K/V to external cache ----
        past_k = self._cache_k[layer_idx]
        past_v = self._cache_v[layer_idx]
        if past_k is None:
            full_k = k_p
            full_v = v_p
        else:
            full_k = torch.cat([past_k, k_p], dim=2)
            full_v = torch.cat([past_v, v_p], dim=2)
        self._cache_k[layer_idx] = full_k
        self._cache_v[layer_idx] = full_v

        # Repeat KV heads up to n_head for GQA support.
        rep = n_head // n_kv
        K_prefix = full_k.repeat_interleave(rep, dim=1) if rep > 1 else full_k
        V_prefix = full_v.repeat_interleave(rep, dim=1) if rep > 1 else full_v

        # ---- prefix self-attn: causal over (L_prev + L_p) ----
        L_prev = 0 if past_k is None else past_k.shape[2]
        Lk = L_prev + L_p
        mask_p = torch.zeros(L_p, Lk, dtype=torch.bool, device=x_p.device)
        mask_p[:, :L_prev] = True  # past fully visible
        mask_p[:, L_prev:] = torch.tril(torch.ones(L_p, L_p, dtype=torch.bool, device=x_p.device))
        out_p = F.scaled_dot_product_attention(
            q_p, K_prefix, V_prefix, attn_mask=mask_p.view(1, 1, L_p, Lk), is_causal=False
        )

        # ---- query self-attn: attend to full prefix cache, no query-to-query ----
        # No explicit mask needed: keys are exactly the cache (prefix only).
        out_q = F.scaled_dot_product_attention(
            q_q, K_prefix, V_prefix, is_causal=False
        )

        out_p = out_p.transpose(1, 2).contiguous().view(B, L_p, dim)
        out_q = out_q.transpose(1, 2).contiguous().view(B, W, dim)
        out_p = att.resid_dropout(att.wo(out_p))
        out_q = att.resid_dropout(att.wo(out_q))
        x_p = x_p + out_p
        x_q = x_q + out_q

        # ---- FFN ----
        x_p = x_p + layer.feed_forward(layer.ffn_norm(x_p))
        x_q = x_q + layer.feed_forward(layer.ffn_norm(x_q))
        return x_p, x_q

    @torch.no_grad()
    def _forward_step(self, prev_rows: Optional[torch.Tensor], class_idx: torch.Tensor):
        """One row step with persistent KV cache.

        Call order: call this H times with prev_rows length 0, 1, ..., H-1.
        The cache is grown internally; call reset_kv_cache() before a new sample.
        """
        B = class_idx.shape[0]
        device = class_idx.device
        H, W = self.H, self.W
        if not hasattr(self, "_cache_k") or self._cache_k is None or len(self._cache_k) == 0:
            self.reset_kv_cache()
        r = 0 if prev_rows is None else prev_rows.shape[1]
        assert r <= H

        freqs = self.freqs_cis.to(device)

        # ---- new prefix tokens for this step ----
        if r == 0:
            cls_emb = self.cls_embedding(class_idx, train=False)[:, :self.cls_token_num]
            new_prefix_emb = cls_emb.to(freqs.dtype if False else cls_emb.dtype)  # [B, 1, d]
            new_prefix_pos = torch.tensor([0], device=device)
        else:
            last_row = prev_rows[:, -1, :]                          # [B, W]
            new_prefix_emb = self.tok_embeddings(last_row)          # [B, W, d]
            start = 1 + (r - 1) * W                                 # prefix index of row r-1
            new_prefix_pos = torch.arange(start, start + W, device=device)

        # ---- query embeddings for this row ----
        if r == 0:
            query_emb = self.bos_row.view(1, 1, -1).expand(B, W, -1).to(new_prefix_emb.dtype)
        else:
            last_row = prev_rows[:, -1, :]
            query_emb = self.tok_embeddings(last_row)               # [B, W, d]
        q_start_in_prefix = 1 + r * W                               # slot (r, 0) in prefix indexing
        query_pos = torch.arange(q_start_in_prefix, q_start_in_prefix + W, device=device)

        freqs_p = freqs[new_prefix_pos]
        freqs_q = freqs[query_pos]

        x_p = self.tok_dropout(new_prefix_emb)
        x_q = self.tok_dropout(query_emb)

        for i, layer in enumerate(self.layers):
            x_p, x_q = self._step_layer_cached(i, layer, x_p, x_q, freqs_p, freqs_q)

        x_q = self.norm(x_q)
        logits = self.output(x_q).float()                           # [B, W, V]
        return logits

    # ------------------------------------------------------------------ ckpt I/O
    def load_llamagen_state_dict(self, state_dict: dict, verbose: bool = True):
        """Map a LlamaGen c2i checkpoint into RowAR. Returns (loaded, skipped).

        LlamaGen's c2i Transformer uses the SAME llama block module names as PAR
        (since PAR forked LlamaGen). We can lift:
          cls_embedding.embedding_table.weight
          tok_embeddings.weight
          layers.{i}.attention.{wqkv,wo}.weight
          layers.{i}.feed_forward.{w1,w2,w3}.weight
          layers.{i}.attention_norm.weight
          layers.{i}.ffn_norm.weight
          norm.weight
          output.weight

        Skipped (not present or incompatible):
          freqs_cis (rebuilt), causal_mask (rebuilt),
          spe_tok_embeddings.* (PAR-only), bos_row (new).
        """
        own = self.state_dict()
        loaded, skipped = [], []
        for k, v in state_dict.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                loaded.append(k)
            else:
                skipped.append((k, tuple(v.shape) if hasattr(v, "shape") else None))
        # also report own keys that received nothing (i.e. fresh-init in our model)
        loaded_set = set(loaded)
        fresh = [k for k in own.keys() if k not in loaded_set]
        if verbose:
            print(f"[RowAR.load_llamagen] loaded {len(loaded)} tensors, "
                  f"skipped {len(skipped)} from ckpt, fresh-init {len(fresh)} in model")
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
