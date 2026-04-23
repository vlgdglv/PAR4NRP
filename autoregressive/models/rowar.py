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

1D RoPE over the flat sequence (LlamaGen convention):
  Position p in [0, 1+H*W) gets standard llama-style 1D RoPE at index p.
  CLS at p=0 receives identity rotation. The 'row' structure of RowAR is
  enforced ONLY by the attention mask, not by positional encoding -- this
  keeps Q/K projections in the same rotation regime they were warm-started
  in, so attention patterns transfer instead of being scrambled by a 2D split.

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

    # Tiny within-row AR head. Trunk predicts row r in parallel (factorized
    # marginals); head walks across the W columns and conditions on previously
    # sampled tokens of the same row, so inference is no longer factorized.
    use_head: bool = False
    head_dim: int = 384
    head_n_layer: int = 2
    head_n_head: int = 6
    # Loss weight on trunk's own factorized CE. Head loss is always 1.0.
    # Keeping trunk loss > 0 prevents the trunk from "outsourcing" all
    # within-row work to the head and stops its hidden states from drifting
    # into an arbitrary internal code only the head can read.
    trunk_loss_weight: float = 1.0

    # GLAT (glancing training, Qian et al. 2021) --- single-forward inference,
    # two-pass training. Pass 1 (no grad): factorized predictions on each row,
    # count Hamming mismatches vs GT. Pass 2: reveal a fraction of GT tokens
    # (proportional to pass-1 error) by adding tok_emb(row_r) + reveal_flag_emb
    # to the block-r input at revealed positions. CE is taken only on the
    # un-revealed positions, so the trunk is still trained to predict row-r
    # marginals from x_<r alone, but its hidden states are forced to encode
    # partial-row joint structure --- which carries through to inference (no
    # reveals) and tightens the factorized marginals against each other.
    use_glat: bool = True
    glat_lambda: float = 0.5


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
    """1D RoPE over the flat sequence [CLS, block_0, block_1, ..., block_{H-1}],
    length 1 + H*W. Matches LlamaGen's standard 1D positional convention so that
    attention weights warm-started from a c2i raster checkpoint operate in the
    same rotation regime they were trained in. The 'row' structure of RowAR is
    expressed entirely through the block-causal attention mask, not through RoPE.

    Returns tensor of shape (1 + H*W, head_dim // 2, 2).
    Position 0 (CLS) gets zero angle (cos=1, sin=0 -> identity rotation), which
    is the natural index-0 case of the formula -- no special handling needed.
    """
    assert head_dim % 2 == 0
    half = head_dim // 2
    # Llama-style: freq_i = 1 / base^(2i/d), i in [0, half)
    freqs = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * 2 / head_dim))
    S = 1 + H * W
    t = torch.arange(S, dtype=torch.float32)
    angles = torch.outer(t, freqs)                                   # (S, half)
    cache = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (S, half, 2)
    return cache


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


class TinyBlock(nn.Module):
    """Minimal pre-norm transformer block, causal SDPA, GELU MLP."""

    def __init__(self, dim: int, n_head: int):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        self.ff_w1 = nn.Linear(dim, 4 * dim, bias=False)
        self.ff_w2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        B, L, _ = h.shape
        qkv = self.qkv(h).view(B, L, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        x = x + self.wo(out)
        x = x + self.ff_w2(F.gelu(self.ff_w1(self.norm2(x))))
        return x


class TinyARHead(nn.Module):
    """Within-row autoregressive head.

    Inputs per column c (0 <= c < W):
      cond_c  = proj_in(h_row[:, c, :])           # trunk's row hidden state at col c
      tok_c   = bos                  if c == 0
                tok_emb(x_{r, c-1})  if c >= 1    # previous token in same row
      x_c     = cond_c + tok_c + pos_emb[c]

    Causal self-attention over the W positions; each column attends to itself
    and prior columns. Output: per-column logits over the VQ vocab.

    Why additive cond + tok rather than concat: keeps the head dim fixed and
    matches the magnitudes the trunk hidden states already live in. Direct CE
    supervision on the head output ensures gradient flows into tok_emb / qkv
    of the head; nothing prevents the head from using prev tokens, and it
    will, because that's the only path to lower CE on positions where the
    factorized trunk is wrong.

    Inference: KV cache grows by one position per column. W steps per row.
    Cost ~5% of trunk per row at d_head=384, n_layer=2.
    """

    def __init__(
        self,
        d_trunk: int,
        d_head: int,
        n_layer: int,
        n_head: int,
        vocab_size: int,
        W: int,
        initializer_range: float = 0.02,
    ):
        super().__init__()
        self.W = W
        self.d_head = d_head
        self.n_head = n_head
        self.head_dim_attn = d_head // n_head
        self.proj_in = nn.Linear(d_trunk, d_head, bias=False)
        self.tok_emb = nn.Embedding(vocab_size, d_head)
        self.bos = nn.Parameter(torch.zeros(d_head))
        self.pos_emb = nn.Parameter(torch.zeros(W, d_head))
        self.layers = nn.ModuleList([TinyBlock(d_head, n_head) for _ in range(n_layer)])
        self.norm = RMSNorm(d_head)
        self.out = nn.Linear(d_head, vocab_size, bias=False)

        # Standard small-init; head is fresh (no warm-start source).
        std = initializer_range
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=std)
        nn.init.normal_(self.bos, mean=0.0, std=std)
        nn.init.normal_(self.pos_emb, mean=0.0, std=std)
        nn.init.normal_(self.proj_in.weight, mean=0.0, std=std)
        nn.init.normal_(self.out.weight, mean=0.0, std=std)
        for blk in self.layers:
            nn.init.normal_(blk.qkv.weight, mean=0.0, std=std)
            nn.init.normal_(blk.wo.weight, mean=0.0, std=std)
            nn.init.normal_(blk.ff_w1.weight, mean=0.0, std=std)
            nn.init.normal_(blk.ff_w2.weight, mean=0.0, std=std)

    def _assemble_inputs(self, cond: torch.Tensor, prev_tokens: torch.Tensor) -> torch.Tensor:
        """cond: [B, L, d_head]; prev_tokens: [B, L] long where prev_tokens[:, 0] is unused
        (replaced by bos). Returns [B, L, d_head].
        """
        B, L = prev_tokens.shape
        tok = self.tok_emb(prev_tokens)                 # [B, L, d_head]
        tok = tok.clone()
        tok[:, 0] = self.bos
        return cond + tok + self.pos_emb[:L].unsqueeze(0)

    def forward(self, h_row: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced parallel forward (training).
        h_row:  [B, W, d_trunk]   trunk hidden state at this row's W positions
        tokens: [B, W] long       ground-truth row (column c sees tokens[:, c-1])
        returns logits [B, W, V]
        """
        B, W = tokens.shape
        prev = torch.zeros_like(tokens)
        prev[:, 1:] = tokens[:, :-1]                    # col 0 prev: dummy, overwritten by bos
        cond = self.proj_in(h_row)
        x = self._assemble_inputs(cond, prev)
        for layer in self.layers:
            x = layer(x)
        return self.out(self.norm(x))                   # [B, W, V]

    @torch.no_grad()
    def step_logits(self, h_row: torch.Tensor, sampled_so_far: torch.Tensor) -> torch.Tensor:
        """Logits at the next column to sample.
        h_row: [B, W, d_trunk]
        sampled_so_far: [B, c] long, 0 <= c < W
        returns [B, V] logits at column c.

        Simple O(W^2) implementation: re-runs first c+1 positions each call.
        At W=16 with a 2-layer dim-384 head this is well under 1ms / row;
        we don't bother with KV cache yet.
        """
        B = h_row.shape[0]
        c = sampled_so_far.shape[1]
        L = c + 1
        device = h_row.device
        prev = torch.zeros(B, L, dtype=torch.long, device=device)
        if c > 0:
            prev[:, 1:] = sampled_so_far                # col 0 prev: dummy -> bos
        cond = self.proj_in(h_row[:, :L])
        x = self._assemble_inputs(cond, prev)
        for layer in self.layers:
            x = layer(x)
        return self.out(self.norm(x[:, -1]))            # [B, V]


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

        self.use_head = config.use_head
        if self.use_head:
            self.head = TinyARHead(
                d_trunk=config.dim,
                d_head=config.head_dim,
                n_layer=config.head_n_layer,
                n_head=config.head_n_head,
                vocab_size=config.vocab_size,
                W=config.grid_w,
                initializer_range=config.initializer_range,
            )

        # GLAT (random-reveal training): no extra parameters. Revealed-position
        # inputs are simply REPLACED with the same-row GT tok_emb (canonical
        # NAT/CMLM design). Bidirectional within-row attention propagates the
        # GT info to un-revealed positions; loss is masked at revealed slots.
        self.use_glat = config.use_glat
        self.glat_lambda = config.glat_lambda

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
    def _build_block_inputs(self, tokens: torch.Tensor, reveal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """tokens: [B, H, W] long. Returns [B, H*W, dim] block inputs.
        Base: block r input at col c = tok_emb(tokens[r-1, c]) for r >= 1, bos_row for r = 0.
        If reveal_mask: [B, H, W] bool is provided (GLAT), the revealed positions
        have their input REPLACED with tok_emb(tokens[r, c]) -- the same-row GT
        embedding -- following the canonical NAT MT / CMLM / MaskGIT design. No
        flag/marker is needed: bidirectional within-row attention lets the
        un-revealed positions read these GT embeddings, and the loss is masked
        at revealed positions so the model can't shortcut by copying input -> output.
        """
        B, H, W = tokens.shape
        prev = torch.empty_like(tokens)
        prev[:, 1:, :] = tokens[:, :-1, :]
        prev[:, 0, :] = 0  # placeholder, overwritten below by bos_row
        emb = self.tok_embeddings(prev)              # [B, H, W, dim]
        emb[:, 0, :, :] = self.bos_row.view(1, 1, -1).expand(B, W, -1)
        if reveal_mask is not None and getattr(self, "use_glat", False):
            cur_emb = self.tok_embeddings(tokens)                                  # [B, H, W, dim]
            m = reveal_mask.unsqueeze(-1)                                           # [B, H, W, 1]
            emb = torch.where(m, cur_emb, emb)
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
            if getattr(self, "use_glat", False):
                return self._forward_train_glat(tokens, class_idx, targets)
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

        # Trunk hidden states at data positions (skip CLS): [B, H, W, d]
        h_data = h[:, 1:, :].view(B, H, W, -1)
        d = h_data.shape[-1]

        trunk_logits = self.output(h_data).float()                                            # [B,H,W,V]
        if targets is None:
            targets = tokens.reshape(B, H * W)
        targets_hw = targets.view(B, H * W)
        trunk_loss = F.cross_entropy(
            trunk_logits.reshape(-1, trunk_logits.size(-1)), targets_hw.reshape(-1)
        )

        if self.use_head:
            # Run head on all H rows in parallel: pack rows into the batch dim.
            h_row = h_data.reshape(B * H, W, d)                                               # [B*H,W,d]
            row_tokens = tokens.reshape(B * H, W)                                             # [B*H,W]
            head_logits = self.head(h_row, row_tokens).float()                                # [B*H,W,V]
            head_loss = F.cross_entropy(
                head_logits.reshape(-1, head_logits.size(-1)),
                row_tokens.reshape(-1),
            )
            loss = self.config.trunk_loss_weight * trunk_loss + head_loss
            # Expose components for logging.
            self.last_loss_components = {
                "loss_trunk": trunk_loss.detach(),
                "loss_head": head_loss.detach(),
            }
            # Return head logits as the "primary" logits the caller sees.
            logits = head_logits.view(B, H, W, -1).reshape(B, H * W, -1)
        else:
            loss = trunk_loss
            self.last_loss_components = {"loss_trunk": trunk_loss.detach()}
            logits = trunk_logits.reshape(B, H * W, -1)

        return logits, loss

    # ---------------------------------------------------------------- GLAT
    def _forward_train_glat(self, tokens, class_idx, targets):
        """GLAT-lite: single-pass random-reveal training.

        Per row, sample reveal rate r ~ Uniform(0, glat_lambda). Bernoulli(r)
        per column: revealed positions get an additive same-row-GT channel
        (tok_emb(row_r) + reveal_flag_emb) on top of the default prev-row
        input. CE on un-revealed positions only.

        Why not GLAT-proper (two-pass, error-adaptive reveal): VQ-16K top-1 is
        noise-dominated (visually-near-equivalent codebook entries), so the
        pass-1 Hamming signal is unreliable and reveal_frac stays pinned high.
        Random reveal with bounded r_max gives the same "learn marginals under
        partial row evidence" pressure at half the compute and without the
        noisy-error dependency. r ~ U(0, r_max) includes r=0 (pure original
        CE) with positive density, so inference (no reveals) is part of the
        training distribution.
        """
        B, H, W = tokens.shape
        assert (H, W) == (self.H, self.W)
        device = tokens.device

        r_max = self.glat_lambda                                                       # 0.5 default
        r_per_row = torch.rand(B, H, 1, device=device) * r_max                         # [B, H, 1]
        reveal_mask = torch.bernoulli(r_per_row.expand(B, H, W)).bool()                # [B, H, W]

        cls_emb = self.cls_embedding(class_idx, train=self.training)[:, :self.cls_token_num]  # [B,1,d]
        block_inputs = self._build_block_inputs(tokens, reveal_mask=reveal_mask)              # [B,H*W,d]
        h = torch.cat([cls_emb, block_inputs], dim=1)
        h = self.tok_dropout(h)
        attn_mask = self.attn_mask.to(device).unsqueeze(0).unsqueeze(0)
        freqs_cis = self.freqs_cis.to(device)
        for layer in self.layers:
            h = layer(h, freqs_cis, attn_mask)
        h = self.norm(h)
        logits = self.output(h[:, 1:, :]).float()                                      # [B, H*W, V]

        V = logits.size(-1)
        if targets is None:
            targets = tokens.reshape(B, H * W)
        loss_keep = (~reveal_mask).view(B, H * W).float()                              # 1 = un-revealed
        ce = F.cross_entropy(
            logits.reshape(-1, V), targets.reshape(-1), reduction="none"
        ).view(B, H * W)
        n_keep = loss_keep.sum().clamp(min=1.0)
        loss = (ce * loss_keep).sum() / n_keep

        self.last_loss_components = {
            "loss_trunk": loss.detach(),
            "glat_reveal_frac": reveal_mask.float().mean().detach(),
        }
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
        """One row step with persistent KV cache. Returns the trunk's row hidden
        state h_row of shape [B, W, d_trunk] -- NOT logits.

        With the AR head present, sampling is no longer "argmax over W
        independent marginals". The caller must run the head column-by-column:

            h_row = model.forward(None, class_idx, prev_rows=prev_rows)
            sampled = empty(B, 0)
            for c in range(W):
                logits_c = model.head.step_logits(h_row, sampled)   # [B, V]
                tok_c = sample(logits_c, ...)
                sampled = cat([sampled, tok_c.unsqueeze(1)], dim=1)
            prev_rows = cat([prev_rows, sampled.unsqueeze(1)], dim=1)

        For CFG: batch the (cond, uncond) class indices into dim 0 (size 2B),
        run one trunk forward, split h_row, run head.step_logits on each, and
        mix logits per column before sampling.

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
        # Return trunk row hidden states at block-r positions (drop CLS at r==0).
        h_row = x[:, 1:, :] if r == 0 else x                                                 # [B, W, d_trunk]
        return h_row

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
