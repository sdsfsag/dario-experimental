import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


class RoPE(nn.Module):
    def __init__(self, head_dim, context_length, theta):
        super().__init__()
        freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(context_length).float(), freq)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

    def forward(self, x, offset=0):
        cos = self.cos[offset:offset+x.size(-2)].to(x.dtype)[None, None]
        sin = self.sin[offset:offset+x.size(-2)].to(x.dtype)[None, None]
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even*cos - odd*sin, even*sin + odd*cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_heads, self.n_kv_heads = c.n_heads, c.n_kv_heads
        self.head_dim = c.embedding_dim // c.n_heads
        self.q_proj = nn.Linear(c.embedding_dim, c.embedding_dim, bias=False)
        self.k_proj = nn.Linear(c.embedding_dim, c.n_kv_heads*self.head_dim, bias=False)
        self.v_proj = nn.Linear(c.embedding_dim, c.n_kv_heads*self.head_dim, bias=False)
        self.out_proj = nn.Linear(c.embedding_dim, c.embedding_dim, bias=False)
        self.rope = RoPE(self.head_dim, c.context_length, c.rope_theta)
        self.dropout = c.dropout

    def forward(self, x, cache=None, use_cache=False):
        b, t, _ = x.shape
        offset = 0 if cache is None else cache[0].size(2)
        q = self.rope(self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2), offset)
        k = self.rope(self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2), offset)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if cache is not None:
            k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
        new_cache = (k, v) if use_cache else None
        repeats = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        mask = None
        if offset and t > 1:
            mask = torch.arange(k.size(2), device=x.device)[None, :] <= (offset + torch.arange(t, device=x.device))[:, None]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0, is_causal=offset == 0)
        y = y.transpose(1, 2).contiguous().view(b, t, -1)
        return self.out_proj(y), new_cache


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Linear(c.embedding_dim, c.ffn_dim, bias=False)
        self.up = nn.Linear(c.embedding_dim, c.ffn_dim, bias=False)
        self.down = nn.Linear(c.ffn_dim, c.embedding_dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = RMSNorm(c.embedding_dim, c.norm_eps)
        self.attention = Attention(c)
        self.ffn_norm = RMSNorm(c.embedding_dim, c.norm_eps)
        self.ffn = SwiGLU(c)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x, cache=None, use_cache=False):
        y, cache = self.attention(self.attn_norm(x), cache, use_cache)
        x = x + self.dropout(y)
        return x + self.dropout(self.ffn(self.ffn_norm(x))), cache


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.embedding_dim, config.norm_eps)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(('out_proj.weight', 'down.weight')):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens, targets=None, caches=None, use_cache=False, last_only=False):
        if tokens.ndim != 2 or tokens.size(1) == 0:
            raise ValueError('tokens must have shape [batch, nonempty sequence]')
        offset = 0 if caches is None else caches[0][0].size(2)
        if tokens.size(1) + offset > self.config.context_length:
            raise ValueError('Context length exceeded')
        if caches is not None and len(caches) != len(self.blocks):
            raise ValueError('Incorrect cache count')
        if targets is not None and (targets.shape != tokens.shape or last_only):
            raise ValueError('Targets must match tokens; last_only cannot be used with loss')
        x = self.embedding(tokens)
        new_caches = []
        for i, block in enumerate(self.blocks):
            if self.training and self.config.gradient_checkpointing and not use_cache:
                x = checkpoint(lambda h, layer=block: layer(h)[0], x, use_reentrant=False)
            else:
                x, cache = block(x, None if caches is None else caches[i], use_cache)
                if use_cache:
                    new_caches.append(cache)
        x = self.norm(x[:, -1:] if last_only else x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size),
                                   targets.reshape(-1), ignore_index=-100)
        return logits, loss, new_caches if use_cache else None


def sample_token(logits, history, temperature=0.8, top_k=40, top_p=0.95,
                 repetition_penalty=1.1, generator=None, forbidden_ids=()):
    if temperature < 0 or top_k < 0 or not 0 < top_p <= 1 or repetition_penalty < 1:
        raise ValueError('Invalid sampling parameters')
    scores = logits.float().clone()
    if history and repetition_penalty != 1:
        ids = torch.tensor(list(set(history)), device=scores.device)
        values = scores[ids]
        scores[ids] = torch.where(values < 0, values * repetition_penalty, values / repetition_penalty)
    if forbidden_ids:
        scores[list(forbidden_ids)] = -float('inf')
    if not torch.isfinite(scores).any() or torch.isnan(scores).any() or torch.isposinf(scores).any():
        raise FloatingPointError('Invalid generation logits')
    if temperature == 0:
        return int(scores.argmax())
    scores /= temperature
    if top_k:
        scores[scores < torch.topk(scores, min(top_k, scores.numel())).values[-1]] = -float('inf')
    if top_p < 1:
        values, indices = scores.sort(descending=True)
        remove = values.softmax(-1).cumsum(-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        scores[indices[remove]] = -float('inf')
    return int(torch.multinomial(scores.softmax(-1), 1, generator=generator))


@torch.inference_mode()
def generate(model, prompt, max_new_tokens=256, stop_ids=(), generator=None, **sampling):
    if not prompt or max_new_tokens < 1:
        raise ValueError('Empty prompt or invalid max_new_tokens')
    if len(prompt) + max_new_tokens > model.config.context_length:
        raise ValueError('Reserve space for generation within context_length')
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    history = list(prompt)
    current = torch.tensor([prompt], dtype=torch.long, device=device)
    caches = None
    try:
        for _ in range(max_new_tokens):
            logits, _, caches = model(current, caches=caches, use_cache=True, last_only=True)
            token = sample_token(logits[0, -1], history[-256:], generator=generator, **sampling)
            if token in stop_ids:
                break
            yield token
            history.append(token)
            current = torch.tensor([[token]], device=device)
    finally:
        model.train(was_training)
