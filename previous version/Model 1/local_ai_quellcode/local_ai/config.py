import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    embedding_dim: int = 768
    ffn_dim: int = 2048
    context_length: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    gradient_checkpointing: bool = False

    def __post_init__(self):
        for name in ('vocab_size', 'n_layers', 'n_heads', 'n_kv_heads',
                     'embedding_dim', 'ffn_dim', 'context_length'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.embedding_dim % self.n_heads or self.n_heads % self.n_kv_heads:
            raise ValueError('embedding_dim / n_heads and n_heads / n_kv_heads must be integers')
        if (self.embedding_dim // self.n_heads) % 2:
            raise ValueError('RoPE needs an even head dimension')
        if not 0 <= self.dropout < 1 or self.context_length < 2:
            raise ValueError('Invalid dropout or context_length')
        if self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError('rope_theta and norm_eps must be positive')

    def parameter_count(self):
        d = self.embedding_dim
        kv = d * self.n_kv_heads // self.n_heads
        return self.vocab_size * d + self.n_layers * (2*d*d + 2*d*kv + 3*d*self.ffn_dim + 2*d) + d


@dataclass
class TrainConfig:
    steps: int = 10000
    batch_size: int = 2
    grad_accum: int = 8
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    save_interval: int = 500
    log_interval: int = 10
    seed: int = 42
    precision: str = 'auto'

    def __post_init__(self):
        for name in ('steps', 'batch_size', 'grad_accum', 'eval_interval',
                     'eval_batches', 'save_interval', 'log_interval'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.warmup_steps < 0 or self.lr <= 0 or self.grad_clip <= 0:
            raise ValueError('Invalid optimizer settings')
        if not 0 <= self.min_lr_ratio <= 1 or self.weight_decay < 0:
            raise ValueError('Invalid decay settings')
        if self.precision not in ('auto', 'fp32', 'bf16', 'fp16'):
            raise ValueError('Unknown precision')


def save_config(path, model, training):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({'model': asdict(model), 'training': asdict(training)}, indent=2), encoding='utf-8')


def load_config(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return ModelConfig(**data['model']), TrainConfig(**data['training'])
