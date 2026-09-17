import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import psutil
import torch

from config import ModelConfig, TrainConfig, save_config

GIB = 1024**3


def select_device(value='auto'):
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(value)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: cpu, cuda, cuda:N')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; install a CUDA-enabled PyTorch build')
    return device


def dtype_supported(device, dtype):
    if device.type != 'cuda':
        return dtype == torch.float32
    if dtype == torch.bfloat16:
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                return False
    try:
        x = torch.randn(16, 16, device=device, dtype=dtype, requires_grad=True)
        (x @ x).float().square().mean().backward()
        torch.cuda.synchronize(device)
        return bool(torch.isfinite(x.grad).all())
    except RuntimeError:
        return False


def precision_for(device, requested='auto'):
    choices = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}
    if requested != 'auto':
        if not dtype_supported(device, choices[requested]):
            raise ValueError(f'{requested} is unavailable on {device}')
        return choices[requested]
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        if dtype_supported(device, dtype):
            return dtype
    raise RuntimeError('No working floating-point dtype')


def detect_hardware(device='auto'):
    device = select_device(device)
    mem = psutil.virtual_memory()
    result = {'device': str(device), 'torch': str(torch.__version__), 'cuda_build': torch.version.cuda,
              'cuda_available': torch.cuda.is_available(), 'ram_gib': round(mem.total/GIB, 2),
              'ram_available_gib': round(mem.available/GIB, 2), 'cpu_threads': psutil.cpu_count(logical=False) or 1,
              'bf16': dtype_supported(device, torch.bfloat16), 'fp16': dtype_supported(device, torch.float16)}
    result['gpus'] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            result['gpus'].append({'index': i, 'name': properties.name,
                                  'vram_gib': round(total/GIB, 2), 'free_vram_gib': round(free/GIB, 2),
                                  'compute_capability': list(torch.cuda.get_device_capability(i))})
    return result


def estimated_training_gib(c, batch_size):
    # FP32 weights, gradients, Adam moments; approximate activations and workspace.
    params = c.parameter_count() * 20
    stored_layers = 3 if c.gradient_checkpointing else c.n_layers
    activations = batch_size*c.context_length*(c.embedding_dim*stored_layers*24 + c.vocab_size*12)
    return (params + activations) / GIB + 1.0


def make_presets(hw, vocab_size=32768):
    cuda = hw['device'].startswith('cuda')
    index = int(hw['device'].split(':')[1]) if ':' in hw['device'] else 0
    available = hw['gpus'][index]['free_vram_gib'] if cuda else hw['ram_available_gib']
    budget = available * (0.8 if cuda else 0.4)
    candidates = [(4,256,4,2,704,512), (8,512,8,2,1408,1024),
                  (12,768,12,4,2048,1024), (16,1024,16,4,2816,2048),
                  (20,1280,20,4,3456,2048), (24,1536,24,6,4096,2048)]
    models = []
    for layers, dim, heads, kv, ffn, ctx in candidates:
        c = ModelConfig(vocab_size, layers, heads, kv, dim, ffn, ctx,
                        gradient_checkpointing=cuda)
        if estimated_training_gib(c, 1) <= budget:
            models.append(c)
    if not models:
        models = [ModelConfig(vocab_size, 2, 4, 2, 128, 352, 256)]
    if not cuda:
        models = models[:2]
    # Single consumer GPUs should leave room for throughput, not just weights.
    if cuda and available < 18:
        models = models[:4]
    test = ModelConfig(vocab_size, 2, 4, 2, 128, 352, 256)
    balanced = models[max(0, len(models)-2)]
    maximum = models[-1]
    output = {}
    for name, c in [('test', test), ('balanced', balanced), ('maximum_practical', maximum)]:
        batch = 1 if name != 'balanced' or not cuda else 2
        if estimated_training_gib(c, batch) > budget:
            batch = 1
        t = TrainConfig(batch_size=batch, grad_accum=max(1, 16384//(batch*c.context_length)),
                        steps=100 if name == 'test' else 10000,
                        warmup_steps=10 if name == 'test' else 200)
        if name == 'test':
            t.grad_accum = 1
        output[name] = (c, t)
    return output


def probe(c, t, device):
    from model import Transformer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    dtype = precision_for(device, t.precision)
    start = time.perf_counter()
    model = Transformer(c).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, foreach=False)
    x = torch.randint(c.vocab_size, (t.batch_size, c.context_length), device=device)
    with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        _, loss, _ = model(x, x)
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    result = {'parameters': sum(p.numel() for p in model.parameters()), 'dtype': str(dtype),
              'loss': float(loss.detach()), 'seconds': round(time.perf_counter()-start, 3),
              'peak_allocated_gib': round(torch.cuda.max_memory_allocated(device)/GIB, 3) if device.type == 'cuda' else None,
              'peak_reserved_gib': round(torch.cuda.max_memory_reserved(device)/GIB, 3) if device.type == 'cuda' else None}
    del model, optimizer, x, loss, scaler
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='auto')
    p.add_argument('--vocab-size', type=int, default=32768)
    p.add_argument('--output', default='configs')
    p.add_argument('--probe', action='store_true')
    a = p.parse_args()
    hw = detect_hardware(a.device)
    presets = make_presets(hw, a.vocab_size)
    report = {'hardware': hw, 'presets': {}}
    for name, (c, t) in presets.items():
        row = {'parameters': c.parameter_count(), 'estimated_training_gib': round(estimated_training_gib(c,t.batch_size), 2)}
        if a.probe:
            try:
                row['measured'] = probe(c, t, select_device(a.device))
            except torch.cuda.OutOfMemoryError:
                raise SystemExit(f'{name}: VRAM exhausted. Close GPU programs or reduce config and rerun.')
        save_config(Path(a.output)/f'{name}.json', c, t)
        report['presets'][name] = row
        print(f'{name}: {json.dumps(row)}', flush=True)
    Path(a.output, 'hardware.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(hw, indent=2))


if __name__ == '__main__':
    main()
