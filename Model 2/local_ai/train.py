import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from config import ModelConfig, TrainConfig, load_config
from data import TokenDataset
from hardware import precision_for, select_device
from model import Transformer
from tokenizer import BPETokenizer, BOS, EOS


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_context(device, dtype):
    return torch.autocast(device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()


def make_optimizer(model, config, device):
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': config.weight_decay},
                             {'params': no_decay, 'weight_decay': 0.0}],
                            lr=config.lr, betas=(0.9, 0.95), eps=1e-8,
                            fused=device.type == 'cuda')


def make_scheduler(optimizer, config):
    warmup = min(config.warmup_steps, max(0, config.steps-1))
    def factor(step):
        if step < warmup:
            return (step+1)/warmup
        ratio = min(1.0, (step-warmup)/max(1, config.steps-warmup-1))
        return config.min_lr_ratio + (1-config.min_lr_ratio)*0.5*(1+math.cos(math.pi*ratio))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def capture_rng():
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Exact resume requires the same CUDA device count')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def load_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format')
    return payload


def supervision_count(targets, reduction):
    if reduction == 'token':
        return int((targets != -100).sum())
    if reduction != 'example':
        raise ValueError('Unknown loss reduction')
    if not torch.all((targets != -100).any(dim=1)):
        raise ValueError('An example has no supervised tokens')
    return targets.shape[0]


def supervised_loss(model, tokens, targets, reduction='token'):
    if reduction == 'token':
        return model(tokens, targets)[1]
    supervision_count(targets, reduction)
    logits = model(tokens)[0]
    losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                             targets.reshape(-1), ignore_index=-100, reduction='none')
    # Short factual answers carry the same example weight as long responses.
    return (losses.reshape_as(targets).sum(1)/(targets != -100).sum(1)).mean()


def train_update(model, optimizer, scaler, dataset, generator, config, device, dtype):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batches = [dataset.batch(config.batch_size, generator) for _ in range(config.grad_accum)]
    reduction = getattr(dataset, 'loss_reduction', 'token')
    valid_total = sum(supervision_count(y, reduction) for _, y in batches)
    if valid_total == 0:
        raise ValueError('Batch has no supervised tokens')
    loss_total, tokens = 0.0, 0
    for x, y in batches:
        weight = supervision_count(y, reduction)/valid_total
        x, y = x.to(device), y.to(device)
        with amp_context(device, dtype):
            loss = supervised_loss(model, x, y, reduction)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite loss; last checkpoint is preserved')
        scaler.scale(loss*weight).backward()
        loss_total += float(loss.detach())*weight
        tokens += x.numel()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    old_scale = scaler.get_scale()
    if not torch.isfinite(norm) and not scaler.is_enabled():
        raise FloatingPointError('Non-finite gradients; last checkpoint is preserved')
    scaler.step(optimizer)
    scaler.update()
    updated = scaler.get_scale() >= old_scale
    return loss_total, tokens, float(norm), updated


@torch.no_grad()
def evaluate(model, dataset, config, device, dtype):
    model.eval()
    generator = torch.Generator().manual_seed(config.seed+123456)
    reduction = getattr(dataset, 'loss_reduction', 'token')
    total, count = 0.0, 0
    for _ in range(config.eval_batches):
        x, y = dataset.batch(config.batch_size, generator)
        n = supervision_count(y, reduction)
        with amp_context(device, dtype):
            loss = supervised_loss(model, x.to(device), y.to(device), reduction)
        total += float(loss)*n
        count += n
    model.train()
    result = total/max(count, 1)
    if not math.isfinite(result):
        raise FloatingPointError('Non-finite validation loss')
    return result


def overfit_test(tokenizer, device, dtype, max_steps=160):
    seed_all(123)
    ids = [BOS] + tokenizer.encode('Die Katze sitzt auf dem Stuhl. The cat sits on the chair.') + [EOS]
    x = torch.tensor([ids[:-1]], dtype=torch.long)
    y = torch.tensor([ids[1:]], dtype=torch.long)
    class TinyData:
        def batch(self, batch_size, generator):
            return x.repeat(batch_size, 1), y.repeat(batch_size, 1)
    c = ModelConfig(tokenizer.vocab_size, 2, 4, 2, 64, 176, max(128,len(ids)))
    t = TrainConfig(steps=max_steps, batch_size=1, grad_accum=1, lr=0.006,
                    warmup_steps=5, eval_batches=1, weight_decay=0.0)
    model = Transformer(c).to(device)
    optimizer = make_optimizer(model, t, device)
    scheduler = make_scheduler(optimizer, t)
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    dataset = TinyData()
    generator = torch.Generator().manual_seed(123)
    initial = evaluate(model, dataset, t, device, dtype)
    final = initial
    for step in range(max_steps):
        _, _, _, updated = train_update(model, optimizer, scaler, dataset, generator, t, device, dtype)
        if updated:
            scheduler.step()
        if (step+1) % 20 == 0:
            final = evaluate(model, dataset, t, device, dtype)
            if final < min(0.15, initial*0.1):
                break
    report = {'initial_loss': initial, 'final_loss': final, 'steps': step+1,
              'passed': final < initial*0.25 and final < 0.5, 'device': str(device), 'dtype': str(dtype)}
    print(f'Overfit: {json.dumps(report)}', flush=True)
    if not report['passed']:
        raise RuntimeError('Overfit test failed; training is blocked')
    return report


class Trainer:
    def __init__(self, model, config, tokenizer, train_data, val_data, output, device,
                 stage='pretrain', resume=None, overfit=None):
        self.model, self.config, self.tokenizer = model, config, tokenizer
        self.train_data, self.val_data = train_data, val_data
        self.output, self.device, self.stage = Path(output), device, stage
        self.dtype = precision_for(device, config.precision)
        self.optimizer = make_optimizer(model, config, device)
        self.scheduler = make_scheduler(self.optimizer, config)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.dtype == torch.float16)
        self.generator = torch.Generator().manual_seed(config.seed)
        self.step, self.best, self.total_tokens = 0, float('inf'), 0
        self.overfit = overfit
        self.data_hashes = [train_data.fingerprint, val_data.fingerprint]
        self.loss_reduction = getattr(train_data, 'loss_reduction', 'token')
        if self.loss_reduction != getattr(val_data, 'loss_reduction', 'token'):
            raise ValueError('Training/validation loss reduction mismatch')
        self.output.mkdir(parents=True, exist_ok=True)
        if resume is not None:
            if resume.get('loss_reduction', 'token') != self.loss_reduction:
                raise ValueError('Resume loss reduction mismatch')
            if resume['stage'] != stage or resume['tokenizer_hash'] != tokenizer.fingerprint:
                raise ValueError('Checkpoint stage/tokenizer mismatch')
            if resume['model_config'] != asdict(model.config) or resume['training_config'] != asdict(config):
                raise ValueError('Resume configuration differs from checkpoint')
            if resume['data_hashes'] != self.data_hashes:
                raise ValueError('Training/validation data changed since checkpoint')
            if resume['precision'] != str(self.dtype) or resume['device_type'] != device.type:
                raise ValueError('Resume requires the original device type and precision')
            model.load_state_dict(resume['model'])
            self.optimizer.load_state_dict(resume['optimizer'])
            self.scheduler.load_state_dict(resume['scheduler'])
            self.scaler.load_state_dict(resume['scaler'])
            self.generator.set_state(resume['batch_rng'].cpu())
            self.step, self.best, self.total_tokens = resume['step'], resume['best_val'], resume['total_tokens']
            self.overfit = resume['overfit']
            restore_rng(resume['rng'])

    def save(self, name='last.pt'):
        atomic_save({'format_version': 1, 'stage': self.stage, 'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(), 'step': self.step, 'best_val': self.best,
            'total_tokens': self.total_tokens, 'model_config': asdict(self.model.config),
            'training_config': asdict(self.config), 'tokenizer_json': self.tokenizer.backend.to_str(),
            'tokenizer_hash': self.tokenizer.fingerprint, 'rng': capture_rng(),
            'batch_rng': self.generator.get_state(), 'data_hashes': self.data_hashes,
            'loss_reduction': self.loss_reduction,
            'overfit': self.overfit, 'precision': str(self.dtype), 'device_type': self.device.type}, self.output/name)

    def run(self, stop_after=None, max_seconds=None, stop_file=None):
        t = self.config
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError('max_seconds must be positive')
        started = time.perf_counter()
        stop_path = Path(stop_file) if stop_file else None
        target = min(t.steps, stop_after) if stop_after is not None else t.steps
        if target <= self.step:
            raise ValueError('No steps left; --stop-after is an absolute optimizer step')
        metrics_path = self.output/'metrics.jsonl'
        self.save()
        window_start, window_tokens, window_loss, window_steps = time.perf_counter(), 0, 0.0, 0
        skipped = 0
        reason = 'steps'
        try:
            while self.step < target:
                if stop_path and stop_path.exists():
                    reason = 'stop_file'
                    self.save()
                    break
                loss, tokens, norm, updated = train_update(self.model, self.optimizer, self.scaler,
                    self.train_data, self.generator, t, self.device, self.dtype)
                if not updated:
                    skipped += 1
                    if skipped >= 20:
                        raise FloatingPointError('20 consecutive fp16 overflows; use bf16 or fp32')
                    continue
                skipped = 0
                lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                self.step += 1
                self.total_tokens += tokens
                window_tokens += tokens
                window_loss += loss
                window_steps += 1
                time_limit = max_seconds is not None and time.perf_counter()-started >= max_seconds
                requested_stop = stop_path is not None and stop_path.exists()
                stopping = time_limit or requested_stop
                validation = self.step % t.eval_interval == 0 or self.step == target or stopping
                if self.step % t.log_interval == 0 or validation:
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize(self.device)
                    row = {'step': self.step, 'steps': t.steps, 'loss': window_loss/window_steps,
                           'lr': lr, 'grad_norm': norm, 'tokens_per_second': window_tokens/max(1e-9,time.perf_counter()-window_start),
                           'total_tokens': self.total_tokens}
                    if validation:
                        row['val_loss'] = evaluate(self.model, self.val_data, t, self.device, self.dtype)
                        if row['val_loss'] < self.best:
                            self.best = row['val_loss']
                            self.save('best.pt')
                    print(json.dumps(row), flush=True)
                    with metrics_path.open('a', encoding='utf-8') as f:
                        f.write(json.dumps(row)+'\n')
                    window_start, window_tokens, window_loss, window_steps = time.perf_counter(), 0, 0.0, 0
                if self.step % t.save_interval == 0 or self.step == target or stopping:
                    self.save()
                if stopping:
                    reason = 'stop_file' if requested_stop else 'time_limit'
                    break
        except KeyboardInterrupt:
            print('Interrupted. Resume from the last completed checkpoint; partial update discarded.')
            return {'reason': 'interrupted', 'step': self.step}
        return {'reason': reason, 'step': self.step, 'elapsed_seconds': time.perf_counter()-started}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/balanced.json')
    p.add_argument('--data', default='prepared/pretrain')
    p.add_argument('--tokenizer', default='artifacts/tokenizer.json')
    p.add_argument('--output')
    p.add_argument('--resume')
    p.add_argument('--device', default='auto')
    p.add_argument('--stop-after', type=int)
    p.add_argument('--max-seconds', type=float)
    p.add_argument('--stop-file')
    p.add_argument('--overfit-only', action='store_true')
    a = p.parse_args()
    a.output = a.output or (str(Path(a.resume).parent) if a.resume else 'runs/pretrain')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = select_device(a.device)
    tokenizer = BPETokenizer.load(a.tokenizer)
    resume = load_checkpoint(a.resume) if a.resume else None
    if resume:
        c, t = ModelConfig(**resume['model_config']), TrainConfig(**resume['training_config'])
    else:
        c, t = load_config(a.config)
        c.vocab_size = tokenizer.vocab_size
    report = resume['overfit'] if resume else overfit_test(tokenizer, device, precision_for(device, t.precision))
    if a.overfit_only:
        return
    if not report or not report['passed']:
        raise ValueError('Missing successful overfit test')
    if not resume and (Path(a.output)/'last.pt').exists():
        raise FileExistsError('Use --resume or a new --output directory')
    train_data = TokenDataset(a.data, 'train', c.context_length, tokenizer)
    val_data = TokenDataset(a.data, 'val', c.context_length, tokenizer)
    seed_all(t.seed)
    model = Transformer(c).to(device)
    print(f'Parameters={c.parameter_count()} device={device}', flush=True)
    Trainer(model, t, tokenizer, train_data, val_data, a.output, device,
            resume=resume, overfit=report).run(a.stop_after, a.max_seconds, a.stop_file)


if __name__ == '__main__':
    main()
