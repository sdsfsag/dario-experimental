"""Bounded SFT experiment; never changes the default chat or the source checkpoint."""
import argparse
import gc
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import ModelConfig, TrainConfig
from context import Conversation
from data import encode_conversation
from hardware import precision_for, select_device
from model import Transformer, generate
from pilot_data import write_data
from tokenizer import ASSISTANT, BOS, END, EOS, PAD, SYSTEM, USER, BPETokenizer
from train import Trainer, amp_context, evaluate, load_checkpoint, seed_all

ROOT = Path(__file__).resolve().parent
GROUP_WEIGHTS = {'dialog': .45, 'copy': .25, 'context': .15, 'arithmetic': .15}


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class CompactDataset:
    """Canonical assistant-only shifted labels, variable right padding, balanced tasks."""
    def __init__(self, rows, tokenizer, max_length=256, date=None, group_weights=None, homogeneous=False):
        self.rows = []
        group_weights = GROUP_WEIGHTS if group_weights is None else group_weights
        groups = Counter(row['group'] for row in rows)
        if not groups or set(groups) - group_weights.keys() or any(v <= 0 for v in group_weights.values()):
            raise ValueError('Empty dataset or unknown task group')
        self.weights = torch.tensor([group_weights[row['group']]/groups[row['group']]
                                     for row in rows], dtype=torch.double)
        self.homogeneous = homogeneous
        self.group_names = sorted(groups)
        self.group_probabilities = torch.tensor([group_weights[g] for g in self.group_names], dtype=torch.double)
        self.group_indices = {g: torch.tensor([i for i, r in enumerate(rows) if r['group'] == g]) for g in groups}
        self.max_length = max_length
        for row in rows:
            messages = [dict(m) for m in row['messages']]
            if date:
                if messages[0]['role'] != 'system':
                    raise ValueError('System message required for date augmentation')
                messages[0]['content'] += f' Datum: {date}.'
            tokens, labels = encode_conversation(messages, tokenizer)
            if len(tokens)-1 > max_length:
                raise ValueError(f'Example length {len(tokens)-1} exceeds {max_length}; no truncation allowed')
            if all(value == -100 for value in labels[1:]):
                raise ValueError('Example has no answer targets')
            self.rows.append((torch.tensor(tokens[:-1]), torch.tensor(labels[1:])))
        payload = {'rows': rows, 'tokenizer': tokenizer.fingerprint, 'date': date,
                   'max_length': max_length, 'group_weights': group_weights, 'homogeneous': homogeneous}
        self.fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                                    ensure_ascii=False).encode()).hexdigest()
        self.stats = {'examples': len(rows), 'groups': dict(groups),
                      'max_length': max(len(x) for x, _ in self.rows),
                      'input_tokens': sum(len(x) for x, _ in self.rows),
                      'answer_targets': sum(int((y != -100).sum()) for _, y in self.rows)}
        self.sampled_input_tokens = self.sampled_answer_targets = 0

    def batch(self, batch_size, generator):
        if self.homogeneous:
            group = self.group_names[int(torch.multinomial(self.group_probabilities, 1, generator=generator))]
            pool = self.group_indices[group]
            indices = pool[torch.randint(len(pool), (batch_size,), generator=generator)]
        else:
            indices = torch.multinomial(self.weights, batch_size, replacement=True, generator=generator)
        rows = [self.rows[int(i)] for i in indices]
        width = min(self.max_length, ((max(len(x) for x, _ in rows)+7)//8)*8)
        x = torch.full((batch_size, width), PAD, dtype=torch.long)
        y = torch.full((batch_size, width), -100, dtype=torch.long)
        for i, (source, labels) in enumerate(rows):
            x[i, :len(source)] = source
            y[i, :len(labels)] = labels
        self.sampled_input_tokens += sum(len(source) for source, _ in rows)
        self.sampled_answer_targets += int((y != -100).sum())
        return x, y


def strict_match(answer, expected):
    if expected is None:
        return None
    normalize = lambda s: ' '.join(s.strip().casefold().split()).rstrip('.!?')
    return normalize(answer) == normalize(expected)


def run_probes(model, tokenizer, probes, device, dtype, label, repetition_penalty=1.1):
    results = []
    model.eval()
    for i, row in enumerate(probes):
        conversation = Conversation(tokenizer, model.config.context_length)
        messages = row.get('messages') or [{'role': 'user', 'content': row['question']}]
        for message in messages:
            conversation.add(message['role'], message['content'])
        limit = 80 if row['expected'] is None else 32
        prompt = conversation.build([], limit)
        with amp_context(device, dtype):
            ids = list(generate(model, prompt, limit, temperature=0, repetition_penalty=repetition_penalty,
                                stop_ids=(EOS, END), forbidden_ids=(PAD, BOS, SYSTEM, USER, ASSISTANT)))
        answer = tokenizer.decode(ids)
        result = {**row, 'answer': answer, 'correct': strict_match(answer, row['expected']),
                  'hit_length_limit': len(ids) == limit}
        results.append(result)
        print(json.dumps({'phase': label, 'probe': i+1, 'total': len(probes),
                          'group': row['group'], 'answer': answer, 'correct': result['correct']},
                         ensure_ascii=True), flush=True)
    return results


def scores(results):
    groups = {}
    for row in results:
        if row['correct'] is None:
            continue
        score = groups.setdefault(row['group'], {'correct': 0, 'total': 0})
        score['total'] += 1
        score['correct'] += int(row['correct'])
    return groups


def write_json(path, content):
    Path(path).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'runs/real_v2/sft/best.pt')
    p.add_argument('--output', type=Path, default=ROOT/'runs/dialogue_pilot_v1')
    p.add_argument('--seconds', type=float, default=300)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    if not 0 < args.seconds <= 300 or not 1 <= args.steps <= 1200:
        p.error('Pilot limits: at most 300 seconds and 1200 updates')
    if args.output.exists():
        p.error('Output directory exists; choose a fresh versioned directory. No automatic repeat/resume.')
    if not args.source.is_file():
        p.error('Source checkpoint missing')
    # A fresh sibling experiment may not contain or overwrite its source.
    if args.source.resolve().is_relative_to(args.output.resolve()):
        p.error('Source must be outside the pilot output directory')
    for run in ('real_v1', 'real_v2'):
        if (ROOT/'runs'/run/'running.lock').exists():
            p.error('Another training lock exists; finish or inspect that run first')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    seed_all(20260918)
    device = select_device(args.device)
    dtype = precision_for(device)
    source_hash = file_hash(args.source)
    state = load_checkpoint(args.source)
    tokenizer = BPETokenizer(Tokenizer.from_str(state['tokenizer_json']))
    if tokenizer.fingerprint != state['tokenizer_hash']:
        raise ValueError('Source tokenizer mismatch')
    model = Transformer(ModelConfig(**state['model_config'])).to(device)
    model.load_state_dict(state['model'])
    source_step = state['step']
    del state
    gc.collect()
    args.output.mkdir(parents=True, exist_ok=False)
    train, val, test = write_data(args.output/'data')
    date = datetime.now().astimezone().date().isoformat()
    train_data = CompactDataset(train, tokenizer, date=date)
    val_data = CompactDataset(val, tokenizer, date=date)
    config = TrainConfig(steps=args.steps, batch_size=8, grad_accum=2, lr=1e-4,
                         min_lr_ratio=.15, warmup_steps=50, weight_decay=.01,
                         eval_interval=200, eval_batches=16, save_interval=200,
                         log_interval=50, seed=20260918, precision='auto')
    manifest = {'source': str(args.source), 'source_sha256': source_hash, 'source_step': source_step,
                'model': asdict(model.config), 'training': asdict(config),
                'limit_seconds': args.seconds, 'date_in_system_prompt': date,
                'train': train_data.stats, 'validation': val_data.stats,
                'task_sampling_weights': GROUP_WEIGHTS,
                'probe_count': len(test), 'default_chat_changed': False,
                'notes': 'Narrow synthetic curriculum; tests never used for training or checkpoint selection.'}
    write_json(args.output/'manifest.json', manifest)
    print(json.dumps({'phase': 'prepared', 'train': train_data.stats, 'validation': val_data.stats}), flush=True)
    before = run_probes(model, tokenizer, test, device, dtype, 'before')
    write_json(args.output/'before.json', before)
    trainer = Trainer(model, config, tokenizer, train_data, val_data, args.output/'sft', device, stage='sft')
    trainer.best = evaluate(model, val_data, config, device, dtype)
    baseline_loss = trainer.best
    trainer.save('best.pt')  # Keep the baseline if every training checkpoint is worse.
    from learn import keep_awake
    with keep_awake():
        outcome = trainer.run(max_seconds=args.seconds, stop_file=args.output/'STOP')
    del trainer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    best = load_checkpoint(args.output/'sft/best.pt')
    model.load_state_dict(best['model'])
    selected_step, selected_loss = best['step'], best['best_val']
    del best
    gc.collect()
    after = run_probes(model, tokenizer, test, device, dtype, 'after')
    unchanged = file_hash(args.source) == source_hash
    report = {'training': outcome, 'selected_step': selected_step,
              'baseline_validation_loss': baseline_loss, 'selected_validation_loss': selected_loss,
              'source_unchanged': unchanged,
              'sampled_input_tokens_without_padding': train_data.sampled_input_tokens,
              'sampled_assistant_targets': train_data.sampled_answer_targets,
              'before_scores': scores(before), 'after_scores': scores(after),
              'before': before, 'after': after,
              'evaluation': 'Fresh context per probe; no persistent memory; greedy; repetition penalty 1.1. '
                            '24 narrow held-out tests; 11 open answers require manual review. '
                            'Validation loss alone selected the checkpoint. No general capability claim.'}
    write_json(args.output/'results.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('before', 'after')}, ensure_ascii=True), flush=True)
    if not unchanged:
        raise RuntimeError('Source changed during the trial; inspect concurrent writers')


if __name__ == '__main__':
    main()
