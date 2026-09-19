"""Four-hour dialogue continuation with a generation gate and resumable checkpoints."""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

from config import ModelConfig, TrainConfig
from data import file_hash
from dialogue_data import DATA, WEIGHTS, build, datasets, probes
from learn import Tee, keep_awake, warm_start
from model import Transformer
from quality_train import evaluate_checkpoint, live_lock, read_json, write_json
from start_chat import latest_metrics
from tokenizer import BPETokenizer
from train import Trainer, evaluate, load_checkpoint, seed_all

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'runs/dialogue_v4'
SOURCE = ROOT/'runs/quality_v3/sft/best.pt'
TERMINAL = ('finished_unreviewed', 'trial_rejected', 'failed', 'paused', 'budget_exhausted')


def status(phase, **values):
    value = {'phase': phase, 'pid': os.getpid(), 'updated_at': datetime.now(timezone.utc).isoformat(), **values}
    write_json(RUN/'status.json', value)
    print(json.dumps(value, ensure_ascii=True), flush=True)


def current_status():
    value = read_json(RUN/'status.json')
    if value.get('phase') == 'sft':
        value.update(latest_metrics(RUN/'sft/metrics.jsonl'))
    return value


def make_plan(model_config, source_hash, data_hash, started):
    return {'version': 1, 'model': model_config, 'source': str(SOURCE), 'source_sha256': source_hash,
            'data_manifest_sha256': data_hash, 'started_at': started, 'deadline': started+4*3600,
            'trial_seconds': 20*60, 'maximum_hours': 4, 'loss_reduction': 'example',
            'sft_sampling': WEIGHTS,
            'training': asdict(TrainConfig(steps=120000, batch_size=4, grad_accum=4, lr=7e-6,
                              min_lr_ratio=.15, warmup_steps=200, weight_decay=.01,
                              eval_interval=500, eval_batches=64, save_interval=500,
                              log_interval=100, seed=2026091904, precision='bf16')),
            'checkpoint_selection': 'Minimum fixed validation loss; separate dev answers gate continuation.',
            'gate': 'At least four additional exact dev answers, no decrease in context or reading, '
                    'validation loss improves, no more than two additional open answers hit the length limit.',
            'early_stopping': 'After the trial: twelve validation blocks without an improvement of 0.002.',
            'time_policy': 'Fixed wall-clock deadline across resumes. Data preparation before budget; '
                           'checkpoint saving and final tests may take a few extra minutes.',
            'source_lineage_tokens': 2172156416, 'default_chat_changed': False}


def continuation_gate(before, after, baseline_loss, best_loss):
    if before['scores'].keys() != after['scores'].keys() or any(
            before['scores'][g]['total'] != after['scores'][g]['total'] for g in before['scores']):
        raise ValueError('Gate requires exactly the same probes')
    gain = sum(v['correct'] for v in after['scores'].values())-sum(v['correct'] for v in before['scores'].values())
    preserved = all(after['scores'][g]['correct'] >= before['scores'][g]['correct'] for g in ('context', 'reading'))
    long_answers = lambda result: sum(r['hit_length_limit'] for r in result['answers'] if r['group'] == 'open_ended')
    reasons = []
    if gain < 4:
        reasons.append('fewer_than_four_additional_exact_answers')
    if not preserved:
        reasons.append('context_or_reading_regression')
    if best_loss >= baseline_loss-.002:
        reasons.append('validation_not_improved')
    if long_answers(after) > long_answers(before)+2:
        reasons.append('more_unfinished_open_answers')
    return {'passed': not reasons, 'gain': gain, 'reasons': reasons,
            'before_scores': before['scores'], 'after_scores': after['scores'],
            'baseline_validation': baseline_loss, 'best_validation': best_loss,
            'note': 'Small development set; passing is not proof of general conversation quality.'}


def run_training(plan, tokenizer, device):
    state = read_json(RUN/'progress.json')
    dev = probes('dev')
    if not (RUN/'before_dev.json').exists():
        status('evaluating_baseline', deadline=plan['deadline'])
        write_json(RUN/'before_dev.json', evaluate_checkpoint(SOURCE, tokenizer, device, dev, 'before_dev'))
    train_data, val_data = datasets(tokenizer)
    last = RUN/'sft/last.pt'
    payload = load_checkpoint(last if last.exists() else SOURCE)
    config = TrainConfig(**plan['training'])
    model = Transformer(ModelConfig(**plan['model'])).to(device)
    warm_start(model, payload, tokenizer)
    trainer = Trainer(model, config, tokenizer, train_data, val_data, RUN/'sft', device,
                      stage='sft', resume=payload if last.exists() else None, overfit=payload.get('overfit'))
    del payload
    gc.collect()
    if 'baseline_loss' not in state:
        state['baseline_loss'] = evaluate(model, val_data, config, device, trainer.dtype)
        trainer.best = state['baseline_loss']
        trainer.save('best.pt')
        trainer.save()
        state.update(best_loss=trainer.best, bad_blocks=0, trial_deadline=time.time()+plan['trial_seconds'])
        write_json(RUN/'progress.json', state)
    reason = 'steps'
    while trainer.step < config.steps and time.time() < plan['deadline']-45:
        if (RUN/'STOP').exists():
            trainer.save()
            status('paused', step=trainer.step, deadline=plan['deadline'])
            return
        if not state.get('gate') and time.time() >= state['trial_deadline']:
            status('evaluating_trial', step=trainer.step, deadline=plan['deadline'])
            candidate = evaluate_checkpoint(RUN/'sft/best.pt', tokenizer, device, dev, 'trial_dev')
            write_json(RUN/'trial_dev.json', candidate)
            decision = continuation_gate(read_json(RUN/'before_dev.json'), candidate,
                                         state['baseline_loss'], trainer.best)
            state['gate'] = decision
            write_json(RUN/'progress.json', state)
            write_json(RUN/'trial_decision.json', decision)
            print(json.dumps({'phase': 'trial_decision', **decision}), flush=True)
            if not decision['passed']:
                trainer.save()
                status('trial_rejected', step=trainer.step, reason=decision['reasons'],
                       checkpoint=str(SOURCE), candidate=str(RUN/'sft/best.pt'),
                       elapsed_seconds=time.time()-plan['started_at'])
                return
        status('sft', step=trainer.step, planned_steps=config.steps, best_val=trainer.best,
               deadline=plan['deadline'], trial_deadline=state['trial_deadline'],
               trial_passed=bool(state.get('gate', {}).get('passed')))
        deadline = plan['deadline']-45 if state.get('gate') else min(state['trial_deadline'], plan['deadline']-45)
        if deadline <= time.time():
            continue
        target = min(config.steps, ((trainer.step//config.eval_interval)+1)*config.eval_interval)
        result = trainer.run(stop_after=target, max_seconds=max(1, deadline-time.time()), stop_file=RUN/'STOP')
        if state['best_loss']-trainer.best >= .002:
            state['best_loss'], state['bad_blocks'] = trainer.best, 0
        else:
            state['bad_blocks'] += 1
        state.update(step=trainer.step, processed_input_positions=trainer.total_tokens,
                     best_validation=trainer.best, last_reason=result['reason'])
        write_json(RUN/'progress.json', state)
        if result['reason'] in ('stop_file', 'interrupted'):
            status('paused', step=trainer.step, deadline=plan['deadline'])
            return
        if state.get('gate', {}).get('passed') and state['bad_blocks'] >= 12:
            reason = 'validation_stalled'
            break
    if time.time() >= plan['deadline']-45:
        reason = 'time_limit'
    trainer.save()
    if not state.get('gate', {}).get('passed'):
        status('budget_exhausted', step=trainer.step, checkpoint=str(SOURCE), reason='trial_not_approved')
        return
    total, final_step = trainer.total_tokens, trainer.step
    del trainer, model, train_data, val_data
    gc.collect()
    torch.cuda.empty_cache()
    status('evaluating_final', reason=reason)
    test = probes('test')
    before = evaluate_checkpoint(SOURCE, tokenizer, device, test, 'before_test')
    after = evaluate_checkpoint(RUN/'sft/best.pt', tokenizer, device, test, 'after_test')
    if file_hash(SOURCE) != plan['source_sha256']:
        raise RuntimeError('Original checkpoint changed during training')
    selected = load_checkpoint(RUN/'sft/best.pt')
    lineage = plan['source_lineage_tokens']+selected['total_tokens']
    selected_step = selected['step']
    del selected
    write_json(RUN/'quality_results.json', {'before': before, 'after': after, 'dev_gate': state['gate'],
               'source_unchanged': True, 'reason': reason, 'selected_step': selected_step,
               'processed_input_positions': total, 'selected_lineage_input_positions': lineage,
               'note': 'Final held-out test is reported, never used to choose a checkpoint. '
                       'Token counts include dynamic padding/repetition, not unique new information.'})
    status('finished_unreviewed', checkpoint=str(RUN/'sft/best.pt'), step=final_step,
           selected_step=selected_step, reason=reason, before_scores=before['scores'], after_scores=after['scores'],
           processed_input_positions=total, selected_lineage_input_positions=lineage,
           elapsed_seconds=time.time()-plan['started_at'])


def chat():
    current = current_status()
    while current.get('phase') not in TERMINAL:
        if not live_lock(RUN/'running.lock'):
            raise RuntimeError('Kein aktiver Dialoglauf. Training_4_Stunden.cmd starten.')
        print(json.dumps(current, ensure_ascii=True), flush=True)
        time.sleep(15)
        current = current_status()
    if current['phase'] == 'finished_unreviewed':
        checkpoint, memory = RUN/'sft/best.pt', ROOT/'artifacts/dialogue_v4_memory.sqlite'
        print('Neuer Dialoglauf. Sprachqualitaet noch experimentell.')
    elif current['phase'] in ('trial_rejected', 'budget_exhausted'):
        checkpoint, memory = SOURCE, ROOT/'artifacts/quality_v3_memory.sqlite'
        print('Kein ausreichend besserer Dialoglauf freigegeben. Der Chat verwendet die bisherige Version.')
    else:
        raise RuntimeError('Training pausiert oder fehlgeschlagen. Siehe runs/dialogue_v4/status.json.')
    return subprocess.call([sys.executable, str(ROOT/'chat.py'), '--checkpoint', str(checkpoint),
                            '--memory', str(memory), '--temperature', '0.4', '--no-auto-memory'], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--chat', action='store_true')
    args = parser.parse_args()
    if args.chat:
        return chat()
    if args.status:
        print(json.dumps(current_status(), indent=2))
        return 0
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA/BF16 required for this bounded training plan')
    for path in (ROOT/'runs').glob('*/running.lock'):
        if live_lock(path):
            raise RuntimeError(f'Another training process is active: {path}')
    plan = read_json(RUN/'plan.json')
    if plan and not args.resume:
        raise RuntimeError('Existing run; use --resume')
    if current_status().get('phase') in ('finished_unreviewed', 'trial_rejected', 'budget_exhausted'):
        print('Dieser Lauf ist abgeschlossen. Es wird kein zweiter Lauf gestartet.')
        return 0
    if plan and time.time() >= plan['deadline']:
        status('budget_exhausted', checkpoint=str(SOURCE), reason='fixed_deadline_expired')
        return 0
    lock = RUN/'running.lock'
    lock.unlink(missing_ok=True)
    with lock.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'process_created': psutil.Process().create_time()}, stream)
    (RUN/'STOP').unlink(missing_ok=True)
    try:
        with keep_awake():
            status('preparing_data')
            build()
            torch.set_num_threads(min(8, os.cpu_count() or 1))
            seed_all(2026091904)
            tokenizer = BPETokenizer.load(ROOT/'artifacts/real_v1/tokenizer.json')
            if not plan:
                payload = load_checkpoint(SOURCE)
                if not (payload.get('overfit') or {}).get('passed'):
                    raise RuntimeError('Source lacks a successful overfit test')
                plan = make_plan(payload['model_config'], file_hash(SOURCE), file_hash(DATA/'manifest.json'), time.time())
                del payload
                write_json(RUN/'plan.json', plan)
            if file_hash(SOURCE) != plan['source_sha256'] or file_hash(DATA/'manifest.json') != plan['data_manifest_sha256']:
                raise ValueError('Source or curriculum changed since training started')
            run_training(plan, tokenizer, torch.device('cuda'))
    except BaseException as error:
        status('failed', error=str(error))
        traceback.print_exc()
        raise
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    if '--status' in sys.argv or '--chat' in sys.argv:
        raise SystemExit(main())
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'training.log').open('a', encoding='utf-8', buffering=1) as out, \
            (RUN/'training-error.log').open('a', encoding='utf-8', buffering=1) as err:
        with redirect_stdout(Tee(sys.stdout, out)), redirect_stderr(Tee(sys.stderr, err)):
            raise SystemExit(main())
