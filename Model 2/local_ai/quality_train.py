"""Four-hour bounded continuation on a new curriculum; old runs stay untouched."""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

from config import ModelConfig, TrainConfig
from data import file_hash
from dialogue_pilot import run_probes, scores
from learn import Tee, keep_awake, warm_start
from model import Transformer, generate
from quality_data import GROUPS, datasets
from start_chat import latest_metrics
from tokenizer import BPETokenizer, BOS, EOS
from train import Trainer, amp_context, evaluate, load_checkpoint, seed_all

ROOT = Path(__file__).resolve().parent
RUN = ROOT/'runs/quality_v3'
PREPARED = ROOT/'prepared/quality_v3'
SOURCE = ROOT/'runs/real_v2/pretrain/best.pt'
BASELINE = ROOT/'runs/real_v2/sft/best.pt'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else {}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def status(phase, **values):
    value = {'phase': phase, 'pid': os.getpid(), 'updated_at': datetime.now(timezone.utc).isoformat(), **values}
    write_json(RUN/'status.json', value)
    print(json.dumps(value, ensure_ascii=True), flush=True)


def current_status():
    value = read_json(RUN/'status.json')
    if value.get('phase') in ('pretrain', 'sft'):
        metrics = latest_metrics(RUN/value['phase']/'metrics.jsonl')
        value.update({k: metrics[k] for k in ('step', 'loss', 'tokens_per_second', 'total_tokens') if k in metrics})
    return value


def live_lock(path):
    value = read_json(path)
    try:
        process = psutil.Process(value.get('pid', -1))
        return abs(process.create_time()-value.get('process_created', 0)) < 1
    except (psutil.NoSuchProcess, ValueError):
        return False
    except psutil.AccessDenied:
        return psutil.pid_exists(value.get('pid', -1))


def probes():
    data = ROOT/'data/quality_v3'
    result = []
    for line in (data/'context_test.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        result.append({'group': 'context', 'messages': row['messages'][1:-1],
                       'expected': row['messages'][-1]['content']})
        if len(result) == 24:
            break
    for line in (data/'reading_test.jsonl').open(encoding='utf-8'):
        row = json.loads(line)
        # Restrict evaluation input to leave room for generation without dropped evidence.
        if len(row['messages'][-2]['content']) > 1800:
            continue
        result.append({'group': 'reading', 'messages': row['messages'][1:-1],
                       'expected': row['messages'][-1]['content']})
        if len(result) == 36:
            break
    for q in ['Hallo! Stell dich bitte vor.', 'Was ist ein Fluss?',
              'Erkläre einfach, warum Wasser gefriert.',
              'Schreibe drei Sätze über eine Katze, die ihren Ball sucht.',
              'Ich weiß nicht, welches davon ich nehmen soll. Was brauchst du für einen Vergleich?',
              'Mein Lieblingswort ist Regenschirm. Welches Wort habe ich genannt?',
              'Ich heiße Mara. Wie heiße ich?', 'Was ist 28 plus 15?']:
        result.append({'group': 'open_ended', 'question': q, 'expected': None})
    return result


def make_plan(model_config, parent_hash, baseline_hash, started):
    return {'version': 1, 'model': model_config, 'source': str(SOURCE), 'source_sha256': parent_hash,
            'baseline_sha256': baseline_hash, 'started_at': started, 'deadline': started+4*3600,
            'pretrain_deadline': started+2.75*3600, 'maximum_hours': 4,
            'pretrain': asdict(TrainConfig(steps=80000, batch_size=8, grad_accum=2, lr=8e-5,
                         warmup_steps=500, eval_interval=1000, eval_batches=32, save_interval=1000,
                         log_interval=100, seed=2026091803, precision='bf16')),
            'sft': asdict(TrainConfig(steps=18000, batch_size=4, grad_accum=4, lr=2e-5,
                     warmup_steps=300, eval_interval=500, eval_batches=48, save_interval=500,
                     log_interval=100, seed=2026091804, precision='bf16')),
            'language_sampling': {'stories': .70, 'knowledge': .05, 'old_language_corpus': .25},
            'sft_sampling': dict(zip(GROUPS, [.35, .20, .15, .10, .18, .02])),
            'early_stopping': 'Six validation blocks without a loss improvement of at least 0.003.',
            'checkpoint_selection': 'Minimum fixed validation loss; test answers never select weights.',
            'default_chat_changed': False,
            'time_policy': 'Fixed deadline across resume; no automatic extension. Data preparation precedes budget. '
                           'Final checkpoint and inference evaluation can take a few extra minutes.'}


def exhausted(plan, stage, now=None):
    now = time.time() if now is None else now
    deadline = plan['pretrain_deadline'] if stage == 'pretrain' else plan['deadline']
    return now >= deadline


def intermediate_samples(trainer, tokenizer, stage):
    if stage == 'pretrain':
        rows = []
        for prefix in ['Ein Mädchen ging in den Garten. Dort',
                       'Ein Hund suchte seinen Ball. Er', 'Eis wird bei Wärme zu']:
            with amp_context(trainer.device, trainer.dtype):
                ids = list(generate(trainer.model, [BOS]+tokenizer.encode(prefix), 64,
                                    temperature=0, stop_ids=(EOS,)))
            rows.append({'prefix': prefix, 'continuation': tokenizer.decode(ids)})
    else:
        rows = run_probes(trainer.model, tokenizer, probes()[-8:], trainer.device, trainer.dtype, 'intermediate')
    write_json(RUN/stage/f'samples_{trainer.step}.json', {'step': trainer.step, 'answers': rows,
               'note': 'Monitoring samples from the current weights, not used for checkpoint selection.'})


def run_stage(stage, plan, tokenizer, device, state):
    config = TrainConfig(**plan[stage])
    resume_path = RUN/stage/'last.pt'
    if resume_path.exists():
        payload = load_checkpoint(resume_path)
        resuming = True
    else:
        payload = load_checkpoint(SOURCE if stage == 'pretrain' else RUN/'pretrain/best.pt')
        resuming = False
    model = Transformer(ModelConfig(**plan['model'])).to(device)
    warm_start(model, payload, tokenizer)
    train_data, val_data = datasets(PREPARED, tokenizer, stage)
    trainer = Trainer(model, config, tokenizer, train_data, val_data, RUN/stage, device,
                      stage=stage, resume=payload if resuming else None, overfit=payload.get('overfit'))
    del payload
    gc.collect()
    if not resuming:
        trainer.best = evaluate(model, val_data, config, device, trainer.dtype)
        trainer.save('best.pt')
        trainer.save()
        state[stage] = {'best_loss': trainer.best, 'baseline_loss': trainer.best, 'bad_blocks': 0}
        write_json(RUN/'progress.json', state)
    phase = state.setdefault(stage, {'best_loss': trainer.best, 'baseline_loss': trainer.best, 'bad_blocks': 0})
    deadline = plan['pretrain_deadline'] if stage == 'pretrain' else plan['deadline']
    reason = 'steps'
    while trainer.step < config.steps and time.time() < deadline-30:
        status(stage, step=trainer.step, planned_steps=config.steps, best_val=trainer.best,
               deadline=deadline, total_deadline=plan['deadline'])
        target = min(config.steps, ((trainer.step//config.eval_interval)+1)*config.eval_interval)
        result = trainer.run(stop_after=target, max_seconds=max(1, deadline-time.time()-30), stop_file=RUN/'STOP')
        if phase['best_loss']-trainer.best >= .003:
            phase['best_loss'], phase['bad_blocks'] = trainer.best, 0
        else:
            phase['bad_blocks'] += 1
        phase.update({'step': trainer.step, 'processed_input_positions': trainer.total_tokens,
                      'best_validation_loss': trainer.best, 'last_reason': result['reason']})
        write_json(RUN/'progress.json', state)
        if result['reason'] in ('stop_file', 'interrupted'):
            status('paused', stage=stage, step=trainer.step, deadline=plan['deadline'])
            return False
        if trainer.step % (5000 if stage == 'pretrain' else 2000) == 0 and time.time() < deadline-90:
            intermediate_samples(trainer, tokenizer, stage)
        if phase['bad_blocks'] >= 6:
            reason = 'validation_stalled'
            break
        if result['reason'] == 'time_limit':
            reason = 'time_limit'
            break
    if time.time() >= deadline-30:
        reason = 'time_limit'
    phase.update({'complete': True, 'reason': reason})
    write_json(RUN/'progress.json', state)
    del trainer, model, train_data, val_data
    gc.collect()
    torch.cuda.empty_cache()
    return True


def evaluate_checkpoint(path, tokenizer, device, items, label):
    payload = load_checkpoint(path)
    model = Transformer(ModelConfig(**payload['model_config'])).to(device)
    warm_start(model, payload, tokenizer)
    metadata = {'step': payload['step'], 'val_loss': payload['best_val']}
    del payload
    gc.collect()
    results = run_probes(model, tokenizer, items, device, torch.bfloat16, label)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {'checkpoint': str(path), **metadata, 'scores': scores(results), 'answers': results}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--resume', action='store_true')
    p.add_argument('--status', action='store_true')
    p.add_argument('--chat', action='store_true')
    args = p.parse_args()
    if args.status or args.chat:
        current = current_status()
        print(json.dumps(current, ensure_ascii=True, indent=2), flush=True)
        if args.chat:
            while current.get('phase') not in ('finished_unreviewed', 'failed', 'paused', 'budget_exhausted'):
                if not live_lock(RUN/'running.lock'):
                    raise RuntimeError('Kein aktiver neuer Trainingslauf. Training_4_Stunden.cmd starten.')
                time.sleep(15)
                updated = current_status()
                if updated != current:
                    print(json.dumps(updated, ensure_ascii=True), flush=True)
                current = updated
            if current.get('phase') != 'finished_unreviewed':
                raise RuntimeError('Der neue Dialoglauf ist noch nicht abgeschlossen. Siehe Status.')
            print('Experimentelles Modell. Automatische Testergebnisse sind kein Nachweis allgemeiner Chatqualitaet.')
            return subprocess.call([sys.executable, str(ROOT/'chat.py'), '--checkpoint', str(RUN/'sft/best.pt'),
                       '--memory', str(ROOT/'artifacts/quality_v3_memory.sqlite'), '--temperature', '0.4',
                       '--no-auto-memory'], cwd=ROOT)
        return 0
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This bounded plan requires the verified CUDA/BF16 GPU')
    for run in (ROOT/'runs').iterdir():
        lock = run/'running.lock'
        old = read_json(lock)
        if lock.exists() and (live_lock(lock) or (old.get('pid') and psutil.pid_exists(old['pid']))):
            raise RuntimeError(f'Active training lock: {lock}')
    if not (PREPARED/'basics_validation/meta.json').exists():
        raise RuntimeError('Curriculum preparation is not complete')
    RUN.mkdir(parents=True, exist_ok=True)
    plan = read_json(RUN/'plan.json')
    if plan and not args.resume:
        raise RuntimeError('Existing run: use --resume; never overwrite it')
    if read_json(RUN/'status.json').get('phase') == 'finished_unreviewed':
        print('Der Vier-Stunden-Lauf ist bereits beendet. Chat_Neues_Modell.cmd verwenden.')
        return 0
    if plan and exhausted(plan, 'sft'):
        print('Das feste Vier-Stunden-Zeitfenster ist beendet. Kein weiterer Lauf wird gestartet.')
        return 0
    (RUN/'STOP').unlink(missing_ok=True)
    lock = RUN/'running.lock'
    lock.unlink(missing_ok=True)
    with lock.open('x', encoding='utf-8') as stream:
        json.dump({'pid': os.getpid(), 'process_created': psutil.Process().create_time()}, stream)
    try:
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        seed_all(2026091803)
        tokenizer = BPETokenizer.load(ROOT/'artifacts/real_v1/tokenizer.json')
        device = torch.device('cuda')
        if not plan:
            parent = load_checkpoint(SOURCE)
            if not (parent.get('overfit') or {}).get('passed'):
                raise RuntimeError('Missing verified source overfit test')
            model_config = parent['model_config']
            del parent
            plan = make_plan(model_config, file_hash(SOURCE), file_hash(BASELINE), time.time())
            plan['data_manifest_sha256'] = file_hash(ROOT/'data/quality_v3/manifest.json')
            write_json(RUN/'plan.json', plan)
        if plan['data_manifest_sha256'] != file_hash(ROOT/'data/quality_v3/manifest.json'):
            raise ValueError('Data manifest changed since the run started')
        items = probes()
        state = read_json(RUN/'progress.json')
        with keep_awake():
            if not (RUN/'before.json').exists():
                status('evaluating_baseline', deadline=plan['deadline'])
                write_json(RUN/'before.json', evaluate_checkpoint(BASELINE, tokenizer, device, items, 'before'))
            for stage in ('pretrain', 'sft'):
                if not state.get(stage, {}).get('complete'):
                    if exhausted(plan, 'sft'):
                        status('budget_exhausted', stage=stage)
                        return 0
                    if not run_stage(stage, plan, tokenizer, device, state):
                        return 0
            status('evaluating', deadline=plan['deadline'])
            after = evaluate_checkpoint(RUN/'sft/best.pt', tokenizer, device, items, 'after')
            before = read_json(RUN/'before.json')
            unchanged = file_hash(SOURCE) == plan['source_sha256'] and file_hash(BASELINE) == plan['baseline_sha256']
            if not unchanged:
                raise RuntimeError('Original checkpoint changed during training; inspect other writers')
            write_json(RUN/'quality_results.json', {'before': before, 'after': after,
                       'source_unchanged': unchanged, 'default_chat_changed': False,
                       'note': 'Strict checks of 24 copying/context tasks and 12 reading tasks; '
                               'open answers require review. No automatic promotion or quality guarantee.'})
            status('finished_unreviewed', checkpoint=str(RUN/'sft/best.pt'),
                   before_scores=before['scores'], after_scores=after['scores'],
                   elapsed_seconds=time.time()-plan['started_at'])
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
