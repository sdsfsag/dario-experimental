import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

from config import ModelConfig, TrainConfig, load_config, save_config
from context import Conversation
from data import SFTDataset, TokenDataset, file_hash
from hardware import detect_hardware, precision_for, select_device
from model import Transformer, generate
from tokenizer import BPETokenizer, BOS, PAD, EOS, END, SYSTEM, USER, ASSISTANT
from train import Trainer, amp_context, evaluate, load_checkpoint, make_optimizer, overfit_test, seed_all, train_update

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data/real_v1'
PREPARED = ROOT/'prepared/real_v1'
TOKENIZER = ROOT/'artifacts/real_v1/tokenizer.json'
RUN = ROOT/'runs/real_v1'


def continuation_plan(parent, parent_hash, report, hours, target_tokens, sft_rows):
    """A new optimizer schedule on existing weights, with explicit token accounting."""
    if parent['stage'] != 'pretrain' or not (parent.get('overfit') or {}).get('passed'):
        raise ValueError('Continuation requires a verified own pretraining checkpoint')
    if parent['tokenizer_hash'] != report['tokenizer_hash']:
        raise ValueError('Parent and benchmark tokenizer differ')
    c = ModelConfig(**parent['model_config'])
    batch = report['selected']['batch_size']
    accum = max(1, 16384//(batch*c.context_length))
    per_step = batch*accum*c.context_length
    sft_steps = max(100, min(3000, math.ceil(2*sft_rows/16)))
    sft_batch = min(4, batch)
    sft_accum = max(1, 16//sft_batch)
    sft_tokens = sft_steps*sft_batch*sft_accum*c.context_length
    base_tokens = parent['total_tokens']
    if target_tokens <= base_tokens+sft_tokens:
        raise ValueError('Token target must leave room for additional language training and SFT')
    steps = math.ceil((target_tokens-base_tokens-sft_tokens)/per_step)
    t = TrainConfig(steps=steps, batch_size=batch, grad_accum=accum, lr=1e-4,
        warmup_steps=min(500, max(1, steps//20)), eval_interval=500, eval_batches=24,
        save_interval=500, log_interval=50, precision=parent['training_config']['precision'], seed=1337)
    return {'model': asdict(c), 'pretrain': asdict(t), 'initial_hours': hours,
            'benchmark': report['selected'], 'tokenizer_hash': parent['tokenizer_hash'],
            'parent': {'checkpoint': 'runs/real_v1/pretrain/best.pt', 'sha256': parent_hash,
                       'step': parent['step'], 'total_tokens': base_tokens},
            'target_tokens': target_tokens, 'sft_steps': sft_steps,
            'sft_batch_size': sft_batch, 'sft_grad_accum': sft_accum,
            'planned_processed_tokens': base_tokens+steps*per_step+sft_tokens,
            'token_accounting': 'Parent pretraining plus new pretraining and new SFT input tokens; '
                                'SFT includes padding. The previous SFT branch is excluded.',
            'optimizer_policy': 'New AdamW state and warmup for this cycle; exact resume within the cycle.'}


def warm_start(model, payload, tokenizer):
    if payload['model_config'] != asdict(model.config) or payload['tokenizer_hash'] != tokenizer.fingerprint:
        raise ValueError('Warm-start architecture/tokenizer mismatch')
    model.load_state_dict(payload['model'])


def check_continuation(hours, target_tokens):
    """Read and validate local inputs without starting training or writing run state."""
    source = ROOT/'runs/real_v1/pretrain/best.pt'
    parent = load_checkpoint(source)
    report = json.loads((ROOT/'runs/real_v1/benchmark.json').read_text(encoding='utf-8'))
    tokenizer = BPETokenizer.load(TOKENIZER)
    c = ModelConfig(**parent['model_config'])
    for split in ('train', 'val'):
        TokenDataset(PREPARED/'pretrain', split, c.context_length, tokenizer)
    train_sft = SFTDataset(PREPARED/'sft_train', c.context_length, tokenizer)
    val_sft = SFTDataset(PREPARED/'sft_validation', c.context_length, tokenizer)
    if set(train_sft.meta['conversation_hashes']) & set(val_sft.meta['conversation_hashes']):
        raise ValueError('SFT train/validation overlap')
    plan = continuation_plan(parent, file_hash(source), report, hours, target_tokens, len(train_sft.x))
    # Strictly load all real weights on CPU; no forward/backward or GPU work.
    model = Transformer(c)
    warm_start(model, parent, tokenizer)
    return plan


@contextmanager
def keep_awake():
    reset = None
    if os.name == 'nt':
        import ctypes
        call = ctypes.windll.kernel32.SetThreadExecutionState
        call.argtypes, call.restype = [ctypes.c_uint], ctypes.c_uint
        if call(0x80000001):  # Keep the system awake while allowing the display to sleep.
            reset = call
        else:
            print('Automatisches Wachhalten nicht verfuegbar; Energiesparmodus manuell pruefen.', flush=True)
    try:
        yield
    finally:
        if reset is not None:
            reset(0x80000000)


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)

    def flush(self):
        self.console.flush()
        self.log.flush()


@contextmanager
def capture_run_logs(enabled):
    if not enabled:
        yield
        return
    with (RUN/'training.log').open('a', encoding='utf-8', buffering=1) as out, \
         (RUN/'training-error.log').open('a', encoding='utf-8', buffering=1) as err:
        with redirect_stdout(Tee(sys.stdout, out)), redirect_stderr(Tee(sys.stderr, err)):
            yield


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temporary, path)


def status(phase, **values):
    result = {'phase': phase, 'pid': os.getpid(), 'updated_at': datetime.now(timezone.utc).isoformat(), **values}
    write_json(RUN/'status.json', result)
    print(json.dumps(result, ensure_ascii=True), flush=True)


def command(script, *args):
    if (RUN/'STOP').exists():
        raise InterruptedError('Stop requested')
    subprocess.run([sys.executable, str(ROOT/script), *map(str,args)], cwd=ROOT, check=True,
                   env={**os.environ, 'PYTHONUTF8':'1'})
    if (RUN/'STOP').exists():
        raise InterruptedError('Stopped after data preparation stage')


def prepare():
    status('preparing_data')
    if not (DATA/'manifest.json').exists():
        command('download_data.py')
    manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    if 'FreedomIntelligence/alpaca-gpt4-deutsch' not in manifest:
        command('download_data.py', '--add-instructions')
    if not TOKENIZER.exists():
        command('tokenizer.py', '--input', DATA/'tokenizer_sample.jsonl', DATA/'sft_train.jsonl',
                '--output', TOKENIZER, '--vocab-size', 32768)
    if not (PREPARED/'pretrain/meta.json').exists():
        command('data.py', '--input', DATA/'train', '--validation', DATA/'validation',
                '--output', PREPARED/'pretrain', '--tokenizer', TOKENIZER)
    for split in ('train','validation'):
        if not (PREPARED/f'sft_{split}/meta.json').exists():
            command('data.py', '--sft', '--input', DATA/f'sft_{split}.jsonl', '--context-length',1024,
                    '--output',PREPARED/f'sft_{split}','--tokenizer',TOKENIZER)
    status('data_ready', tokenizer=str(TOKENIZER), prepared=str(PREPARED))


def benchmark_trial(config, data, batch_size, checkpointing, device, dtype):
    c = replace(config, gradient_checkpointing=checkpointing)
    t = TrainConfig(batch_size=batch_size, grad_accum=1)
    model = Transformer(c).to(device)
    optimizer = make_optimizer(model,t,device)
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    generator = torch.Generator().manual_seed(123)
    times = []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(6):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        loss, tokens, norm, updated = train_update(model,optimizer,scaler,data,generator,t,device,dtype)
        torch.cuda.synchronize(device)
        if not updated:
            raise RuntimeError('Benchmark optimizer update skipped')
        if step >= 2:
            times.append(time.perf_counter()-started)
    return {'batch_size':batch_size,'gradient_checkpointing':checkpointing,
            'tokens_per_second':tokens/(sum(times)/len(times)), 'loss':loss,
            'peak_vram_gib':torch.cuda.max_memory_allocated(device)/2**30}


def benchmark(tokenizer, device):
    if device.type != 'cuda':
        raise RuntimeError('This bounded real-data training launcher requires CUDA')
    c,_ = load_config(ROOT/'configs/balanced.json')
    c.vocab_size = tokenizer.vocab_size
    data = TokenDataset(PREPARED/'pretrain','train',c.context_length,tokenizer)
    dtype = precision_for(device)
    results = []
    for batch, checkpointing in ((4,False),(8,False),(2,True)):
        gc.collect()
        torch.cuda.empty_cache()
        try:
            row = benchmark_trial(c,data,batch,checkpointing,device,dtype)
            results.append(row)
            print(f'Benchmark: {json.dumps(row)}',flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f'Benchmark: batch={batch} checkpointing={checkpointing} does not fit',flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    if not results:
        raise RuntimeError('No benchmark configuration fits')
    best = max(results,key=lambda r:r['tokens_per_second'])
    report = {'hardware':detect_hardware(str(device)), 'model':asdict(c), 'trials':results,'selected':best,
              'tokenizer_hash':tokenizer.fingerprint,'data_hash':data.fingerprint}
    write_json(RUN/'benchmark.json',report)
    status('benchmark_ready', selected=best)
    return report


def sample_answers(model, tokenizer, device, dtype):
    results = []
    for question in ('Hallo, wie geht es dir?', 'Was ist der Unterschied zwischen einer Katze und einem Hund?',
                     'Erkläre kurz, was Pflanzen zum Wachsen brauchen.',
                     'Ich heiße Eva und wohne in Wien. Wie heiße ich und wo wohne ich?',
                     'Fühlst du dich alleine?'):
        conversation = Conversation(tokenizer,model.config.context_length)
        conversation.add('user',question)
        prompt = conversation.build([],128)
        with amp_context(device,dtype):
            ids = list(generate(model,prompt,128,temperature=0,stop_ids=(EOS,END),
                       forbidden_ids=(PAD,BOS,SYSTEM,USER,ASSISTANT)))
        results.append({'question':question,'answer':tokenizer.decode(ids)})
    write_json(RUN/'quality_samples.json', {'reviewed':False,'samples':results})


def train_session(hours, tokenizer, device, report, target_tokens=None):
    stop_file = RUN/'STOP'
    plan_path = RUN/'plan.json'
    budget = hours*3600
    best = report['selected']
    initial_parent = None
    if target_tokens is not None and not plan_path.exists():
        source = ROOT/'runs/real_v1/pretrain/best.pt'
        initial_parent = load_checkpoint(source)
        rows = json.loads((PREPARED/'sft_train/meta.json').read_text(encoding='utf-8'))['rows']
        plan = continuation_plan(initial_parent, file_hash(source), report, hours, target_tokens, rows)
        write_json(plan_path, plan)
        save_config(ROOT/'configs/real_v2.json', ModelConfig(**plan['model']), TrainConfig(**plan['pretrain']))
    elif not plan_path.exists():
        c = ModelConfig(**report['model'])
        c.gradient_checkpointing = best['gradient_checkpointing']
        batch = best['batch_size']
        accum = max(1,16384//(batch*c.context_length))
        # Reserve time for validation, saving, SFT and final samples.
        steps = max(100, int(budget*0.78*best['tokens_per_second']/(batch*accum*c.context_length)))
        t = TrainConfig(steps=steps,batch_size=batch,grad_accum=accum,lr=3e-4,
            warmup_steps=min(500,max(20,steps//20)),eval_interval=250,eval_batches=12,
            save_interval=250,log_interval=20,precision='bf16' if report['hardware']['bf16'] else 'fp16')
        write_json(plan_path,{'model':asdict(c),'pretrain':asdict(t),'initial_hours':hours,
                              'benchmark':best,'tokenizer_hash':tokenizer.fingerprint})
        save_config(ROOT/'configs/real_v1.json',c,t)
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if target_tokens is not None:
        if plan.get('target_tokens') != target_tokens:
            raise ValueError('A different token target already exists; do not change a running schedule')
        if file_hash(ROOT/plan['parent']['checkpoint']) != plan['parent']['sha256']:
            raise ValueError('The parent checkpoint changed; the previous run is preserved')
    if plan['tokenizer_hash'] != tokenizer.fingerprint:
        raise ValueError('Tokenizer changed after training plan was created')
    c,t = ModelConfig(**plan['model']),TrainConfig(**plan['pretrain'])
    dtype = precision_for(device,t.precision)
    started = time.perf_counter()
    checkpoint_path = RUN/'pretrain/last.pt'
    resume = load_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    if resume is None and 'parent' in plan and initial_parent is None:
        initial_parent = load_checkpoint(ROOT/plan['parent']['checkpoint'])
    gate = (resume or initial_parent)['overfit'] if (resume or initial_parent) else overfit_test(tokenizer,device,dtype)
    if not gate['passed']:
        raise RuntimeError('Overfit gate failed')
    pretrain_tokens = resume['total_tokens'] if resume else 0
    if resume is None or resume['step'] < t.steps:
        status('pretraining',hours=hours,planned_steps=t.steps,parameters=c.parameter_count())
        train_data = TokenDataset(PREPARED/'pretrain','train',c.context_length,tokenizer)
        val_data = TokenDataset(PREPARED/'pretrain','val',c.context_length,tokenizer)
        seed_all(t.seed)
        model = Transformer(c).to(device)
        if initial_parent is not None:
            warm_start(model, initial_parent, tokenizer)
            del initial_parent
        trainer = Trainer(model,t,tokenizer,train_data,val_data,RUN/'pretrain',device,
                          resume=resume,overfit=gate)
        if resume is None and 'parent' in plan:
            trainer.best = evaluate(model, val_data, t, device, dtype)
            trainer.save('best.pt')
            write_json(RUN/'pretrain/baseline.json', {'val_loss': trainer.best, 'step': 0})
        del resume
        # Leave time for the final dialogue phase. If the token target takes longer,
        # pause at a checkpoint instead of silently shortening the learning schedule.
        reserve = max(1200, plan.get('sft_steps',3000)*16384/best['tokens_per_second']*1.2+120)
        pretrain_budget = budget-reserve if target_tokens is not None else budget*0.85
        result = trainer.run(max_seconds=max(1,pretrain_budget-(time.perf_counter()-started)),stop_file=stop_file)
        pretrain_tokens = trainer.total_tokens
        del trainer,model,train_data,val_data
        gc.collect()
        torch.cuda.empty_cache()
        if result['reason'] in ('stop_file','interrupted'):
            status('paused',stage='pretrain',result=result)
            return
        if result['step'] < t.steps:
            status('paused',stage='pretrain',reason='time_budget',result=result)
            return
    resume = None
    if stop_file.exists():
        status('paused',stage='before_sft')
        return
    remaining = budget-(time.perf_counter()-started)-60
    if remaining <= 0:
        status('paused',stage='before_sft',reason='time_budget')
        return
    status('sft',remaining_seconds=remaining)
    sft_path = RUN/'sft/last.pt'
    lineage_path = RUN/'sft/source.json'
    parent_hash = file_hash(RUN/'pretrain/best.pt')
    if sft_path.exists():
        lineage = json.loads(lineage_path.read_text(encoding='utf-8'))
        if lineage['pretrain_sha256'] != parent_hash:
            raise ValueError('Pretraining checkpoint changed after SFT began; use a new SFT run')
    else:
        write_json(lineage_path,{'pretrain_sha256':parent_hash})
    payload = load_checkpoint(sft_path if sft_path.exists() else RUN/'pretrain/best.pt')
    is_resume = payload['stage'] == 'sft'
    sft_train = SFTDataset(PREPARED/'sft_train',c.context_length,tokenizer)
    sft_val = SFTDataset(PREPARED/'sft_validation',c.context_length,tokenizer)
    if set(sft_train.meta['conversation_hashes']) & set(sft_val.meta['conversation_hashes']):
        raise ValueError('SFT train/validation overlap')
    sft_config = TrainConfig(**payload['training_config']) if is_resume else TrainConfig(
        steps=plan.get('sft_steps',max(100,min(3000,math.ceil(2*len(sft_train.x)/16)))),
        batch_size=plan.get('sft_batch_size',min(4,t.batch_size)),
        grad_accum=plan.get('sft_grad_accum',4),lr=3e-5,warmup_steps=100,weight_decay=0.01,eval_interval=100,
        eval_batches=20,save_interval=100,log_interval=20,precision=t.precision)
    seed_all(sft_config.seed)
    model = Transformer(c).to(device)
    model.load_state_dict(payload['model'])
    trainer = Trainer(model,sft_config,tokenizer,sft_train,sft_val,RUN/'sft',device,stage='sft',
                      resume=payload if is_resume else None,overfit=gate)
    if not is_resume:
        write_json(lineage_path, {'pretrain_sha256': parent_hash, 'pretrain_tokens': payload['total_tokens']})
    del payload
    if trainer.step < sft_config.steps:
        result = trainer.run(max_seconds=max(1,remaining),stop_file=stop_file)
        if result['reason'] in ('stop_file','interrupted'):
            status('paused',stage='sft',result=result)
            return
    best_state = load_checkpoint(RUN/'sft/best.pt')
    model.load_state_dict(best_state['model'])
    selected_sft_tokens = best_state['total_tokens']
    del best_state
    sample_answers(model,tokenizer,device,dtype)
    base_tokens = plan.get('parent', {}).get('total_tokens', 0)
    lineage = json.loads(lineage_path.read_text(encoding='utf-8'))
    status('finished_unreviewed' if trainer.step == sft_config.steps else 'paused',stage='sft',
           step=trainer.step,planned_steps=sft_config.steps,checkpoint=str(RUN/'sft/best.pt'),
           samples=str(RUN/'quality_samples.json'),elapsed_seconds=time.perf_counter()-started,
           processed_tokens=base_tokens+pretrain_tokens+trainer.total_tokens,
           selected_checkpoint_tokens=base_tokens+lineage.get('pretrain_tokens',pretrain_tokens)+selected_sft_tokens)


def main():
    global RUN
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--benchmark-only',action='store_true')
    p.add_argument('--hours',type=float)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--continue-to-tokens', type=int,
                   help='New real_v2 cycle from own real_v1 pretraining weights; cumulative input-token target')
    p.add_argument('--check', action='store_true', help='Verify continuation inputs on CPU; do not train or create a run')
    a = p.parse_args()
    if a.hours is not None and not 0 < a.hours <= 24:
        p.error('--hours must be between 0 and 24')
    if a.continue_to_tokens is not None:
        if a.continue_to_tokens <= 0 or a.hours is None or a.prepare_only or a.benchmark_only:
            p.error('Continuation requires --hours and a positive token target, without preparation/benchmark-only')
        RUN = ROOT/'runs/real_v2'
    if a.check:
        if a.continue_to_tokens is None:
            p.error('--check requires --continue-to-tokens and --hours')
        torch.set_num_threads(min(8,os.cpu_count() or 1))
        print(json.dumps(check_continuation(a.hours, a.continue_to_tokens), indent=2))
        print('Pruefung bestanden. Es wurde kein Training gestartet und kein neuer Lauf angelegt.')
        return 0
    os.chdir(ROOT)
    RUN.mkdir(parents=True,exist_ok=True)
    lock_path = RUN/'running.lock'
    for other in (ROOT/'runs/real_v1', ROOT/'runs/real_v2'):
        if other != RUN and (other/'running.lock').exists():
            old = json.loads((other/'running.lock').read_text())
            if psutil.pid_exists(old['pid']):
                raise SystemExit('Another training cycle is already running; finish or stop it first')
    if lock_path.exists():
        old = json.loads(lock_path.read_text())
        if psutil.pid_exists(old['pid']):
            raise SystemExit(f'A training launcher is already running: PID {old["pid"]}')
        lock_path.unlink()
    with lock_path.open('x') as f:
        json.dump({'pid':os.getpid()},f)
    try:
        with keep_awake(), capture_run_logs(a.continue_to_tokens is not None):
            try:
                if (RUN/'STOP').exists():
                    if not a.resume:
                        raise ValueError('Stop requested previously; use --resume to continue')
                    (RUN/'STOP').unlink()
                torch.set_num_threads(min(8,os.cpu_count() or 1))
                prepare()
                if (RUN/'STOP').exists():
                    status('paused',stage='data_ready')
                    return 0
                if a.prepare_only:
                    return 0
                tokenizer = BPETokenizer.load(TOKENIZER)
                device = select_device()
                benchmark_path = (ROOT/'runs/real_v1/benchmark.json' if a.continue_to_tokens is not None
                                  else RUN/'benchmark.json')
                if benchmark_path.exists() and not a.benchmark_only:
                    report = json.loads(benchmark_path.read_text(encoding='utf-8'))
                    if report['tokenizer_hash'] != tokenizer.fingerprint:
                        raise ValueError('Tokenizer changed; rerun --benchmark-only')
                else:
                    status('benchmarking')
                    report = benchmark(tokenizer,device)
                if a.benchmark_only or a.hours is None:
                    return 0
                train_session(a.hours,tokenizer,device,report,a.continue_to_tokens)
            except (InterruptedError, KeyboardInterrupt) as error:
                status('paused',reason=str(error) or 'Interrupted before the next completed checkpoint')
            except Exception as error:
                import traceback
                status('failed',error=str(error))
                traceback.print_exc()
                return 1
    finally:
        lock_path.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
