import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
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
from train import Trainer, amp_context, load_checkpoint, make_optimizer, overfit_test, seed_all, train_update

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data/real_v1'
PREPARED = ROOT/'prepared/real_v1'
TOKENIZER = ROOT/'artifacts/real_v1/tokenizer.json'
RUN = ROOT/'runs/real_v1'


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


def train_session(hours, tokenizer, device, report):
    stop_file = RUN/'STOP'
    plan_path = RUN/'plan.json'
    budget = hours*3600
    best = report['selected']
    if not plan_path.exists():
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
    if plan['tokenizer_hash'] != tokenizer.fingerprint:
        raise ValueError('Tokenizer changed after training plan was created')
    c,t = ModelConfig(**plan['model']),TrainConfig(**plan['pretrain'])
    dtype = precision_for(device,t.precision)
    started = time.perf_counter()
    checkpoint_path = RUN/'pretrain/last.pt'
    resume = load_checkpoint(checkpoint_path) if checkpoint_path.exists() else None
    gate = resume['overfit'] if resume else overfit_test(tokenizer,device,dtype)
    if not gate['passed']:
        raise RuntimeError('Overfit gate failed')
    if resume is None or resume['step'] < t.steps:
        status('pretraining',hours=hours,planned_steps=t.steps,parameters=c.parameter_count())
        train_data = TokenDataset(PREPARED/'pretrain','train',c.context_length,tokenizer)
        val_data = TokenDataset(PREPARED/'pretrain','val',c.context_length,tokenizer)
        seed_all(t.seed)
        model = Transformer(c).to(device)
        trainer = Trainer(model,t,tokenizer,train_data,val_data,RUN/'pretrain',device,
                          resume=resume,overfit=gate)
        del resume
        result = trainer.run(max_seconds=max(1,budget*0.85-(time.perf_counter()-started)),stop_file=stop_file)
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
        steps=max(100,min(3000,math.ceil(2*len(sft_train.x)/16))),batch_size=min(4,t.batch_size),
        grad_accum=4,lr=3e-5,warmup_steps=100,weight_decay=0.01,eval_interval=100,
        eval_batches=20,save_interval=100,log_interval=20,precision=t.precision)
    seed_all(sft_config.seed)
    model = Transformer(c).to(device)
    model.load_state_dict(payload['model'])
    trainer = Trainer(model,sft_config,tokenizer,sft_train,sft_val,RUN/'sft',device,stage='sft',
                      resume=payload if is_resume else None,overfit=gate)
    del payload
    if trainer.step < sft_config.steps:
        result = trainer.run(max_seconds=max(1,remaining),stop_file=stop_file)
        if result['reason'] in ('stop_file','interrupted'):
            status('paused',stage='sft',result=result)
            return
    best_state = load_checkpoint(RUN/'sft/best.pt')
    model.load_state_dict(best_state['model'])
    del best_state
    sample_answers(model,tokenizer,device,dtype)
    status('finished_unreviewed' if trainer.step == sft_config.steps else 'paused',stage='sft',
           step=trainer.step,planned_steps=sft_config.steps,checkpoint=str(RUN/'sft/best.pt'),
           samples=str(RUN/'quality_samples.json'),elapsed_seconds=time.perf_counter()-started)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--benchmark-only',action='store_true')
    p.add_argument('--hours',type=float)
    p.add_argument('--resume',action='store_true')
    a = p.parse_args()
    if a.hours is not None and not 0 < a.hours <= 24:
        p.error('--hours must be between 0 and 24')
    os.chdir(ROOT)
    RUN.mkdir(parents=True,exist_ok=True)
    lock_path = RUN/'running.lock'
    if lock_path.exists():
        old = json.loads(lock_path.read_text())
        if psutil.pid_exists(old['pid']):
            raise SystemExit(f'A training launcher is already running: PID {old["pid"]}')
        lock_path.unlink()
    with lock_path.open('x') as f:
        json.dump({'pid':os.getpid()},f)
    try:
        if (RUN/'STOP').exists():
            if not a.resume:
                raise ValueError('Stop requested previously; use --resume to continue')
            (RUN/'STOP').unlink()
        torch.set_num_threads(min(8,os.cpu_count() or 1))
        prepare()
        if (RUN/'STOP').exists():
            status('paused',stage='data_ready')
            return
        if a.prepare_only:
            return
        tokenizer = BPETokenizer.load(TOKENIZER)
        device = select_device()
        if (RUN/'benchmark.json').exists() and not a.benchmark_only:
            report = json.loads((RUN/'benchmark.json').read_text(encoding='utf-8'))
            if report['tokenizer_hash'] != tokenizer.fingerprint:
                raise ValueError('Tokenizer changed; rerun --benchmark-only')
        else:
            status('benchmarking')
            report = benchmark(tokenizer,device)
        if a.benchmark_only or a.hours is None:
            return
        train_session(a.hours,tokenizer,device,report)
    except InterruptedError as error:
        status('paused',reason=str(error))
    except Exception as error:
        status('failed',error=str(error))
        raise
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
