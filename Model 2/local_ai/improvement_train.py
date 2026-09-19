"""Use one authorized time window for several measured training attempts, keeping the best."""
import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch

from config import ModelConfig, TrainConfig
from data import file_hash
from improvement_data import DATA, GROUPS, build, datasets
from improvement_eval import eligible, evaluate_checkpoint, probes, selection_score
from learn import Tee, keep_awake, warm_start
from model import Transformer
from quality_train import live_lock, read_json, write_json
from start_chat import latest_metrics
from tokenizer import BPETokenizer
from train import Trainer, evaluate, load_checkpoint, seed_all

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/improvement_v5'
SOURCE=ROOT/'runs/dialogue_v4/sft/best.pt'
RECIPES=[
    ('natural_answers', [.18,.28,.20,.10,.14,.05,.05],6e-6),
    ('reading_practice', [.08,.20,.36,.08,.15,.08,.05],4e-6),
    ('worked_arithmetic', [.10,.20,.20,.10,.15,.20,.05],6e-6),
    ('balanced_review', [.12,.25,.25,.10,.15,.08,.05],2e-6),
    ('natural_review', [.18,.28,.20,.10,.14,.05,.05],2e-6),
    ('gentle_review', [.12,.25,.25,.10,.15,.08,.05],1e-6),
]
TERMINAL=('finished_unreviewed','paused','failed','budget_exhausted')


def status(phase, **fields):
    value={'phase':phase,'pid':os.getpid(),'updated_at':datetime.now(timezone.utc).isoformat(),**fields}
    write_json(RUN/'status.json',value)
    print(json.dumps(value,ensure_ascii=True),flush=True)


def current_status():
    value=read_json(RUN/'status.json')
    if value.get('phase')=='sft':
        value.update(latest_metrics(RUN/value['attempt']/'metrics.jsonl'))
    return value


def make_plan(model_config, source_hash, data_hash, deadline):
    now=time.time()
    if not now+180 < deadline <= now+4*3600+1:
        raise ValueError('Training deadline must be within the authorized four hours and leave three minutes')
    return {'version':1,'source':str(SOURCE),'source_sha256':source_hash,'model':model_config,
            'data_manifest_sha256':data_hash,'started_at':now,'deadline':deadline,
            'source_lineage_input_positions':2187502912,
            'groups':GROUPS,'recipes':[{'name':name,'weights':weights,'lr':lr} for name,weights,lr in RECIPES],
            'budget_policy':'One fixed deadline shared by all attempts, preparation, evaluation and resumes. '
                            'Reserve two minutes for final checks; final saving may overrun slightly.',
            'selection':'Within each attempt: minimum fixed validation loss. Across attempts: development '
                        'answer score with original context/reading/format floors and improved validation. '
                        'Final test never selects weights. Old checkpoints preserved.',
            'stagnation':'A stalled attempt leads to the next recipe, not automatic completion of the whole session.'}


def attempt(plan, record, tokenizer, device):
    folder=RUN/record['name']
    last=folder/'last.pt'
    resumed=last.exists()
    payload=load_checkpoint(last if resumed else record['source'])
    model=Transformer(ModelConfig(**plan['model'])).to(device)
    warm_start(model,payload,tokenizer)
    train_data,val_data=datasets(tokenizer,record['weights'])
    config=TrainConfig(**record['training'])
    trainer=Trainer(model,config,tokenizer,train_data,val_data,folder,device,stage='sft',
                    resume=payload if resumed else None,overfit=payload.get('overfit'))
    del payload
    gc.collect()
    if not resumed:
        trainer.best=evaluate(model,val_data,config,device,trainer.dtype)
        trainer.save('best.pt')
        trainer.save()
        record.update(baseline_loss=trainer.best,best_loss=trainer.best,bad_blocks=0)
        write_json(folder/'progress.json',record)
    reason='time_limit'
    while time.time()<min(record['deadline'],plan['deadline']-120) and trainer.step<config.steps:
        status('sft',attempt=record['name'],step=trainer.step,best_val=trainer.best,
               attempt_deadline=record['deadline'],deadline=plan['deadline'])
        target=min(config.steps,((trainer.step//config.eval_interval)+1)*config.eval_interval)
        seconds=min(record['deadline'],plan['deadline']-120)-time.time()
        if seconds<=0:
            break
        result=trainer.run(stop_after=target,max_seconds=seconds,stop_file=RUN/'STOP')
        if record['best_loss']-trainer.best>=.002:
            record['best_loss'],record['bad_blocks']=trainer.best,0
        else:
            record['bad_blocks']+=1
        record.update(step=trainer.step,processed_input_positions=trainer.total_tokens,best_validation=trainer.best)
        write_json(folder/'progress.json',record)
        if result['reason'] in ('stop_file','interrupted'):
            status('paused',attempt=record['name'],step=trainer.step,deadline=plan['deadline'])
            return False
        # Preserve the best weights; spend the remaining session on a different recipe.
        if record['bad_blocks']>=8 and trainer.step>=3000:
            reason='switch_recipe_after_stagnation'
            break
    trainer.save()
    if trainer.step>=config.steps:
        reason='steps'
    record.update(complete=True,reason=reason,best_validation=trainer.best,
                  step=trainer.step,processed_input_positions=trainer.total_tokens)
    write_json(folder/'progress.json',record)
    del trainer,model,train_data,val_data
    gc.collect()
    torch.cuda.empty_cache()
    return True


def run_session(plan,tokenizer,device):
    state=read_json(RUN/'progress.json')
    dev=probes('dev')
    if not state:
        status('evaluating_baseline',deadline=plan['deadline'])
        baseline=evaluate_checkpoint(SOURCE,tokenizer,device,dev,'before_dev')
        write_json(RUN/'before_dev.json',baseline)
        state={'next_attempt':0,'champion':str(SOURCE),'champion_result':'before_dev.json',
               'attempts':[],'processed_input_positions':0,'lineage_input_positions':plan['source_lineage_input_positions']}
        write_json(RUN/'progress.json',state)
    baseline=read_json(RUN/'before_dev.json')
    while time.time()<plan['deadline']-180 and state['next_attempt']<len(plan['recipes']):
        index=state['next_attempt']
        recipe=plan['recipes'][index]
        name=f'{index+1:02d}_{recipe["name"]}'
        folder=RUN/name
        folder.mkdir(exist_ok=True)
        record=read_json(folder/'progress.json')
        if not record:
            remaining=plan['deadline']-time.time()-120
            slots=len(plan['recipes'])-index
            deadline=min(plan['deadline']-120,time.time()+remaining/slots)
            config=TrainConfig(steps=12000,batch_size=4,grad_accum=4,lr=recipe['lr'],min_lr_ratio=.2,
                               warmup_steps=100,weight_decay=.02,eval_interval=500,eval_batches=192,
                               save_interval=500,log_interval=100,seed=2026091905,precision='bf16')
            record={**recipe,'name':name,'source':state['champion'],'deadline':deadline,
                    'parent_lineage_input_positions':state['lineage_input_positions'],'training':asdict(config)}
            write_json(folder/'progress.json',record)
        if not record.get('complete'):
            if not attempt(plan,record,tokenizer,device):
                return
        if 'original_validation' not in state:
            state['original_validation']=record['baseline_loss']
        status('comparing_attempt',attempt=name,deadline=plan['deadline'])
        candidate=evaluate_checkpoint(folder/'best.pt',tokenizer,device,dev,name+'_dev')
        write_json(folder/'development.json',candidate)
        champion=read_json(RUN/state['champion_result'])
        accepted=eligible(candidate,baseline,champion,record['best_validation'],state['original_validation'])
        record['accepted']=accepted
        if accepted:
            payload=load_checkpoint(folder/'best.pt')
            state['champion']=str(folder/'best.pt')
            state['champion_result']=str(Path(name)/'development.json')
            state['lineage_input_positions']=record['parent_lineage_input_positions']+payload['total_tokens']
            del payload
        state['attempts'].append({k:record[k] for k in ('name','step','best_validation','processed_input_positions','accepted')})
        state['processed_input_positions']+=record['processed_input_positions']
        state['next_attempt']=index+1
        write_json(RUN/'progress.json',state)
        print(json.dumps({'phase':'attempt_result','attempt':name,'accepted':accepted,
                          'scores':candidate['scores'],'score':selection_score(candidate),'next_attempt':index+2}),flush=True)
        if (RUN/'STOP').exists():
            status('paused',deadline=plan['deadline'],checkpoint=state['champion'])
            return
    status('evaluating_final',checkpoint=state['champion'],deadline=plan['deadline'])
    test=probes('test')
    before=evaluate_checkpoint(SOURCE,tokenizer,device,test,'before_final')
    after=evaluate_checkpoint(state['champion'],tokenizer,device,test,'after_final')
    if file_hash(SOURCE)!=plan['source_sha256']:
        raise RuntimeError('Source checkpoint changed')
    write_json(RUN/'quality_results.json',{'before':before,'after':after,'attempts':state['attempts'],
               'source_unchanged':True,'selected_checkpoint':state['champion'],
               'processed_input_positions':state['processed_input_positions'],
               'selected_lineage_input_positions':state['lineage_input_positions'],
               'note':'Final test only reports results. Natural-language keyword checks do not prove answer quality. '
                      'Read generated answers. Counts include padding/repeated examples.'})
    status('finished_unreviewed',checkpoint=state['champion'],before_scores=before['scores'],after_scores=after['scores'],
           processed_input_positions=state['processed_input_positions'],
           selected_lineage_input_positions=state['lineage_input_positions'],
           elapsed_seconds=time.time()-plan['started_at'],deadline=plan['deadline'])


def chat():
    value=current_status()
    while value.get('phase') not in TERMINAL:
        if not live_lock(RUN/'running.lock'):
            raise RuntimeError('Kein aktiver Verbesserungslauf. Training_4_Stunden.cmd starten.')
        print(json.dumps(value),flush=True)
        time.sleep(15)
        value=current_status()
    if value['phase'] not in ('finished_unreviewed','budget_exhausted'):
        raise RuntimeError('Lauf pausiert oder fehlgeschlagen. Siehe Status.')
    checkpoint=value.get('checkpoint',str(SOURCE))
    print('Bester ausgewaehlter Stand. Freie Sprachqualitaet bleibt experimentell.')
    return subprocess.call([sys.executable,str(ROOT/'chat.py'),'--checkpoint',checkpoint,
                            '--memory',str(ROOT/'artifacts/improvement_v5_memory.sqlite'),
                            '--temperature','0','--repetition-penalty','1.0','--no-auto-memory'],cwd=ROOT)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--resume',action='store_true')
    p.add_argument('--status',action='store_true')
    p.add_argument('--chat',action='store_true')
    p.add_argument('--deadline',type=float)
    args=p.parse_args()
    if args.status:
        print(json.dumps(current_status(),indent=2)); return 0
    if args.chat:
        return chat()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA/BF16 required')
    for lock in (ROOT/'runs').glob('*/running.lock'):
        if live_lock(lock):
            raise RuntimeError(f'Active training: {lock}')
    plan=read_json(RUN/'plan.json')
    if current_status().get('phase')=='finished_unreviewed':
        print('Dieser Vergleich ist abgeschlossen; kein automatischer Neustart.'); return 0
    if plan and not args.resume:
        raise RuntimeError('Existing session; use --resume')
    # An omitted deadline inherits the already authorized original window, never a fresh four hours.
    deadline=plan['deadline'] if plan else args.deadline or read_json(ROOT/'runs/dialogue_v4/plan.json')['deadline']
    if time.time()>=deadline:
        state=read_json(RUN/'progress.json')
        status('budget_exhausted',checkpoint=state.get('champion',str(SOURCE))); return 0
    lock=RUN/'running.lock'
    lock.unlink(missing_ok=True)
    with lock.open('x',encoding='utf-8') as stream:
        json.dump({'pid':os.getpid(),'process_created':psutil.Process().create_time()},stream)
    (RUN/'STOP').unlink(missing_ok=True)
    try:
        with keep_awake():
            status('preparing_data',deadline=deadline)
            build()
            torch.set_num_threads(8)
            seed_all(2026091905)
            tokenizer=BPETokenizer.load(ROOT/'artifacts/real_v1/tokenizer.json')
            if not plan:
                payload=load_checkpoint(SOURCE)
                if not (payload.get('overfit') or {}).get('passed'):
                    raise ValueError('Source lacks a successful overfit test')
                plan=make_plan(payload['model_config'],file_hash(SOURCE),file_hash(DATA/'manifest.json'),deadline)
                del payload
                write_json(RUN/'plan.json',plan)
            if file_hash(DATA/'manifest.json')!=plan['data_manifest_sha256'] or file_hash(SOURCE)!=plan['source_sha256']:
                raise ValueError('Source or data changed')
            run_session(plan,tokenizer,torch.device('cuda'))
    except BaseException as error:
        status('failed',error=str(error))
        traceback.print_exc()
        raise
    finally:
        lock.unlink(missing_ok=True)
    return 0


if __name__=='__main__':
    RUN.mkdir(parents=True,exist_ok=True)
    if '--status' in sys.argv or '--chat' in sys.argv:
        raise SystemExit(main())
    with (RUN/'training.log').open('a',encoding='utf-8',buffering=1) as out, \
            (RUN/'training-error.log').open('a',encoding='utf-8',buffering=1) as err:
        with redirect_stdout(Tee(sys.stdout,out)),redirect_stderr(Tee(sys.stderr,err)):
            raise SystemExit(main())
