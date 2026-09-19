import time
from dataclasses import asdict

import pytest
import torch

from config import ModelConfig,TrainConfig
from data import file_hash
from improvement_data import add_working
from improvement_eval import eligible,grade,natural_probes,probe_key
import improvement_train
from improvement_train import make_plan,attempt
from model import Transformer
from quality_data import dialog
from quality_train import read_json
from test_dialogue_v4 import ShortData
from tokenizer import BPETokenizer
from train import Trainer,load_checkpoint


def test_worked_arithmetic_preserves_value():
    for a in range(30):
        for b in range(a+1):
            for op in ('+','-'):
                row=dialog(f'{a} {op} {b} = ?',str(a+b if op=='+' else a-b),'test','key')
                result=add_working(row)
                assert result['messages'][-1]['content'].endswith(f'Ergebnis: {a+b if op=="+" else a-b}.')
                assert result['source_id']=='key'


def test_rubrics_accept_equivalent_number_but_reject_fiction():
    row={'group':'natural','answer':'Eine Stunde hat 60 Minuten.','hit_length_limit':False,
         'contains':['sechzig'],'forbidden':[]}
    assert grade(row)
    row.update(answer='Corvin wurde im Jahr 1932 geboren.',contains=['corvin'],forbidden=['geboren'])
    assert not grade(row)
    assert not ({probe_key(r) for r in natural_probes('dev')} & {probe_key(r) for r in natural_probes('test')})


def benchmark(context=20,reading=14,format=6,natural=3,arithmetic=0):
    return {'scores':{g:{'correct':v,'total':24} for g,v in locals().items()}}


def test_promotion_requires_gain_and_preserves_source_skills():
    base=benchmark()
    assert not eligible(base,base,base,1,2)
    assert eligible(benchmark(natural=5),base,base,1,2)
    assert not eligible(benchmark(context=10,natural=16),base,base,1,2)
    assert not eligible(benchmark(natural=5),base,base,3,2)


def test_budget_not_extended_by_new_attempts(monkeypatch):
    monkeypatch.setattr(improvement_train.time,'time',lambda:1000)
    plan=make_plan(asdict(ModelConfig()),'source','data',1600)
    assert plan['deadline']==1600
    assert len(plan['recipes'])==6
    assert 'fixed deadline' in plan['budget_policy']
    with pytest.raises(ValueError):
        make_plan({},'s','d',20000)


def test_attempt_safe_stop_resume_and_preserved_source(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    tokenizer=BPETokenizer.train(['Kurze Antworten prüfen.'],280,1)
    c=ModelConfig(vocab_size=tokenizer.vocab_size,n_layers=1,n_heads=2,n_kv_heads=1,
                  embedding_dim=16,ffn_dim=32,context_length=32)
    t=TrainConfig(steps=2,batch_size=1,grad_accum=1,eval_interval=1,eval_batches=1,
                  save_interval=1,log_interval=1,precision='fp32')
    data=ShortData()
    source=tmp_path/'source'
    trainer=Trainer(Transformer(c),t,tokenizer,data,data,source,torch.device('cpu'),stage='sft')
    trainer.save('best.pt')
    digest=file_hash(source/'best.pt')
    run=tmp_path/'new'
    run.mkdir()
    monkeypatch.setattr(improvement_train,'RUN',run)
    monkeypatch.setattr(improvement_train,'datasets',lambda *a:(data,data))
    plan={'model':asdict(c),'deadline':time.time()+600}
    record={'name':'trial','source':str(source/'best.pt'),'weights':[1],
            'deadline':time.time()+500,'training':asdict(t)}
    (run/'STOP').touch()
    assert not attempt(plan,record,tokenizer,torch.device('cpu'))
    assert read_json(run/'status.json')['phase']=='paused'
    (run/'STOP').unlink()
    assert attempt(plan,record,tokenizer,torch.device('cpu'))
    assert record['complete'] and record['step']==2
    assert file_hash(source/'best.pt')==digest
    assert load_checkpoint(run/'trial/last.pt')['step']==2
