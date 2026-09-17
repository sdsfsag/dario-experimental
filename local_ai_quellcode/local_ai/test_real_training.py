import json
import hashlib
import sqlite3

import pytest
import torch

from config import ModelConfig, TrainConfig
from download_data import accepted_message, extract_dialogues, normalized, split_for
from tokenizer import BPETokenizer
from train import Trainer, load_checkpoint


def test_dialogue_tree_split_and_missing_parents():
    def row(mid,parent,role,text,rank=None,approved=True):
        return dict(message_id=mid,parent_id=parent,role=role,text=text,lang='de',
                    deleted=False,review_result=approved,rank=rank,labels={})
    rows = [row('a',None,'prompter','Was ist Wasser?'),row('b','a','assistant','Wasser ist H2O.',0),
            row('c','b','prompter','Und Eis?'),row('d','c','assistant','Eis ist festes Wasser.',0),
            row('bad','missing','assistant','Ohne Kontext.',0),row('low','a','assistant','Falsch.',2)]
    result = list(extract_dialogues(rows))
    assert len(result)==1
    split,conversation = result[0]
    assert split == split_for(normalized('Was ist Wasser?'))
    assert len(conversation['messages']) == 5
    assert conversation['source_ids']==['a','b','c','d']
    assert not accepted_message(row('x',None,'prompter','Nicht bewertet.',approved=False))


def test_training_time_limit_and_stop_file(tmp_path):
    torch.set_num_threads(1)
    tokenizer = BPETokenizer.train(['Training on real text.'],280,1)
    c = ModelConfig(tokenizer.vocab_size,1,2,1,16,32,16)
    t = TrainConfig(steps=20,batch_size=1,grad_accum=1,eval_batches=1)
    class Data:
        fingerprint='timer-test'
        def batch(self,batch_size,generator):
            x=torch.randint(7,tokenizer.vocab_size,(batch_size,9),generator=generator)
            return x[:,:-1],x[:,1:]
    data=Data()
    trainer=Trainer(__import__('model').Transformer(c),t,tokenizer,data,data,tmp_path/'timed',torch.device('cpu'))
    result=trainer.run(max_seconds=0.00001)
    assert result['reason']=='time_limit' and result['step']==1
    assert load_checkpoint(tmp_path/'timed/last.pt')['step']==1
    assert (tmp_path/'timed/best.pt').exists()
    stop=tmp_path/'STOP'
    stop.write_text('stop')
    result=trainer.run(stop_file=stop)
    assert result['reason']=='stop_file' and result['step']==1


def test_recover_interrupted_corpus_preparation(tmp_path):
    from data import prepare_corpus
    tokenizer = BPETokenizer.train(['Train text and validation.'],280,1)
    training = 'Train text. '*30
    train_file, val_file = tmp_path/'train.txt', tmp_path/'val.txt'
    train_file.write_text(training,encoding='utf-8')
    val_file.write_text('Validation text. '*30,encoding='utf-8')
    folder=tmp_path/'prepared'
    folder.mkdir()
    db=sqlite3.connect(folder/'dedup.tmp.sqlite')
    db.execute('CREATE TABLE seen(hash TEXT PRIMARY KEY)')
    db.execute('INSERT INTO seen VALUES (?)',(hashlib.sha256(training.strip().encode()).hexdigest(),))
    db.commit()
    db.close()
    meta=prepare_corpus([train_file],folder,tokenizer,[val_file])
    assert meta['counts']['duplicates']==0
    assert meta['counts']['train']>100


def test_preparation_stop_request(tmp_path,monkeypatch):
    import learn
    monkeypatch.setattr(learn,'RUN',tmp_path)
    (tmp_path/'STOP').write_text('stop')
    monkeypatch.setattr(learn.subprocess,'run',lambda *a,**k:pytest.fail('Command must not start'))
    with pytest.raises(InterruptedError):
        learn.command('tokenizer.py')
