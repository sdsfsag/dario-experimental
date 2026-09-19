import json
import time
from dataclasses import asdict

import pytest
import torch

from config import ModelConfig
from config import TrainConfig
from data import file_hash
from model import Transformer
from tokenizer import BPETokenizer
from train import Trainer, load_checkpoint
import quality_train
from quality_data import Mixture, acceptable, partition
from quality_train import exhausted, live_lock, make_plan, write_json


class ExampleData:
    def __init__(self, token, length):
        self.token, self.length, self.fingerprint = token, length, str(token)

    def batch(self, size, generator):
        x = torch.zeros(size, 64, dtype=torch.long)
        y = torch.full_like(x, -100)
        x[:, :self.length] = self.token
        y[:, :self.length] = self.token+1
        return x, y


def test_mixture_preserves_targets_and_resume_rng():
    data = Mixture([ExampleData(10, 21), ExampleData(20, 9)], [.6, .4], trim=True)
    generator = torch.Generator().manual_seed(44)
    state = generator.get_state()
    x, y = data.batch(32, generator)
    assert x.shape == y.shape == (32, 32)
    assert set(x[:, 0].tolist()) == {10, 20}
    assert torch.all(y[x == 0] == -100)
    assert torch.equal(y[x != 0], x[x != 0]+1)
    generator.set_state(state)
    resumed = data.batch(32, generator)
    assert all(torch.equal(a, b) for a, b in zip((x, y), resumed))
    assert data.fingerprint != Mixture(data.datasets, [.4, .6], trim=True).fingerprint


def test_language_mixture_keeps_all_targets():
    data = Mixture([ExampleData(10, 64)], [1], trim=False)
    x, y = data.batch(2, torch.Generator())
    assert x.shape == (2, 64)
    assert torch.all(y == 11)


def test_partition_stable_normalized_and_three_way():
    assert partition('Ein Text.\nNoch ein Satz.') == partition(' EIN TEXT. noch ein Satz. ')
    values = {partition(f'Unique document {i}') for i in range(1000)}
    assert values == {'train', 'validation', 'test'}


def test_filter_rejects_repeated_or_broken_text():
    assert acceptable('Eine Katze spielt im Garten. Danach geht sie ins Haus und schläft auf einem Kissen.')
    assert not acceptable('Der Hund ist ein kleiner Baum. '*20)
    assert not acceptable('Text mit falschem Unicode \ufffd in der Ausgabe.')
    assert not acceptable('Text mit <|assistant|> Rollenmarkierung.')


def test_four_hour_budget_survives_resume_without_extension():
    plan = make_plan(asdict(ModelConfig()), 'parent', 'baseline', 1000)
    assert plan['deadline'] == 1000+14400
    assert not exhausted(plan, 'pretrain', now=1000+9000)
    assert exhausted(plan, 'pretrain', now=1000+9900)
    resumed = json.loads(json.dumps(plan))
    assert exhausted(resumed, 'sft', now=1000+14400)
    assert resumed['default_chat_changed'] is False


def test_atomic_status_and_stale_lock(tmp_path):
    path = tmp_path/'status.json'
    write_json(path, {'phase': 'pretrain', 'step': 100})
    assert json.loads(path.read_text())['step'] == 100
    assert not path.with_suffix('.json.tmp').exists()
    assert not live_lock(tmp_path/'missing.lock')
    write_json(path, {'pid': -1, 'process_created': 0})
    assert not live_lock(path)


def test_stage_stop_resume_and_sft_handoff_preserve_source(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    tokenizer = BPETokenizer.train(['Ein kleiner Funktionstest.'], 300, 1)
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, n_layers=1, n_heads=2,
                              n_kv_heads=1, embedding_dim=16, ffn_dim=32, context_length=64)
    config = TrainConfig(steps=2, batch_size=1, grad_accum=1, warmup_steps=0,
                         eval_interval=1, eval_batches=1, save_interval=1, precision='fp32')
    data = ExampleData(10, 64)
    source = tmp_path/'source'
    trainer = Trainer(Transformer(model_config), config, tokenizer, data, data, source,
                      torch.device('cpu'), overfit={'passed': True})
    trainer.save('best.pt')
    digest = file_hash(source/'best.pt')
    run = tmp_path/'new'
    run.mkdir()
    monkeypatch.setattr(quality_train, 'RUN', run)
    monkeypatch.setattr(quality_train, 'SOURCE', source/'best.pt')
    monkeypatch.setattr(quality_train, 'datasets', lambda *args: (data, data))
    plan = make_plan(asdict(model_config), digest, 'baseline', time.time())
    plan['pretrain'] = plan['sft'] = asdict(config)
    state = {}
    (run/'STOP').touch()
    assert not quality_train.run_stage('pretrain', plan, tokenizer, torch.device('cpu'), state)
    assert quality_train.read_json(run/'status.json')['phase'] == 'paused'
    (run/'STOP').unlink()
    assert quality_train.run_stage('pretrain', plan, tokenizer, torch.device('cpu'), state)
    assert state['pretrain']['complete'] and state['pretrain']['step'] == 2
    assert quality_train.run_stage('sft', plan, tokenizer, torch.device('cpu'), state)
    assert load_checkpoint(run/'sft/last.pt')['stage'] == 'sft'
    assert file_hash(source/'best.pt') == digest
