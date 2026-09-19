from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from config import ModelConfig, TrainConfig
from dialogue_data import arithmetic_rows
from dialogue_train import continuation_gate, make_plan
from model import Transformer
from tokenizer import BPETokenizer
from train import Trainer, load_checkpoint, seed_all, supervised_loss, train_update


def tiny_model():
    return Transformer(ModelConfig(vocab_size=32, n_layers=1, n_heads=2, n_kv_heads=1,
                                   embedding_dim=16, ffn_dim=32, context_length=32))


def test_example_loss_equal_weight_and_padding():
    model = tiny_model()
    x = torch.tensor([[7, 8, 9, 10], [11, 12, 13, 14]])
    y = torch.tensor([[-100, -100, -100, 15], [12, 13, 14, 15]])
    expected = (model(x[:1], y[:1])[1]+model(x[1:], y[1:])[1])/2
    actual = supervised_loss(model, x, y, 'example')
    torch.testing.assert_close(actual, expected)
    padded_x = torch.cat([x, torch.zeros(2, 3, dtype=torch.long)], dim=1)
    padded_y = torch.cat([y, torch.full((2, 3), -100, dtype=torch.long)], dim=1)
    torch.testing.assert_close(supervised_loss(model, padded_x, padded_y, 'example'), actual)
    assert not torch.isclose(actual, model(x, y)[1])
    with pytest.raises(ValueError, match='no supervised'):
        supervised_loss(model, x, torch.full_like(y, -100), 'example')


class ShortData:
    loss_reduction = 'example'
    fingerprint = 'unequal-answer-lengths'

    def batch(self, size, generator):
        idx = torch.randint(1, 5, (size,), generator=generator)
        x = torch.arange(7, 15).repeat(size, 1)
        y = x.clone()
        for i, n in enumerate(idx):
            y[i, :8-int(n)] = -100
        return x, y


def test_example_accumulation_matches_large_batch():
    seed_all(71)
    first = tiny_model()
    second = deepcopy(first)
    for model, batch, accum in [(first, 4, 1), (second, 2, 2)]:
        config = TrainConfig(batch_size=batch, grad_accum=accum, grad_clip=1000)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        train_update(model, optimizer, scaler, ShortData(), torch.Generator().manual_seed(7),
                     config, torch.device('cpu'), torch.float32)
    for p, q in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(p, q, atol=1e-7, rtol=1e-6)


def test_example_checkpoint_resume_and_objective_guard(tmp_path):
    tokenizer = BPETokenizer.train(['Ein Test mit kurzen Antworten.'], 280, 1)
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, n_layers=1, n_heads=2,
                              n_kv_heads=1, embedding_dim=16, ffn_dim=32, context_length=32)
    config = TrainConfig(steps=4, batch_size=2, grad_accum=2, eval_interval=2,
                         eval_batches=2, save_interval=2, precision='fp32')
    data = ShortData()
    def create(folder, resume=None):
        return Trainer(Transformer(model_config), config, tokenizer, data, data,
                       tmp_path/folder, torch.device('cpu'), resume=resume)
    seed_all(8)
    full = create('full')
    full.run()
    seed_all(8)
    partial = create('partial')
    (tmp_path/'STOP').touch()
    assert partial.run(stop_file=tmp_path/'STOP')['reason'] == 'stop_file'
    (tmp_path/'STOP').unlink()
    partial.run(stop_after=2)
    payload = load_checkpoint(tmp_path/'partial/last.pt')
    resumed = create('resumed', payload)
    resumed.run()
    for p, q in zip(full.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    payload['loss_reduction'] = 'token'
    with pytest.raises(ValueError, match='loss reduction'):
        create('wrong', payload)


def result(context, reading, math):
    return {'scores': {'context': {'correct': context, 'total': 24},
                       'reading': {'correct': reading, 'total': 24},
                       'arithmetic': {'correct': math, 'total': 12}}, 'answers': []}


def test_gate_requires_answers_not_only_lower_loss():
    before = result(8, 1, 0)
    assert not continuation_gate(before, before, 3, 1)['passed']
    assert continuation_gate(before, result(12, 1, 0), 3, 2)['passed']
    assert not continuation_gate(before, result(7, 12, 10), 3, 2)['passed']
    assert not continuation_gate(before, result(12, 1, 0), 3, 3.1)['passed']


def test_arithmetic_reversed_pairs_stay_in_same_split():
    seen = {}
    for split, row in arithmetic_rows(20):
        key = row['source_id']
        if key in seen:
            assert seen[key] == split
        seen[key] = split
    assert len(set(seen.values())) == 4


def test_v4_fixed_budget_and_old_source_preserved():
    plan = make_plan(asdict(ModelConfig()), 'source', 'data', 1000)
    assert plan['deadline'] == 15400
    assert plan['trial_seconds'] == 1200
    assert plan['maximum_hours'] == 4
    assert plan['loss_reduction'] == 'example'
    assert 'quality_v3' in plan['source']
    assert not plan['default_chat_changed']
