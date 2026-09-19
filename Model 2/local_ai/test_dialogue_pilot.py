import re

import pytest
import torch

from config import ModelConfig, TrainConfig
from data import encode_conversation
from dialogue_pilot import CompactDataset, scores, strict_match
from model import Transformer
from pilot_data import build_data, pair, write_data
from tokenizer import END, PAD, BPETokenizer
from train import Trainer, load_checkpoint


def test_pilot_splits_and_reserved_tasks(tmp_path):
    train, val, test = write_data(tmp_path)
    assert (train, val, test) == build_data() == write_data(tmp_path)
    reserved = {tuple(row['operands']) for row in test if row['group'] == 'arithmetic'}
    signatures = []
    for rows in (train, val):
        keys = set()
        for row in rows:
            assert row['synthetic'] is True
            if row['group'] == 'arithmetic':
                question, answer = row['messages'][-2:]
                a, b = sorted(map(int, re.findall(r'\d+', question['content'])))
                assert (a, b) not in reserved
                assert int(answer['content']) == a+b
                keys.add((a, b))
        signatures.append(keys)
    assert not signatures[0] & signatures[1]
    for group in ('copy', 'context'):
        train_answers = {r['messages'][-1]['content'] for r in train if r['group'] == group}
        test_answers = {r['expected'] for r in test if r['group'] == group}
        assert not train_answers & test_answers
    assert len(test) == 35


def test_pilot_preserves_existing_data(tmp_path):
    write_data(tmp_path)
    (tmp_path/'train.jsonl').write_text('different', encoding='utf-8')
    with pytest.raises(FileExistsError):
        write_data(tmp_path)


@pytest.fixture
def compact_data():
    rows = [pair('dialog', 'Hallo', 'Guten Tag'), pair('copy', 'Wort: Sonne', 'Sonne')]
    tokenizer = BPETokenizer.train([m['content'] for r in rows for m in r['messages']], 512, 1)
    return rows, tokenizer, CompactDataset(rows, tokenizer, max_length=512)


def test_compact_padding_and_shift(compact_data):
    rows, tokenizer, data = compact_data
    for row, (x, y) in zip(rows, data.rows):
        ids, labels = encode_conversation(row['messages'], tokenizer)
        assert x.tolist() == ids[:-1]
        assert y.tolist() == labels[1:]
        assert y[-1] == END
    x, y = data.batch(8, torch.Generator().manual_seed(17))
    for bx, by in zip(x, y):
        candidates = [(sx, sy) for sx, sy in data.rows if torch.equal(bx[:len(sx)], sx)]
        assert len(candidates) == 1
        sx, sy = candidates[0]
        assert torch.equal(by[:len(sy)], sy)
        assert torch.all(bx[len(sx):] == PAD)
        assert torch.all(by[len(sy):] == -100)
    with pytest.raises(ValueError, match='no truncation'):
        CompactDataset(rows, tokenizer, max_length=8)


def test_strict_scores_reject_partial_matches():
    assert strict_match(' Banane. ', 'Banane')
    assert strict_match('43', '43')
    assert not strict_match('43 oder 44', '43')
    assert not strict_match('Die Antwort ist 43', '43')
    assert not strict_match('Apfel', 'Banane')
    assert strict_match('anything', None) is None
    assert scores([{'group': 'open_ended', 'correct': None}, {'group': 'copy', 'correct': True}]) == {
        'copy': {'correct': 1, 'total': 1}}


def test_compact_trainer_checkpoint_compatibility(compact_data, tmp_path):
    torch.set_num_threads(1)
    rows, tokenizer, data = compact_data
    model = Transformer(ModelConfig(vocab_size=tokenizer.vocab_size, n_layers=1, n_heads=2,
                                   n_kv_heads=1, embedding_dim=16, ffn_dim=32, context_length=512))
    config = TrainConfig(steps=2, batch_size=2, grad_accum=1, warmup_steps=0,
                         eval_batches=1, eval_interval=1, save_interval=1)
    trainer = Trainer(model, config, tokenizer, data, data, tmp_path, torch.device('cpu'), stage='sft')
    assert trainer.run(stop_after=1)['step'] == 1
    state = load_checkpoint(tmp_path/'last.pt')
    assert state['data_hashes'] == [data.fingerprint]*2
    resumed = Trainer(model, config, tokenizer, data, data, tmp_path, torch.device('cpu'),
                      stage='sft', resume=state)
    assert resumed.run()['step'] == 2
