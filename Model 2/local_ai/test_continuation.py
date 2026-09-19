import json
from dataclasses import asdict

import pytest
import torch

import learn
import train
from config import ModelConfig, TrainConfig
from data import TokenDataset, file_hash, prepare_corpus, prepare_sft
from model import Transformer
from tokenizer import BPETokenizer
from train import Trainer, load_checkpoint, seed_all


@pytest.fixture
def continuation_inputs(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(learn, 'ROOT', tmp_path)
    monkeypatch.setattr(learn, 'RUN', tmp_path/'runs/real_v2')
    monkeypatch.setattr(learn, 'PREPARED', tmp_path/'prepared/real_v1')
    monkeypatch.setattr(learn, 'TOKENIZER', tmp_path/'artifacts/real_v1/tokenizer.json')
    tokenizer = BPETokenizer.train(['Die Katze sitzt da. Ein Hund spielt. Frage Antwort.'], 300, 1)
    tokenizer.save(learn.TOKENIZER)
    c = ModelConfig(tokenizer.vocab_size, 1, 2, 1, 16, 32, 128)
    t = TrainConfig(steps=2, batch_size=1, grad_accum=1, eval_batches=1, precision='fp32')
    (tmp_path/'train.txt').write_text('Die Katze sitzt da. '*100, encoding='utf-8')
    (tmp_path/'val.txt').write_text('Ein Hund spielt. '*100, encoding='utf-8')
    prepare_corpus([tmp_path/'train.txt'], learn.PREPARED/'pretrain', tokenizer, [tmp_path/'val.txt'])
    train_data = TokenDataset(learn.PREPARED/'pretrain', 'train', c.context_length, tokenizer)
    val_data = TokenDataset(learn.PREPARED/'pretrain', 'val', c.context_length, tokenizer)
    seed_all(123)
    parent_folder = tmp_path/'runs/real_v1/pretrain'
    Trainer(Transformer(c), t, tokenizer, train_data, val_data, parent_folder,
            torch.device('cpu'), overfit={'passed': True, 'fixture_only': True}).run()
    for split, question in (('train', 'Frage eins?'), ('validation', 'Frage zwei?')):
        path = tmp_path/f'{split}.jsonl'
        path.write_text(json.dumps({'messages': [{'role': 'user', 'content': question},
                                                {'role': 'assistant', 'content': 'Eine kurze Antwort.'}]}))
        prepare_sft([path], learn.PREPARED/f'sft_{split}', tokenizer, c.context_length)
    report = {'selected': {'batch_size': 8, 'tokens_per_second': 45000,
                           'gradient_checkpointing': False}, 'tokenizer_hash': tokenizer.fingerprint}
    (tmp_path/'runs/real_v1/benchmark.json').write_text(json.dumps(report))
    parent = load_checkpoint(parent_folder/'best.pt')
    target = parent['total_tokens'] + 2*16384 + 100*16*c.context_length
    return tokenizer, report, parent, target


def test_plan_counts_parent_once_and_rounds_by_one_update(continuation_inputs):
    _, report, parent, target = continuation_inputs
    plan = learn.continuation_plan(parent, 'sha', report, 9, target+1, 1)
    assert plan['pretrain']['steps'] == 3
    assert target+1 <= plan['planned_processed_tokens'] < target+1+16384
    assert plan['parent']['total_tokens'] == parent['total_tokens']
    assert plan['model'] == parent['model_config']
    with pytest.raises(ValueError, match='leave room'):
        learn.continuation_plan(parent, 'sha', report, 9, 1, 1)


def test_warm_start_copies_weights_without_mutating_parent(continuation_inputs):
    tokenizer, _, parent, _ = continuation_inputs
    model = Transformer(ModelConfig(**parent['model_config']))
    learn.warm_start(model, parent, tokenizer)
    for name, value in model.state_dict().items():
        assert torch.equal(value, parent['model'][name])
    with pytest.raises(ValueError, match='mismatch'):
        learn.warm_start(model, {**parent, 'tokenizer_hash': 'wrong'}, tokenizer)


def test_continuation_stop_resume_and_dialogue_pipeline(continuation_inputs, monkeypatch):
    tokenizer, report, parent, target = continuation_inputs
    original = learn.ROOT/'runs/real_v1/pretrain/best.pt'
    original_hash = file_hash(original)
    learn.RUN.mkdir(parents=True)
    real_update = train.train_update
    def stop_after_completed_update(*args, **kwargs):
        result = real_update(*args, **kwargs)
        (learn.RUN/'STOP').write_text('stop')
        return result
    monkeypatch.setattr(train, 'train_update', stop_after_completed_update)
    samples = []
    monkeypatch.setattr(learn, 'sample_answers', lambda *args: samples.append(True))
    learn.train_session(1, tokenizer, torch.device('cpu'), report, target)
    paused = json.loads((learn.RUN/'status.json').read_text())
    checkpoint = load_checkpoint(learn.RUN/'pretrain/last.pt')
    assert paused['phase'] == 'paused' and checkpoint['step'] == 1
    assert checkpoint['training_config']['steps'] == 2
    assert checkpoint['scheduler']['last_epoch'] == 1
    assert file_hash(original) == original_hash
    monkeypatch.setattr(train, 'train_update', real_update)
    (learn.RUN/'STOP').unlink()
    learn.train_session(1, tokenizer, torch.device('cpu'), report, target)
    finished = json.loads((learn.RUN/'status.json').read_text())
    assert finished['phase'] == 'finished_unreviewed'
    assert finished['processed_tokens'] == target
    assert finished['selected_checkpoint_tokens'] <= finished['processed_tokens']
    assert load_checkpoint(learn.RUN/'sft/last.pt')['stage'] == 'sft'
    assert samples == [True]
    assert file_hash(original) == original_hash
    with pytest.raises(ValueError, match='different token target'):
        learn.train_session(1, tokenizer, torch.device('cpu'), report, target+16384)


def test_read_only_check_does_not_create_run(continuation_inputs):
    _, _, _, target = continuation_inputs
    plan = learn.check_continuation(9, target)
    assert plan['planned_processed_tokens'] == target
    assert not learn.RUN.exists()
