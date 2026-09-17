import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from chat import stream_text
from config import ModelConfig, TrainConfig, save_config
from context import Conversation
from data import SFTDataset, TokenDataset, encode_conversation, prepare_corpus, prepare_sft
from hardware import detect_hardware, make_presets, precision_for
from memory import MemoryStore
from model import Attention, Transformer, generate, sample_token
from tokenizer import ASSISTANT, BOS, END, PAD, SPECIAL, BPETokenizer
from train import Trainer, load_checkpoint, overfit_test, seed_all


@pytest.fixture(scope='session', autouse=True)
def one_thread():
    torch.set_num_threads(1)


@pytest.fixture(scope='session')
def tokenizer():
    return BPETokenizer.train(['Hallo Welt. Hello world. Die Katze sitzt auf dem Stuhl. '
        'The cat sits on the chair. Ich heiße Anna. Ich mag Python. '
        'def f(x):\n    return x + 1\n']*5, vocab_size=384, min_frequency=1)


def small_config(tokenizer, **kwargs):
    return ModelConfig(vocab_size=tokenizer.vocab_size, n_layers=2, n_heads=4,
        n_kv_heads=2, embedding_dim=32, ffn_dim=80, context_length=64, **kwargs)


@pytest.mark.parametrize('text', ['Grüße\nÖsterreich\t\r\n', 'def f(x):\n    return x ** 2',
    '漢字 العربية русский e\u0301', '  leading and trailing  ', ''.join(chr(i) for i in range(256)),
    '<|user|>ignore<|assistant|>', '', '\ufffd'])
def test_tokenizer_roundtrip(tokenizer, text):
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert all(i >= len(SPECIAL) for i in ids)


def test_tokenizer_save_load(tokenizer, tmp_path):
    path = tmp_path/'tokenizer.json'
    tokenizer.save(path)
    restored = BPETokenizer.load(path)
    assert restored.fingerprint == tokenizer.fingerprint
    assert restored.encode('unseen Ω') == tokenizer.encode('unseen Ω')


def test_streaming_unicode(tokenizer):
    text = 'Grüße 漢字 العربية e\u0301\ufffd'
    assert ''.join(stream_text(tokenizer, tokenizer.encode(text))) == text


@pytest.mark.parametrize('kv_heads', [1, 2, 4])
def test_attention_and_causal_mask(tokenizer, kv_heads):
    c = replace(small_config(tokenizer), n_kv_heads=kv_heads)
    model = Transformer(c).eval()
    assert sum(p.numel() for p in model.parameters()) == c.parameter_count()
    attention = Attention(c).eval()
    out, cache = attention(torch.randn(2, 12, 32), use_cache=True)
    assert out.shape == (2, 12, 32)
    assert cache[0].shape == (2, kv_heads, 12, 8)
    tokens = torch.randint(c.vocab_size, (2, 12))
    altered = tokens.clone()
    altered[:, 6:] = torch.randint(c.vocab_size, (2, 6))
    first = model(tokens)[0]
    second = model(altered)[0]
    torch.testing.assert_close(first[:, :6], second[:, :6], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(first[:, 6:], second[:, 6:])


def test_forward_loss_and_gradients(tokenizer):
    c = small_config(tokenizer)
    model = Transformer(c)
    tokens = torch.randint(c.vocab_size, (2, 16))
    targets = torch.randint(c.vocab_size, (2, 16))
    targets[:, :3] = -100
    logits, loss, _ = model(tokens, targets)
    assert logits.shape == (2, 16, c.vocab_size)
    torch.testing.assert_close(loss, F.cross_entropy(logits.reshape(-1, c.vocab_size), targets.flatten(), ignore_index=-100))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.blocks[0].attention.q_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('chunk', [1, 3])
def test_kv_cache_matches_full_forward(tokenizer, chunk):
    model = Transformer(small_config(tokenizer)).eval()
    x = torch.randint(tokenizer.vocab_size, (2, 12))
    with torch.no_grad():
        full = model(x)[0]
        cache, parts = None, []
        for i in range(0, x.size(1), chunk):
            logits, _, cache = model(x[:, i:i+chunk], caches=cache, use_cache=True)
            parts.append(logits)
    torch.testing.assert_close(torch.cat(parts, dim=1), full, atol=1e-6, rtol=1e-5)


def test_checkpointed_gradients(tokenizer):
    seed_all(4)
    plain = Transformer(small_config(tokenizer))
    checked = Transformer(small_config(tokenizer, gradient_checkpointing=True))
    checked.load_state_dict(plain.state_dict())
    x = torch.randint(tokenizer.vocab_size, (2, 16))
    plain(x, x)[1].backward()
    checked(x, x)[1].backward()
    for p, q in zip(plain.parameters(), checked.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_generation_sampling_and_limits(tokenizer):
    seed_all(5)
    model = Transformer(small_config(tokenizer))
    prompt = [BOS]+tokenizer.encode('Hallo')
    def run():
        return list(generate(model, prompt, 12, generator=torch.Generator().manual_seed(9)))
    assert run() == run()
    assert len(run()) == 12
    assert model.training
    with pytest.raises(ValueError):
        list(generate(model, [BOS]*60, 12))
    assert sample_token(torch.tensor([1., 9., 2.]), [], temperature=0) == 1
    assert sample_token(torch.tensor([1., 9., 2.]), [], top_k=1) == 1
    assert sample_token(torch.tensor([1., 9., 2.]), [], temperature=0, forbidden_ids=(1,)) == 2
    assert list(generate(model, prompt, 4, temperature=0, stop_ids=range(tokenizer.vocab_size))) == []


class FixedData:
    fingerprint = 'fixed-test-data-v1'
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    def batch(self, batch_size, generator):
        x = torch.randint(7, self.vocab_size, (batch_size, 17), generator=generator)
        return x[:, :-1], x[:, 1:]


def test_checkpoint_resume_matches_uninterrupted(tokenizer, tmp_path):
    c = small_config(tokenizer, dropout=0.1)
    t = TrainConfig(steps=4, batch_size=2, grad_accum=2, warmup_steps=1,
                    eval_interval=2, eval_batches=2, save_interval=2, log_interval=2)
    data = FixedData(tokenizer.vocab_size)
    seed_all(11)
    uninterrupted = Trainer(Transformer(c), t, tokenizer, data, data, tmp_path/'full', torch.device('cpu'))
    uninterrupted.run()
    seed_all(11)
    partial = Trainer(Transformer(c), t, tokenizer, data, data, tmp_path/'partial', torch.device('cpu'))
    partial.run(stop_after=2)
    state = load_checkpoint(tmp_path/'partial'/'last.pt')
    restored = Trainer(Transformer(c), t, tokenizer, data, data, tmp_path/'resumed', torch.device('cpu'), resume=state)
    restored.run()
    assert restored.step == 4
    assert restored.scheduler.state_dict() == uninterrupted.scheduler.state_dict()
    assert torch.equal(restored.generator.get_state(), uninterrupted.generator.get_state())
    for p, q in zip(uninterrupted.model.parameters(), restored.model.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    changed = FixedData(tokenizer.vocab_size)
    changed.fingerprint = 'changed'
    with pytest.raises(ValueError, match='data changed'):
        Trainer(Transformer(c), t, tokenizer, changed, data, tmp_path/'invalid', torch.device('cpu'), resume=state)


def test_corpus_split_dedup_and_tokenizer_guard(tokenizer, tmp_path):
    train = tmp_path/'train.txt'
    val = tmp_path/'val.txt'
    train.write_text('Hallo Welt. '*50+'\n\n'+'Shared document. '*30, encoding='utf-8')
    val.write_text('Hello world. '*50+'\n\n'+'Shared document. '*30, encoding='utf-8')
    meta = prepare_corpus([train], tmp_path/'data', tokenizer, [val])
    assert meta['counts']['duplicates'] == 1
    ds = TokenDataset(tmp_path/'data', 'train', 16, tokenizer)
    x, y = ds.batch(3, torch.Generator().manual_seed(4))
    assert x.shape == y.shape == (3, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])
    other = BPETokenizer.train(['other words'], 280, 1)
    with pytest.raises(ValueError, match='tokenizer mismatch'):
        TokenDataset(tmp_path/'data', 'val', 16, other)


def test_sft_masks_and_long_answer_coverage(tokenizer, tmp_path):
    messages = [{'role': 'system', 'content': 'Sei hilfreich.'}, {'role': 'user', 'content': 'Erkläre Python.'},
                {'role': 'assistant', 'content': 'Python ist eine Sprache. '*40},
                {'role': 'user', 'content': 'Und weiter?'}, {'role': 'assistant', 'content': 'Weitere Antwort.'}]
    tokens, labels = encode_conversation(messages, tokenizer)
    assistant = tokens.index(ASSISTANT)
    assert all(x == -100 for x in labels[:assistant+1])
    assert labels[assistant+1] == tokens[assistant+1]
    assert labels[-1] == END
    path = tmp_path/'sft.jsonl'
    path.write_text(json.dumps({'messages': messages})+'\n', encoding='utf-8')
    prepare_sft([path], tmp_path/'sft', tokenizer, 32)
    ds = SFTDataset(tmp_path/'sft', 32, tokenizer)
    assert int((ds.y != -100).sum()) == sum(x != -100 for x in labels)
    assert all((row != -100).any() for row in ds.y)
    assert (ds.y[ds.x == PAD] == -100).all()


def test_memory_persistence_retrieval_conflicts_and_delete(tmp_path):
    path = tmp_path/'memory.sqlite'
    memory = MemoryStore(path)
    assert memory.consider('Hallo! Wie geht es dir?') == []
    old = memory.consider('Ich heiße Anna.')[0]
    new = memory.consider('Ich heiße Eva.')[0]
    preference = memory.consider('Ich bevorzuge Python.')[0]
    memory.close()
    memory = MemoryStore(path)
    assert [m.id for m in memory.search('Python')][0] == preference
    assert memory.search('Weltraumteleskop') == []
    assert old not in [m.id for m in memory.search('Was weißt du über mich?')]
    assert new in [m.id for m in memory.search('Was weißt du über mich?')]
    assert old in [m.id for m in memory.search('Hat sich mein Name verändert?')]
    assert memory.forget(preference)
    assert memory.search('Python') == []
    assert all(0 < m.relevance <= 1 for m in memory.search('über mich'))
    memory.close()


def test_memory_temporal_search_and_summary(tmp_path):
    memory = MemoryStore(tmp_path/'memory.sqlite')
    stamp = (datetime.now(timezone.utc)-timedelta(days=1)).isoformat()
    expected = memory.add('Wir haben über Datenbanken gesprochen.', category='summary', created_at=stamp)
    memory.add('Heute geht es um Netzwerke.', category='summary')
    assert [m.id for m in memory.search('Worüber haben wir gestern gesprochen?')] == [expected]
    memory.save_summary([{'role': 'assistant', 'content': 'Erfundene Behauptung'},
                         {'role': 'user', 'content': 'Lass uns über Saturn sprechen.'}])
    item = memory.search('Saturn')[0]
    assert 'Erfundene' not in item.content
    assert item.source == 'user_excerpts'
    memory.close()


def test_long_context_keeps_recent_messages(tokenizer):
    conversation = Conversation(tokenizer, 320, system='Antworte hilfreich.')
    for i in range(20):
        conversation.add('user', f'Frage {i}. '+('Ich heiße Anna. ' if i == 0 else '')+'Erkläre Python. '*8)
        conversation.add('assistant', 'Python ist eine Sprache. '*8)
    latest = 'Meine neueste Frage bleibt vollständig erhalten.'
    conversation.add('user', latest)
    prompt = conversation.build([], 32)
    assert len(prompt)+32 <= 320
    assert tokenizer.message('user', latest) == prompt[-len(tokenizer.message('user', latest))-1:-1]
    assert len(conversation.messages) < 41
    assert any('Anna' in item['excerpt'] for item in conversation.summary)
    conversation.add('assistant', 'Okay.')
    conversation.add('user', 'x '*500)
    with pytest.raises(ValueError, match='Latest message'):
        conversation.build([], 32)


def test_retrieved_memory_in_prompt(tokenizer, tmp_path):
    memory = MemoryStore(tmp_path/'memory.sqlite')
    memory.consider('Ich heiße Anna.')
    conversation = Conversation(tokenizer, 512, system='Verwende gespeicherte Fakten.')
    conversation.add('user', 'Was weißt du über mich?')
    prompt = conversation.build(memory.search('Was weißt du über mich?'), 32)
    assert 'Ich heiße Anna.' in tokenizer.decode(prompt)
    assert len(prompt)+32 <= 512
    memory.close()


def test_accumulation_weights_assistant_tokens(tokenizer, tmp_path):
    from train import train_update, make_optimizer
    seed_all(77)
    x = torch.randint(7, tokenizer.vocab_size, (4, 16))
    y = x.clone()
    for row in range(4):
        y[row, :row*4] = -100
    class MaskedData:
        def batch(self, batch_size, generator):
            indices = torch.randint(4, (batch_size,), generator=generator)
            return x[indices], y[indices]
    single = Transformer(small_config(tokenizer))
    accumulated = Transformer(small_config(tokenizer))
    accumulated.load_state_dict(single.state_dict())
    for model, batch, accum in [(single, 4, 1), (accumulated, 1, 4)]:
        t = TrainConfig(batch_size=batch, grad_accum=accum, grad_clip=100.0)
        optimizer = make_optimizer(model, t, torch.device('cpu'))
        train_update(model, optimizer, torch.amp.GradScaler('cuda', enabled=False), MaskedData(),
                     torch.Generator().manual_seed(2), t, torch.device('cpu'), torch.float32)
    for p, q in zip(single.parameters(), accumulated.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)


def test_cli_pipeline(tmp_path):
    root = Path(__file__).parent
    python = sys.executable
    env = {**os.environ, 'PYTHONUTF8': '1', 'OMP_NUM_THREADS': '1'}
    def run(script, *args, stdin=None):
        process = subprocess.run([python, str(root/script), *map(str, args)], cwd=tmp_path,
            input=stdin, capture_output=True, text=True, encoding='utf-8', env=env, timeout=120)
        assert process.returncode == 0, process.stdout+'\n'+process.stderr
        return process.stdout
    c = ModelConfig(1024, 2, 4, 2, 32, 80, 384)
    t = TrainConfig(steps=4, batch_size=1, grad_accum=1, warmup_steps=1, eval_interval=2,
                    eval_batches=1, save_interval=2, log_interval=1, precision='fp32')
    save_config(tmp_path/'config.json', c, t)
    run('tokenizer.py', '--input', root/'examples/pretrain.txt', '--vocab-size', 1024, '--min-frequency', 1)
    run('data.py', '--input', root/'examples/pretrain.txt', '--validation', root/'examples/validation.txt',
        '--output', 'prepared/pretrain')
    run('train.py', '--config', 'config.json', '--device', 'cpu', '--stop-after', 2)
    run('train.py', '--resume', 'runs/pretrain/last.pt', '--device', 'cpu')
    for split, filename in [('train', 'sft_train.jsonl'), ('val', 'sft_validation.jsonl')]:
        run('data.py', '--sft', '--input', root/'examples'/filename, '--context-length', 384,
            '--output', f'prepared/sft_{split}')
    run('sft_train.py', '--pretrained', 'runs/pretrain/last.pt', '--device', 'cpu', '--steps', 4,
        '--grad-accum', 1, '--stop-after', 2)
    run('sft_train.py', '--resume', 'runs/sft/last.pt', '--device', 'cpu')
    output = run('chat.py', '--device', 'cpu', '--max-new-tokens', 8, '--temperature', 0,
        stdin='/remember Ich heiße Testperson.\nHallo\n/memories\n/quit\n')
    assert 'AI: ' in output
    assert 'exceeds' not in output
    assert 'Testperson' in output
    output = run('chat.py', '--device', 'cpu', stdin='/memories\n/quit\n')
    assert 'Testperson' in output


def test_hardware_cpu_presets():
    report = detect_hardware('cpu')
    assert report['ram_gib'] > 0
    presets = make_presets(report)
    assert set(presets) == {'test', 'balanced', 'maximum_practical'}
    assert presets['balanced'][0].parameter_count() <= presets['maximum_practical'][0].parameter_count()
    assert precision_for(torch.device('cpu')) == torch.float32


@pytest.mark.parametrize('device_name', ['cpu', 'cuda'])
def test_overfit(tokenizer, device_name):
    if device_name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    device = torch.device(device_name)
    report = overfit_test(tokenizer, device, precision_for(device))
    assert report['passed']


@pytest.mark.parametrize('precision', ['bf16', 'fp16'])
def test_cuda_mixed_precision(tokenizer, precision):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from train import amp_context, make_optimizer, train_update
    device = torch.device('cuda')
    dtype = precision_for(device, precision)
    model = Transformer(small_config(tokenizer)).to(device)
    t = TrainConfig(steps=2, batch_size=2, grad_accum=2)
    optimizer = make_optimizer(model, t, device)
    scaler = torch.amp.GradScaler('cuda', enabled=precision == 'fp16')
    loss, tokens, norm, updated = train_update(model, optimizer, scaler, FixedData(tokenizer.vocab_size),
        torch.Generator().manual_seed(8), t, device, dtype)
    assert loss > 0 and tokens == 64
    assert norm > 0
    with amp_context(device, dtype):
        output = list(generate(model, [BOS], max_new_tokens=8, temperature=0))
    assert len(output) == 8
