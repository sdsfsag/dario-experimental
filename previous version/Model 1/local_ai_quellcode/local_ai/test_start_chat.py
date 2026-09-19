import json

import pytest

import start_chat


def state(run, phase):
    (run / 'status.json').write_text(json.dumps({'phase': phase, 'pid': 12345}))


def test_wait_until_trainer_exits_before_using_checkpoint(tmp_path, monkeypatch):
    (tmp_path / 'sft').mkdir()
    (tmp_path / 'sft/best.pt').write_bytes(b'checkpoint')
    alive = [True]
    monkeypatch.setattr(start_chat, 'training_alive', lambda pid: bool(pid) and alive[0])
    state(tmp_path, 'sft')
    assert start_chat.snapshot(tmp_path)[0] == 'waiting'
    state(tmp_path, 'finished_unreviewed')
    assert start_chat.snapshot(tmp_path)[0] == 'waiting'
    alive[0] = False
    assert start_chat.snapshot(tmp_path)[0] == 'ready'


def test_dead_or_paused_training_does_not_wait_forever(tmp_path, monkeypatch):
    monkeypatch.setattr(start_chat, 'training_alive', lambda pid: False)
    for phase in ('pretraining', 'paused', 'failed', 'finished_unreviewed'):
        state(tmp_path, phase)
        # Even a completion status requires an actual checkpoint.
        assert start_chat.wait_for_chat(tmp_path, poll_seconds=0) is None


def test_progress_survives_partially_written_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(start_chat, 'training_alive', lambda pid: bool(pid))
    state(tmp_path, 'pretraining')
    (tmp_path / 'pretrain').mkdir()
    (tmp_path / 'pretrain/metrics.jsonl').write_text(
        '{"step": 100, "steps": 1000}\n{"step": 101,')
    status, message = start_chat.snapshot(tmp_path)
    assert status == 'waiting'
    assert '10.0%' in message
    assert 'Danach folgt das Dialogtraining' in message


def test_wait_loop_can_finish_or_be_cancelled_without_starting_training(tmp_path, monkeypatch):
    states = iter([('waiting', 'Training'), ('ready', 'Fertig')])
    monkeypatch.setattr(start_chat, 'snapshot', lambda run: next(states))
    monkeypatch.setattr(start_chat.time, 'sleep', lambda seconds: None)
    assert start_chat.wait_for_chat(tmp_path) == tmp_path / 'sft/best.pt'
    monkeypatch.setattr(start_chat, 'snapshot', lambda run: ('waiting', 'Training'))
    def cancel(seconds):
        raise KeyboardInterrupt
    monkeypatch.setattr(start_chat.time, 'sleep', cancel)
    with pytest.raises(KeyboardInterrupt):
        start_chat.wait_for_chat(tmp_path)
