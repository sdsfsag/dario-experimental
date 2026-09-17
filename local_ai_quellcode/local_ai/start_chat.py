"""Show training progress, then start chat when the training process has finished."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/real_v1'


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def training_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        process = psutil.Process(pid)
        # Do not mistake a reused PID for our training process.
        return process.is_running() and any(
            Path(arg).name.lower() == 'learn.py' for arg in process.cmdline()
        )
    except psutil.AccessDenied:
        # Wait conservatively when Windows cannot expose the command line.
        return psutil.pid_exists(pid)
    except psutil.NoSuchProcess:
        return False


def latest_metrics(path):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 16384))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                value = json.loads(line)
                if 'step' in value:
                    return value
            except (ValueError, UnicodeError):
                continue  # The trainer may still be writing the last line.
    except OSError:
        pass
    return {}


def snapshot(run=RUN):
    status = read_json(run / 'status.json')
    lock = read_json(run / 'running.lock')
    phase = status.get('phase', '')
    active = any(training_alive(pid) for pid in (lock.get('pid'), status.get('pid')))
    checkpoint = run / 'sft/best.pt'
    if active:
        names = {'preparing_data': 'Trainingsdaten werden vorbereitet',
                 'data_ready': 'Training wird vorbereitet',
                 'benchmarking': 'GPU-Geschwindigkeit wird gemessen',
                 'pretraining': 'Sprachtraining', 'sft': 'Dialogtraining',
                 'finished_unreviewed': 'Training wird abgeschlossen',
                 'paused': 'Training wird gespeichert', 'failed': 'Training wird beendet'}
        message = names.get(phase, 'Training laeuft')
        if phase in ('pretraining', 'sft'):
            stage = 'pretrain' if phase == 'pretraining' else 'sft'
            metrics = latest_metrics(run / stage / 'metrics.jsonl')
            step = metrics.get('step', 0)
            total = metrics.get('steps', status.get('planned_steps', 0))
            if total:
                message += f': {step:,} / {total:,} Schritte ({100*step/total:.1f}%)'
            if phase == 'pretraining':
                message += '. Danach folgt das Dialogtraining.'
        if (run / 'STOP').exists():
            message += ' Anhalten wurde angefordert; letzter Schritt wird gespeichert.'
        return 'waiting', message
    if phase == 'finished_unreviewed' and checkpoint.is_file():
        return 'ready', 'Trainingslauf beendet. Der Chat startet; die Sprachqualitaet ist noch ungeprueft.'
    if phase == 'failed':
        return 'blocked', ('Training abgebrochen: ' + str(status.get('error', 'Unbekannter Fehler'))
                           + '\nDetails: local_ai\\runs\\real_v1\\training-error.log')
    if phase == 'paused':
        return 'blocked', ('Das Training ist pausiert und noch nicht fertig. Der Fortschritt ist gespeichert.'
                           '\nZum Fortsetzen Training_starten.cmd doppelt anklicken, danach Chat_starten.cmd.')
    return 'blocked', ('Es laeuft gerade kein Training und es gibt noch kein fertig trainiertes Dialogmodell.'
                       '\nTraining_starten.cmd doppelt anklicken, danach Chat_starten.cmd.')


def wait_for_chat(run=RUN, poll_seconds=15):
    print('Der Chat wartet auf das Ende des Trainings.', flush=True)
    print('Dieses Fenster kann offen bleiben. Strg+C beendet nur das Warten, nicht das Training.', flush=True)
    previous = None
    while True:
        state, message = snapshot(run)
        if message != previous:
            print(f'[{datetime.now():%H:%M:%S}] {message}', flush=True)
            previous = message
        if state == 'ready':
            return run / 'sft/best.pt'
        if state == 'blocked':
            return None
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true', help='Show current status without waiting or starting chat')
    args = parser.parse_args()
    if args.status:
        print(snapshot()[1])
        return 0
    try:
        checkpoint = wait_for_chat()
    except KeyboardInterrupt:
        print('\nWarten beendet. Das Training wurde dadurch nicht gestoppt.', flush=True)
        return 0
    if checkpoint is None:
        return 1
    print('Bei You: schreiben und Enter druecken. Beenden mit /quit.', flush=True)
    return subprocess.call([sys.executable, str(ROOT / 'chat.py'), '--checkpoint', str(checkpoint)], cwd=ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
