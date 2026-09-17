import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import ModelConfig
from context import Conversation
from hardware import precision_for, select_device
from memory import MemoryStore
from model import Transformer, generate
from tokenizer import BPETokenizer, BOS, PAD, EOS, END, SYSTEM, USER, ASSISTANT
from train import amp_context, load_checkpoint, seed_all


def safe_terminal(text):
    return ''.join(c for c in text if c in '\n\t' or (c.isprintable() and ord(c) != 127))


def stream_text(tokenizer, tokens):
    received, emitted = [], ''
    for token in tokens:
        received.append(token)
        text = tokenizer.decode(received).rstrip('\ufffd')
        if text.startswith(emitted):
            yield text[len(emitted):]
            emitted = text
    text = tokenizer.decode(received)
    if text.startswith(emitted):
        yield text[len(emitted):]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint')
    p.add_argument('--memory', default='artifacts/memory.sqlite')
    p.add_argument('--device', default='auto')
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top-k', type=int, default=40)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--repetition-penalty', type=float, default=1.1)
    p.add_argument('--max-new-tokens', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-auto-memory', action='store_true')
    a = p.parse_args()
    if a.checkpoint is None:
        trained = Path('runs/real_v1/sft/best.pt')
        a.checkpoint = str(trained) if trained.is_file() else 'runs/sft/best.pt'
    if a.temperature < 0 or a.top_k < 0 or not 0 < a.top_p <= 1 or a.repetition_penalty < 1 or (a.max_new_tokens is not None and a.max_new_tokens < 1):
        p.error('Invalid generation settings')
    if not Path(a.checkpoint).is_file():
        p.error(f'Checkpoint missing: {a.checkpoint}. Run pretraining and SFT first.')
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    seed_all(a.seed)
    device = select_device(a.device)
    dtype = precision_for(device)
    payload = load_checkpoint(a.checkpoint)
    tokenizer = BPETokenizer(Tokenizer.from_str(payload['tokenizer_json']))
    if tokenizer.fingerprint != payload['tokenizer_hash']:
        raise ValueError('Checkpoint tokenizer mismatch')
    model = Transformer(ModelConfig(**payload['model_config'])).to(device)
    model.load_state_dict(payload['model'])
    if a.max_new_tokens is None:
        a.max_new_tokens = min(256, model.config.context_length//4)
    model.eval()
    stage = payload['stage']
    del payload
    generator = torch.Generator(device=device).manual_seed(a.seed)
    conversation = Conversation(tokenizer, model.config.context_length)
    memory = MemoryStore(a.memory)
    print(f'Loaded {stage} checkpoint. /help for commands.')
    try:
        while True:
            try:
                text = input('You: ').strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text in ('/quit', '/exit'):
                break
            if text == '/help':
                print('/remember TEXT | /memories | /forget ID | /summary | /reset | /quit')
                continue
            if text == '/reset':
                conversation.clear()
                print('Session context cleared.')
                continue
            if text.startswith('/remember '):
                print(f'Memory saved: {memory.add(text[10:], category="explicit")}')
                continue
            if text.startswith('/forget '):
                try:
                    print(f'Deleted: {memory.forget(int(text[8:]))}')
                except ValueError:
                    print('Use /forget ID')
                continue
            if text == '/memories':
                for item in memory.list():
                    print(safe_terminal(json.dumps(asdict(item), ensure_ascii=False)))
                continue
            if text == '/summary':
                source = [{'role': s['role'], 'content': s['excerpt']} for s in conversation.summary]+conversation.messages
                print(f'Summary saved: {memory.save_summary(source)}')
                continue
            conversation.add('user', text)
            memories = memory.search(text)
            try:
                prompt = conversation.build(memories, a.max_new_tokens)
            except ValueError as error:
                conversation.messages.pop()
                print(str(error))
                continue
            if not a.no_auto_memory:
                saved = memory.consider(text)
                if saved:
                    print('Memory saved: '+', '.join(map(str, saved)))
            print('AI: ', end='', flush=True)
            answer = ''
            try:
                with amp_context(device, dtype):
                    tokens = generate(model, prompt, a.max_new_tokens, stop_ids=(EOS, END),
                        generator=generator, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p,
                        repetition_penalty=a.repetition_penalty, forbidden_ids=(PAD, BOS, SYSTEM, USER, ASSISTANT))
                    for piece in stream_text(tokenizer, tokens):
                        answer += piece
                        print(safe_terminal(piece), end='', flush=True)
            except KeyboardInterrupt:
                pass
            print()
            conversation.add('assistant', answer)
    finally:
        memory.close()


if __name__ == '__main__':
    main()
