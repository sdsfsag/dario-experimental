import argparse
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import torch

from tokenizer import BPETokenizer, BOS, EOS, END, ASSISTANT, SPECIAL, iter_documents, input_files


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_corpus(inputs, output, tokenizer, validation=None, val_fraction=0.05, seed=42):
    if not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be between zero and one')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ('meta.json', 'train.bin', 'val.bin')):
        raise FileExistsError('Use a new data directory to avoid overwriting a prepared corpus')
    db = sqlite3.connect(out/'dedup.tmp.sqlite')
    db.executescript('DROP TABLE IF EXISTS seen; CREATE TABLE seen (hash TEXT PRIMARY KEY);')
    counts = {'train': 0, 'val': 0, 'duplicates': 0, 'documents': 0}
    handles = {name: (out/f'{name}.bin.tmp').open('wb') for name in ('train', 'val')}
    try:
        sources = [('val', validation), ('train', inputs)] if validation else [(None, inputs)]
        for forced_split, paths in sources:
            for text in iter_documents(paths):
                text = text.replace('\r\n', '\n')
                if not text.strip():
                    continue
                digest = hashlib.sha256(text.strip().encode('utf-8')).hexdigest()
                if db.execute('INSERT OR IGNORE INTO seen VALUES (?)', (digest,)).rowcount == 0:
                    counts['duplicates'] += 1
                    continue
                score = int(hashlib.sha256(f'{seed}:{digest}'.encode()).hexdigest()[:16], 16) / 2**64
                split = forced_split or ('val' if score < val_fraction else 'train')
                ids = [BOS] + tokenizer.encode(text) + [EOS]
                handles[split].write(np.asarray(ids, dtype='<u4').tobytes())
                counts[split] += len(ids)
                counts['documents'] += 1
                if counts['documents'] % 10000 == 0:
                    db.commit()
                    print(f'Prepared documents={counts["documents"]}', flush=True)
        if min(counts['train'], counts['val']) < 2:
            raise ValueError('Insufficient train/validation data; supply separate --validation files')
    finally:
        for f in handles.values():
            f.close()
        db.close()
        (out/'dedup.tmp.sqlite').unlink(missing_ok=True)
    for split in ('train', 'val'):
        os.replace(out/f'{split}.bin.tmp', out/f'{split}.bin')
    meta = {'version': 1, 'dtype': '<u4', 'tokenizer_hash': tokenizer.fingerprint,
            'vocab_size': tokenizer.vocab_size, 'seed': seed, 'counts': counts,
            'hashes': {name: file_hash(out/f'{name}.bin') for name in ('train', 'val')}}
    (out/'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


class TokenDataset:
    def __init__(self, folder, split, context_length, tokenizer):
        folder = Path(folder)
        meta = json.loads((folder/'meta.json').read_text(encoding='utf-8'))
        if meta['tokenizer_hash'] != tokenizer.fingerprint:
            raise ValueError('Corpus tokenizer mismatch')
        path = folder/f'{split}.bin'
        if path.stat().st_size % 4 or file_hash(path) != meta['hashes'][split]:
            raise ValueError('Corrupt token data')
        self.tokens = np.memmap(path, mode='r', dtype='<u4')
        self.context_length = context_length
        if len(self.tokens) < context_length+1:
            raise ValueError(f'{split} needs at least {context_length+1} tokens')
        self.fingerprint = meta['hashes'][split]

    def batch(self, batch_size, generator):
        starts = torch.randint(len(self.tokens)-self.context_length, (batch_size,), generator=generator).tolist()
        rows = np.stack([self.tokens[s:s+self.context_length+1] for s in starts]).astype(np.int64)
        tokens = torch.from_numpy(rows)
        return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a nonempty list')
    expected = 'user'
    for i, message in enumerate(messages):
        role, content = message.get('role'), message.get('content')
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Each message needs nonempty string content')
        if role == 'system' and i == 0:
            continue
        if role != expected:
            raise ValueError('Expected optional system, then alternating user/assistant messages')
        expected = 'assistant' if role == 'user' else 'user'
    if messages[-1]['role'] != 'assistant':
        raise ValueError('SFT conversations must end with assistant')


def encode_conversation(messages, tokenizer):
    validate_messages(messages)
    tokens, labels = [BOS], [-100]
    for message in messages:
        ids = tokenizer.message(message['role'], message['content'])
        tokens.extend(ids)
        labels.extend(([-100] + ids[1:]) if message['role'] == 'assistant' else [-100]*len(ids))
    return tokens, labels


def prepare_sft(inputs, output, tokenizer, context_length):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if any((out/name).exists() for name in ('meta.json', 'tokens.bin', 'labels.bin')):
        raise FileExistsError('Use a new SFT data directory')
    rows, conversations = 0, 0
    hashes = set()
    with (out/'tokens.bin.tmp').open('wb') as xf, (out/'labels.bin.tmp').open('wb') as yf:
        for path in input_files(inputs):
            if path.suffix.lower() != '.jsonl':
                raise ValueError('SFT expects JSONL with messages')
            with path.open(encoding='utf-8-sig') as f:
                for line in f:
                    if not line.strip():
                        continue
                    messages = json.loads(line)['messages']
                    tokens, labels = encode_conversation(messages, tokenizer)
                    digest = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                    if digest in hashes:
                        continue
                    hashes.add(digest)
                    conversations += 1
                    # Overlap provides context without training the same target twice.
                    stride = max(1, context_length//2)
                    trained_until = 1
                    for start in range(0, len(tokens)-1, stride):
                        end = min(len(tokens), start+context_length+1)
                        x, y = tokens[start:end-1], labels[start+1:end].copy()
                        for i in range(min(len(y), max(0, trained_until-start-1))):
                            y[i] = -100
                        trained_until = max(trained_until, end)
                        if any(target != -100 for target in y):
                            x += [0]*(context_length-len(x))
                            y += [-100]*(context_length-len(y))
                            xf.write(np.asarray(x, dtype='<i4').tobytes())
                            yf.write(np.asarray(y, dtype='<i4').tobytes())
                            rows += 1
                        if end == len(tokens):
                            break
    if not rows:
        raise ValueError('No trainable assistant targets')
    for name in ('tokens', 'labels'):
        os.replace(out/f'{name}.bin.tmp', out/f'{name}.bin')
    meta = {'version': 1, 'context_length': context_length, 'rows': rows,
            'tokenizer_hash': tokenizer.fingerprint, 'conversation_hashes': sorted(hashes),
            'hashes': {name: file_hash(out/f'{name}.bin') for name in ('tokens', 'labels')}}
    (out/'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return meta


class SFTDataset:
    def __init__(self, folder, context_length, tokenizer):
        folder = Path(folder)
        self.meta = json.loads((folder/'meta.json').read_text(encoding='utf-8'))
        if self.meta['context_length'] != context_length or self.meta['tokenizer_hash'] != tokenizer.fingerprint:
            raise ValueError('SFT data/config/tokenizer mismatch; prepare again in a new directory')
        arrays = {}
        for name in ('tokens', 'labels'):
            if file_hash(folder/f'{name}.bin') != self.meta['hashes'][name]:
                raise ValueError('Corrupt SFT data')
            arrays[name] = np.memmap(folder/f'{name}.bin', mode='r', dtype='<i4').reshape(-1, context_length)
        self.x, self.y = arrays['tokens'], arrays['labels']
        self.fingerprint = hashlib.sha256(json.dumps(self.meta, sort_keys=True).encode()).hexdigest()

    def batch(self, batch_size, generator):
        idx = torch.randint(len(self.x), (batch_size,), generator=generator).numpy()
        return torch.from_numpy(self.x[idx].astype(np.int64)), torch.from_numpy(self.y[idx].astype(np.int64))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--validation', nargs='+')
    p.add_argument('--output', required=True)
    p.add_argument('--tokenizer', default='artifacts/tokenizer.json')
    p.add_argument('--val-fraction', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sft', action='store_true')
    p.add_argument('--context-length', type=int, default=1024)
    a = p.parse_args()
    tokenizer = BPETokenizer.load(a.tokenizer)
    if a.sft:
        result = prepare_sft(a.input, a.output, tokenizer, a.context_length)
    else:
        result = prepare_corpus(a.input, a.output, tokenizer, a.validation, a.val_fraction, a.seed)
    print(json.dumps({k: v for k, v in result.items() if k != 'conversation_hashes'}, indent=2))


if __name__ == '__main__':
    main()
