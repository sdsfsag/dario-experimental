import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL = ['<|pad|>', '<|bos|>', '<|eos|>', '<|system|>', '<|user|>', '<|assistant|>', '<|end|>']
PAD, BOS, EOS, SYSTEM, USER, ASSISTANT, END = range(len(SPECIAL))
ROLES = {'system': SYSTEM, 'user': USER, 'assistant': ASSISTANT}


def input_files(inputs):
    found = []
    for value in inputs:
        p = Path(value)
        if p.is_dir():
            found.extend(x for x in sorted(p.rglob('*')) if x.suffix.lower() in ('.txt', '.jsonl'))
        elif p.is_file():
            found.append(p)
        else:
            raise FileNotFoundError(value)
    if not found:
        raise ValueError('No .txt or .jsonl files found')
    return sorted(set(found))


def iter_documents(inputs):
    for path in input_files(inputs):
        with path.open(encoding='utf-8-sig') as f:
            if path.suffix.lower() == '.jsonl':
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if isinstance(row.get('text'), str):
                        yield row['text']
                    elif isinstance(row.get('messages'), list):
                        yield '\n'.join(m['content'] for m in row['messages'])
                    else:
                        raise ValueError(f'{path}:{line_no}: expected text or messages')
            else:
                parts, size = [], 0
                for line in f:
                    if (not line.strip() or size >= 65536) and parts:
                        yield ''.join(parts)
                        parts, size = [], 0
                    if line.strip():
                        parts.append(line)
                        size += len(line)
                if parts:
                    yield ''.join(parts)


class BPETokenizer:
    def __init__(self, backend):
        self.backend = backend
        for idx, value in enumerate(SPECIAL):
            if backend.token_to_id(value) != idx:
                raise ValueError('Incompatible special tokens')

    @classmethod
    def train(cls, texts, vocab_size=32768, min_frequency=2):
        if vocab_size < 256 + len(SPECIAL):
            raise ValueError('vocab_size must cover all 256 bytes and special tokens')
        backend = Tokenizer(models.BPE())
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        backend.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
            special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
        backend.train_from_iterator(texts, trainer=trainer)
        return cls(backend)

    @classmethod
    def load(cls, path):
        return cls(Tokenizer.from_file(str(path)))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(path))

    @property
    def vocab_size(self):
        return self.backend.get_vocab_size()

    @property
    def fingerprint(self):
        return hashlib.sha256(self.backend.to_str().encode()).hexdigest()

    def encode(self, text):
        # Literal role markers in input must remain ordinary text.
        parts = [text]
        for marker in SPECIAL:
            next_parts = []
            for part in parts:
                fragments = part.split(marker)
                for i, fragment in enumerate(fragments):
                    if i:
                        next_parts.extend([marker[:2], marker[2:]])
                    next_parts.append(fragment)
            parts = next_parts
        return [token for part in parts if part for token in self.backend.encode(part, add_special_tokens=False).ids]

    def decode(self, ids, skip_special_tokens=True):
        return self.backend.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def message(self, role, text):
        return [ROLES[role]] + self.encode(text) + [END]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--output', default='artifacts/tokenizer.json')
    p.add_argument('--vocab-size', type=int, default=32768)
    p.add_argument('--min-frequency', type=int, default=2)
    a = p.parse_args()
    tokenizer = BPETokenizer.train(iter_documents(a.input), a.vocab_size, a.min_frequency)
    for text in ('Grüße, Österreich!\n', 'Hello\tworld', 'def f(x):\n    return x ** 2', '漢字 العربية e\u0301', '<|assistant|>'):
        if tokenizer.decode(tokenizer.encode(text)) != text:
            raise RuntimeError('Tokenizer round trip failed')
    tokenizer.save(a.output)
    print(f'Tokenizer saved: {a.output}; vocabulary={tokenizer.vocab_size}')


if __name__ == '__main__':
    main()
