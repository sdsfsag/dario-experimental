"""Prepare a separate German language/dialogue curriculum from attributed local sources."""
import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from context import SYSTEM_PROMPT
from data import SFTDataset, TokenDataset, encode_conversation, file_hash, prepare_corpus, prepare_sft
from download_data import clean_text, normalized, write_row
from pilot_data import build_data
from tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent
GROUPS = ('dialog', 'reading', 'knowledge', 'stories', 'context', 'basics')


def partition(text):
    n = int(hashlib.sha256(normalized(text).encode()).hexdigest()[:8], 16) % 100
    return 'test' if n == 0 else 'validation' if n < 3 else 'train'


def acceptable(text, minimum=20):
    if len(text) < minimum or '\ufffd' in text or '<|' in text or 'http' in text:
        return False
    words = re.findall(r'\w+', text.casefold())
    if len(words) > 35:
        trigrams = list(zip(words, words[1:], words[2:]))
        if Counter(trigrams).most_common(1)[0][1] > max(4, len(trigrams)*.06):
            return False
    return True


def dialog(question, answer, source, key, history=None):
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT + f' Datum: {date.today().isoformat()}.'}]
    messages += history or []
    messages += [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]
    return {'messages': messages, 'source': source, 'source_id': str(key), 'lang': 'de'}


class Mixture:
    def __init__(self, datasets, weights, trim=False):
        self.datasets = datasets
        self.weights = torch.tensor(weights, dtype=torch.double)
        if not datasets or len(weights) != len(datasets) or min(weights) <= 0:
            raise ValueError('Invalid mixture')
        self.trim = trim
        self.fingerprint = hashlib.sha256(json.dumps([d.fingerprint for d in datasets]+
                                                     [list(weights), trim]).encode()).hexdigest()

    def batch(self, batch_size, generator):
        choices = torch.multinomial(self.weights, batch_size, replacement=True, generator=generator)
        batches = [self.datasets[i].batch(int((choices == i).sum()), generator)
                   for i in range(len(self.datasets)) if (choices == i).any()]
        x, y = (torch.cat([b[i] for b in batches]) for i in (0, 1))
        if self.trim:
            width = min(x.shape[1], ((int((x != 0).sum(1).max())+15)//16)*16)
            x, y = x[:, :width].contiguous(), y[:, :width].contiguous()
        return x, y


def datasets(folder, tokenizer, stage):
    folder = Path(folder)
    if stage == 'pretrain':
        locations = [folder/'stories', folder/'knowledge', ROOT/'prepared/real_v1/pretrain']
        return tuple(Mixture([TokenDataset(p, split, 512, tokenizer) for p in locations], [.70, .05, .25])
                     for split in ('train', 'val'))
    return tuple(Mixture([SFTDataset(folder/f'{group}_{split}', 1024, tokenizer) for group in GROUPS],
                         [.35, .20, .15, .10, .18, .02], trim=True)
                 for split in ('train', 'validation'))


def prepare(sources_file, output, prepared, tokenizer):
    output, prepared = Path(output), Path(prepared)
    if (output/'manifest.json').exists():
        manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        if manifest['tokenizer_hash'] != tokenizer.fingerprint:
            raise ValueError('Tokenizer changed')
        for rel, digest in manifest['files'].items():
            if file_hash(output/rel) != digest:
                raise ValueError(f'Changed curriculum file: {rel}')
    else:
        if output.exists():
            raise FileExistsError('Incomplete data directory; inspect it before rebuilding')
        output.mkdir(parents=True)
        sources = json.loads(Path(sources_file).read_text(encoding='utf-8'))
        for source in sources:
            if file_hash(source['local']) != source['sha256']:
                raise ValueError('Source checksum mismatch')
        counts, seen = Counter(), set()
        handles = {f'{group}_{split}': (output/f'{group}_{split}.jsonl').open('w', encoding='utf-8')
                   for group in (*GROUPS, 'language_stories', 'language_knowledge')
                   for split in ('train', 'validation', 'test')}
        heldout_words = []
        words = set()

        def add(group, split, row):
            if 'messages' in row:
                ids, labels = encode_conversation(row['messages'], tokenizer)
                if len(ids)-1 > 1024:
                    counts[group+'_too_long'] += 1
                    return
                if not any(x != -100 for x in labels):
                    raise ValueError('No answer labels')
            write_row(handles[f'{group}_{split}'], row)
            counts[f'{group}_{split}'] += 1

        try:
            for source in sources:
                if 'TinyStoriesGerman' not in source['repo']:
                    continue
                for batch in pq.ParquetFile(source['local']).iter_batches(batch_size=1024):
                    for row in batch.to_pylist():
                        text = clean_text(row['text']).replace('\u200b', '')
                        digest = hashlib.sha256(normalized(text).encode()).hexdigest()
                        # Deterministic half-sample over all four shards, with normalized deduplication.
                        if int(digest[:2], 16) % 2 or digest in seen:
                            continue
                        seen.add(digest)
                        if not 250 <= len(text) <= 2800 or not acceptable(text):
                            counts['stories_filtered'] += 1
                            continue
                        split = partition(text)
                        add('language_stories', split, {'text': text, 'source': source['repo'], 'source_id': digest})
                        if int(digest[2:6], 16) % 16 == 0:
                            parts = re.split(r'(?<=[.!?])\s+', text, maxsplit=1)
                            if len(parts) == 2:
                                prompt = ['Setze diese Geschichte fort:', 'Schreibe die Geschichte weiter:',
                                          'Wie könnte diese Geschichte weitergehen?'][int(digest[6:8], 16)%3]
                                add('stories', split, dialog(prompt+'\n'+parts[0], parts[1], source['repo'], digest))
                    if counts['language_stories_train'] % 25000 < 600:
                        print(json.dumps({'phase': 'stories', 'counts': dict(counts)}), flush=True)
            for source in sources:
                if source['repo'] != 'dennlinger/klexikon':
                    continue
                for line in Path(source['local']).open(encoding='utf-8'):
                    row = json.loads(line)
                    split = 'validation' if 'validation' in source['path'] else partition(row['title'])
                    sentences = [s.strip() for s in row['klexikon_sentences'] if s.strip() and not s.startswith('=')]
                    text = ' '.join(sentences)
                    if not acceptable(text):
                        continue
                    add('language_knowledge', split, {'text': text, 'source': source['repo'],
                        'source_url': row['klexikon_url'], 'source_id': str(row['u_id'])})
                    answer = ' '.join(sentences[:4])
                    for prompt in [f'Erkläre kurz: {row["title"]}.', f'Was weißt du über {row["title"]}?']:
                        add('knowledge', split, dialog(prompt, answer, source['repo'], row['u_id']))
                    if split == 'train':
                        words.update(w for w in re.findall(r'\b[A-ZÄÖÜ][a-zäöüß]{3,15}\b', text)
                                     if partition('word:'+w) == 'train')
            for source in sources:
                if source['repo'] != 'deepset/germanquad':
                    continue
                for batch in pq.ParquetFile(source['local']).iter_batches(batch_size=256):
                    for row in batch.to_pylist():
                        split = 'test' if '/test/' in source['path'] else partition(row['context'])
                        answers = row['answers']['text']
                        if not answers or answers[0] not in row['context']:
                            counts['reading_invalid'] += 1
                            continue
                        prompt = 'Beantworte die Frage anhand des Textes.\nText: '+row['context']+'\nFrage: '+row['question']
                        add('reading', split, dialog(prompt, answers[0], source['repo'], row['id']))
            for filename in ('sft_train.jsonl', 'sft_validation.jsonl'):
                for line in (ROOT/'data/real_v1'/filename).open(encoding='utf-8'):
                    row = json.loads(line)
                    if row.get('lang') != 'de':
                        continue
                    question = next(m['content'] for m in row['messages'] if m['role'] == 'user')
                    answers = [m['content'] for m in row['messages'] if m['role'] == 'assistant']
                    if any(not 15 <= len(a) <= 1000 or not acceptable(a) or
                           re.search(r'als (?:ein )?(?:ki|sprachmodell)|ich kann (?:keine|nicht)|```', a, re.I)
                           for a in answers):
                        counts['dialog_filtered'] += 1
                        continue
                    split = 'validation' if 'validation' in filename else partition(question)
                    row['messages'][0]['content'] = SYSTEM_PROMPT+f' Datum: {date.today().isoformat()}.'
                    add('dialog', split, row)
            # Broad copying/context curriculum: thousands of words, varied prompts and distractors.
            # Synthetic strings train copying, not factual claims about real people.
            words = sorted(words)
            if len(words) < 1000:
                raise ValueError('Insufficient lexical diversity for context curriculum')
            for split, count in [('train', 30000), ('validation', 500), ('test', 100)]:
                rng = np.random.default_rng({'train': 610, 'validation': 611, 'test': 612}[split])
                for index in range(count):
                    word, distractor = rng.choice(words, 2, replace=False)
                    if split != 'train':
                        word = f'{word}{int(rng.integers(100, 999))}'
                        if split == 'test':
                            heldout_words.append(word)
                    mode = index % 8
                    if mode == 0:
                        q, a, history = f'Wiederhole genau dieses Wort: {word}', word, None
                    elif mode == 1:
                        q, a, history = f'Mein Lieblingswort ist {word}. Was ist mein Lieblingswort?', f'Dein Lieblingswort ist {word}.', None
                    elif mode == 2:
                        q, a, history = f'Merke dir: {word}. Gib nur das genannte Wort aus.', word, None
                    elif mode == 3:
                        q, a, history = 'Was ist mein Lieblingswort?', f'Dein Lieblingswort ist {word}.', [
                            {'role': 'user', 'content': f'Mein Lieblingswort ist {word}.'},
                            {'role': 'assistant', 'content': 'Alles klar.'}]
                    elif mode == 4:
                        q, a, history = f'Der gesuchte Ausdruck ist {word}, nicht {distractor}. Welcher Ausdruck ist gesucht?', word, None
                    elif mode == 5:
                        q, a, history = 'Wie lautet meine Kennung? Antworte nur mit der Kennung.', word, [
                            {'role': 'user', 'content': f'Meine Kennung lautet {word}.'},
                            {'role': 'assistant', 'content': 'Danke für die Angabe.'}]
                    elif mode == 6:
                        q, a, history = f'Im Text steht: Das Passwort lautet {word}. Wie lautet es?', word, None
                    else:
                        q, a, history = f'Erst war mein Lieblingswort {distractor}. Jetzt ist es {word}. Wie lautet mein aktuelles Lieblingswort?', word, None
                    add('context', split, dialog(q, a, 'local_synthetic_copying', f'{split}:{index}', history))
            basics_train, basics_val, _ = build_data()
            for split, rows in [('train', basics_train), ('validation', basics_val)]:
                for row in rows:
                    if row['group'] != 'dialog':
                        continue
                    row['messages'][0]['content'] += f' Datum: {date.today().isoformat()}.'
                    add('basics', split, row)
        finally:
            for stream in handles.values():
                stream.close()
        manifest = {'version': 1, 'tokenizer_hash': tokenizer.fingerprint, 'sources': sources,
                    'counts': dict(counts), 'licenses': {'fabi2347/TinyStoriesGerman': 'CDLA-Sharing-1.0',
                    'dennlinger/klexikon': 'CC-BY-SA-4.0', 'deepset/germanquad': 'CC-BY-4.0',
                    'OpenAssistant/oasst2': 'Apache-2.0', 'FreedomIntelligence/alpaca-gpt4-deutsch': 'Apache-2.0'},
                    'notes': 'Stories are automatically translated synthetic fiction, not factual reference. '
                             'Heuristic filtering and sample review do not guarantee all data is correct. '
                             'Held-out data is excluded from this cycle; old pretraining may contain related facts.',
                    'files': {p.name: file_hash(p) for p in output.glob('*.jsonl')}}
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    prepared.mkdir(parents=True, exist_ok=True)
    for group in ('stories', 'knowledge'):
        destination = prepared/group
        if not (destination/'meta.json').exists():
            prepare_corpus([output/f'language_{group}_train.jsonl'], destination, tokenizer,
                           [output/f'language_{group}_validation.jsonl'])
    for group in GROUPS:
        for split in ('train', 'validation'):
            destination = prepared/f'{group}_{split}'
            if not (destination/'meta.json').exists():
                prepare_sft([output/f'{group}_{split}.jsonl'], destination, tokenizer, 1024)
    print(json.dumps({'phase': 'ready', 'counts': manifest['counts']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sources', required=True)
    args = p.parse_args()
    prepare(args.sources, ROOT/'data/quality_v3', ROOT/'prepared/quality_v3',
            BPETokenizer.load(ROOT/'artifacts/real_v1/tokenizer.json'))
