import argparse
import hashlib
import json
import os
import re
import shutil
import time
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from context import SYSTEM_PROMPT

HEADERS = {'User-Agent': 'LocalAI-training-data/1.0'}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
        return json.load(response)


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def source_files(repo, folder):
    revision = get_json(f'https://huggingface.co/api/datasets/{repo}')['sha']
    rows = get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/{folder}?limit=1000')
    return [{'repo': repo, 'revision': revision, 'path': r['path'], 'bytes': r['size'],
             'sha256': r.get('lfs', {}).get('oid')} for r in rows if r['path'].endswith(('.parquet','.json'))]


def download(source, cache):
    path = Path(cache)/source['repo'].replace('/', '_')/source['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == source['bytes'] and sha256(path) == source['sha256']:
        return path
    if shutil.disk_usage(path.parent).free < source['bytes']+5*1024**3:
        raise RuntimeError('Not enough free disk space; at least 5 GiB reserve required')
    url = f'https://huggingface.co/datasets/{source["repo"]}/resolve/{source["revision"]}/{source["path"]}'
    partial = path.with_suffix(path.suffix+'.part')
    for attempt in range(3):
        try:
            print(f'Download: {source["repo"]}/{source["path"]} ({source["bytes"]/1e6:.1f} MB)', flush=True)
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as f:
                shutil.copyfileobj(response, f, length=1024*1024)
            if partial.stat().st_size != source['bytes'] or sha256(partial) != source['sha256']:
                raise ValueError('Download checksum mismatch')
            os.replace(partial, path)
            return path
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2)


def clean_text(text):
    text = text.replace('\r\n', '\n').replace('\x00', '')
    text = re.sub('[\U0001f000-\U0001faff\u2600-\u27bf\ufe0f]', '', text)
    return text.strip()


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()


def split_for(key, val_fraction=0.02):
    score = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)/2**64
    return 'validation' if score < val_fraction else 'train'


def write_row(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False)+'\n')


def import_wikipedia(source, cache, output, language, sample_budget):
    path = download(source, cache)
    stem = f'wiki_{language}_{Path(source["path"]).stem}'
    counts = Counter()
    handles = {split: (output/split/(stem+'.jsonl.tmp')).open('w', encoding='utf-8') for split in ('train','validation')}
    sample = (output/'tokenizer_sample.jsonl').open('a', encoding='utf-8')
    try:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            for row in batch.to_pylist():
                text = clean_text(row['text'])
                if len(text) < 400 or text.count('\ufffd') > 2:
                    counts['filtered'] += 1
                    continue
                if sum(c.isalpha() for c in text[:4000])/min(len(text),4000) < 0.5:
                    counts['filtered'] += 1
                    continue
                split = split_for(normalized(text))
                # All chunks of an article stay in the same split.
                for start in range(0, len(text), 60000):
                    chunk = text[start:start+60000]
                    if len(chunk) < 100:
                        continue
                    write_row(handles[split], {'text': chunk, 'lang': language, 'title': row['title'],
                        'source_id': row['id'], 'source_url': row['url'], 'source': source['repo']})
                    counts[split+'_chars'] += len(chunk)
                    counts[split+'_documents'] += 1
                if split == 'train' and counts['sample_chars'] < sample_budget and int(row['id']) % 13 == 0:
                    excerpt = text[:min(8192, sample_budget-counts['sample_chars'])]
                    write_row(sample, {'text': excerpt})
                    counts['sample_chars'] += len(excerpt)
    finally:
        for f in handles.values():
            f.close()
        sample.close()
    for split in handles:
        os.replace(output/split/(stem+'.jsonl.tmp'), output/split/(stem+'.jsonl'))
    print(f'Wikipedia {language}: {json.dumps(counts)}', flush=True)
    return dict(counts)


def accepted_message(row):
    if row.get('deleted') or row.get('review_result') is not True or row.get('lang') not in ('de','en'):
        return False
    labels = row.get('labels') or {}
    scores = dict(zip(labels.get('name', []), labels.get('value', [])))
    if any(scores.get(key, 0) >= 0.5 for key in ('spam','lang_mismatch','pii','fails_task')):
        return False
    if scores.get('quality', 1) < 0.5:
        return False
    if row['role'] == 'assistant' and row.get('rank') not in (None, 0):
        return False
    return bool(clean_text(row.get('text', '')))


def extract_dialogues(rows):
    by_id = {r['message_id']: r for r in rows}
    paths = {}
    for leaf in rows:
        if leaf['role'] != 'assistant' or not accepted_message(leaf):
            continue
        chain, seen, current = [], set(), leaf
        while current:
            key = current['message_id']
            if key in seen or not accepted_message(current) or current['lang'] != leaf['lang']:
                chain = []
                break
            seen.add(key)
            chain.append(current)
            parent = current.get('parent_id')
            if parent and parent not in by_id:
                chain = []
                break
            current = by_id.get(parent)
        chain.reverse()
        if not chain or len(chain) > 16 or chain[0]['role'] != 'prompter':
            continue
        if any(r['role'] != ('prompter' if i % 2 == 0 else 'assistant') for i,r in enumerate(chain)):
            continue
        root = normalized(chain[0]['text'])
        if root not in paths or len(chain) > len(paths[root]):
            paths[root] = chain
    for root, chain in sorted(paths.items()):
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        messages += [{'role': 'user' if r['role'] == 'prompter' else 'assistant',
                      'content': clean_text(r['text'])} for r in chain]
        yield split_for(root), {'messages': messages, 'lang': chain[0]['lang'],
            'source': 'OpenAssistant/oasst2', 'source_ids': [r['message_id'] for r in chain],
            'tree_id': chain[0].get('message_tree_id', chain[0]['message_id'])}


def import_dialogues(sources, cache, output):
    rows = []
    for source in sources:
        path = download(source, cache)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            rows.extend(r for r in batch.to_pylist() if r['lang'] in ('de','en'))
    counts = Counter()
    handles = {split: (output/f'sft_{split}.jsonl.tmp').open('w', encoding='utf-8') for split in ('train','validation')}
    try:
        for split, row in extract_dialogues(rows):
            write_row(handles[split], row)
            counts[split+'_'+row['lang']] += 1
    finally:
        for f in handles.values():
            f.close()
    for split in handles:
        os.replace(output/f'sft_{split}.jsonl.tmp', output/f'sft_{split}.jsonl')
    print(f'OpenAssistant: {json.dumps(counts)}', flush=True)
    return dict(counts)


def add_german_instructions(output, cache):
    manifest_path = output/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    repo = 'FreedomIntelligence/alpaca-gpt4-deutsch'
    if repo in manifest:
        print('German instructions already imported', flush=True)
        return
    source = source_files(repo, '')[0]
    path = download(source, cache)
    rows = json.loads(path.read_text(encoding='utf-8'))
    seen, counts = set(), Counter()
    handles = {split: (output/f'sft_{split}.jsonl.tmp').open('w', encoding='utf-8') for split in ('train','validation')}
    try:
        for split in handles:
            with (output/f'sft_{split}.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    row = json.loads(line)
                    root = normalized(next(m['content'] for m in row['messages'] if m['role']=='user'))
                    seen.add(root)
                    handles[split].write(line)
        for row in rows:
            turns = row['conversations']
            if len(turns) != 2 or [t['from'] for t in turns] != ['human','gpt']:
                counts['filtered'] += 1
                continue
            question, answer = [clean_text(t['value']) for t in turns]
            root = normalized(question)
            if root in seen or len(question) < 8 or len(answer) < 15 or len(question)+len(answer) > 16000:
                counts['filtered'] += 1
                continue
            if re.search(r'als (?:ein )?(?:ki[- ]?)?(?:sprach)?modell von|ich bin chatgpt|as an ai language model|ich habe (?:im internet|online) recherchiert', answer, re.I):
                counts['filtered'] += 1
                continue
            if '\ufffd' in question+answer:
                counts['filtered'] += 1
                continue
            seen.add(root)
            split = split_for(root)
            write_row(handles[split], {'messages': [{'role':'system','content':SYSTEM_PROMPT},
                {'role':'user','content':question}, {'role':'assistant','content':answer}],
                'lang':'de', 'source':repo, 'source_id':row['id'], 'synthetic':True})
            counts[split+'_de'] += 1
    finally:
        for f in handles.values():
            f.close()
    for split in handles:
        os.replace(output/f'sft_{split}.jsonl.tmp', output/f'sft_{split}.jsonl')
    manifest[repo] = {'url':f'https://huggingface.co/datasets/{repo}',
        'license':'Apache-2.0 (dataset card)', 'synthetic':True, 'statistics':dict(counts)}
    manifest['sources'].append(source)
    manifest['files'] = {str(f.relative_to(output)): {'bytes':f.stat().st_size,'sha256':sha256(f)}
                         for f in sorted(output.rglob('*.jsonl'))}
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'German instructions: {json.dumps(counts)}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='data/real_v1')
    p.add_argument('--cache', default='../../work/data-cache')
    p.add_argument('--add-instructions', action='store_true')
    a = p.parse_args()
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    if a.add_instructions:
        add_german_instructions(output, a.cache)
        return
    if (output/'manifest.json').exists():
        print(f'Already prepared: {output}/manifest.json')
        return
    if (output/'tokenizer_sample.jsonl').exists():
        raise FileExistsError('Partial import exists; use a new output directory (downloads are cached)')
    for split in ('train','validation'):
        (output/split).mkdir(exist_ok=True)
    wiki_de = source_files('wikimedia/wikipedia','20231101.de')
    wiki_en = source_files('wikimedia/wikipedia','20231101.en')
    wiki = [(wiki_de[i], 'de') for i in (0,10,19)]+[(wiki_en[20], 'en')]
    dialogue = source_files('OpenAssistant/oasst2','data')
    manifest = {'wikipedia': {'url': 'https://huggingface.co/datasets/wikimedia/wikipedia',
                    'license': 'CC-BY-SA-3.0 and GFDL (dataset card)', 'snapshot': '20231101'},
                'openassistant': {'url': 'https://huggingface.co/datasets/OpenAssistant/oasst2', 'license': 'Apache-2.0'},
                'sources': [s for s,_ in wiki]+dialogue, 'statistics': {}}
    (output/'sources.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    manifest['statistics']['dialogues'] = import_dialogues(dialogue, a.cache, output)
    for source, language in wiki:
        manifest['statistics'][source['path']] = import_wikipedia(source,a.cache,output,language,6_000_000)
    manifest['files'] = {str(f.relative_to(output)): {'bytes': f.stat().st_size, 'sha256': sha256(f)}
                         for f in sorted(output.rglob('*.jsonl'))}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Data ready: {output}',flush=True)


if __name__ == '__main__':
    main()
