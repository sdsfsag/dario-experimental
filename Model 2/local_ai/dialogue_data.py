"""Dialogue v4: local attributed data, shorter evidence, no story-completion SFT."""
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from data import SFTDataset, encode_conversation, file_hash, prepare_sft
from quality_data import Mixture, dialog
from tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data/dialogue_v4'
PREPARED = ROOT/'prepared/dialogue_v4'
WEIGHTS = {'dialog': .30, 'reading': .24, 'context': .22, 'knowledge': .08,
           'arithmetic': .10, 'format': .04, 'basics': .02}
SPLITS = ('train', 'validation', 'dev', 'test')


def split_key(key):
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 90 else 'validation' if bucket < 94 else 'dev' if bucket < 97 else 'test'


def arithmetic_rows(limit=100):
    for a in range(limit):
        for b in range(limit):
            for operation in ('+', '-'):
                if operation == '-' and b > a:
                    continue
                # Addition and its reversed pair always share a split.
                key = f'arithmetic:{operation}:{min(a,b)}:{max(a,b)}'
                prompts = [f'Was ist {a} {operation} {b}? Antworte nur mit der Zahl.',
                           f'Berechne {a} {operation} {b}. Gib nur das Ergebnis aus.',
                           f'{a} {operation} {b} = ? Nur die Zahl bitte.']
                answer = str(a+b if operation == '+' else a-b)
                yield split_key(key), dialog(prompts[(a+b)%3], answer, 'local_exact_arithmetic', key)


def synthetic_rows(count=18000):
    rng = random.Random(2026091901)
    names = ('Anna Ben Clara David Emma Felix Greta Hannes Ida Jonas Klara Leon Mia Nils Paula Rudi Sara Theo Vera Yann '
             'Abel Adam Adrian Alex Amelie Andreas Anita Anton Arne Arthur Axel Bea Bella Carla Carmen Daria Denis '
             'Dora Elias Elena Ella Emil Emilia Eric Eva Finn Flora Frank Georg Hannah Hugo Ines Iris Jakob Jan '
             'Jana Jens Joel Jonathan Julia Juna Kai Karl Katja Kevin Laura Lena Leo Lina Lisa Livia Luca Luis '
             'Lukas Luna Malte Marc Maria Marie Martin Max Michael Milan Mira Mona Moritz Nadja Naomi Nele Noah '
             'Nora Olaf Oskar Otto Paul Peter Pia Rafael Ralf Ria Rita Robert Robin Ronja Rosa Samuel Sandra Simon '
             'Sofia Sven Tanja Tessa Till Tim Tina Tobias Tom Udo Uwe Valentin Victor Walter').split()
    objects = ('Ball Buch Heft Glas Schuh Stift Becher Stein Schlüssel Rucksack Hut Löffel Brief Teller Würfel '
               'Lampe Tasche Kerze Kissen Flasche Schachtel Brille Uhr Bild Ring Schere Lineal Pinsel Seife '
               'Tuch Tasse Kanne Topf Schale Korb Knopf Münze Karte Zeitung Schal Jacke Spiegel Kabel Besen').split()
    places = ['im Garten', 'in der Küche', 'auf dem Tisch', 'im Flur', 'im Regal', 'unter dem Bett',
              'neben der Tür', 'auf dem Stuhl', 'in der Garage', 'am Fenster']
    for i in range(count):
        name, other = rng.sample(names, 2)
        item, distractor = rng.sample(objects, 2)
        place, elsewhere = rng.sample(places, 2)
        # Related prompt variants of the same fact tuple must stay together.
        key = f'situation:{name}:{other}:{item}:{distractor}:{place}:{elsewhere}'
        split = split_key(key)
        mode = i % 6
        if mode == 0:
            q = f'{name} hat einen Gegenstand: {item}. {other} hat einen anderen: {distractor}. Was hat {name}? Nenne nur den Gegenstand.'
            a, history, group = item, None, 'reading'
        elif mode == 1:
            q = f'Der Gegenstand liegt {place}. {name} sitzt {elsewhere}. Wo liegt der Gegenstand? Antworte nur mit dem Ort.'
            a, history, group = place, None, 'reading'
        elif mode == 2:
            history = [{'role': 'user', 'content': f'Ich heiße {name}.'},
                       {'role': 'assistant', 'content': f'Hallo {name}!'},
                       {'role': 'user', 'content': f'{other} ist auch hier.'},
                       {'role': 'assistant', 'content': 'Verstanden.'}]
            q, a, group = 'Wie heiße ich? Nenne nur meinen Namen.', name, 'context'
        elif mode == 3:
            history = [{'role': 'user', 'content': f'Mein gesuchtes Wort lautet {distractor}.'},
                       {'role': 'assistant', 'content': f'Du hast {distractor} genannt.'},
                       {'role': 'user', 'content': f'Korrektur: Jetzt ist mein gesuchtes Wort {item}.'},
                       {'role': 'assistant', 'content': f'Alles klar, jetzt ist es {item}.'}]
            q, a, group = 'Was ist mein aktuelles Wort? Nenne nur das Wort.', item, 'context'
        elif mode == 4:
            history = [{'role': 'user', 'content': f'Der Schlüssel liegt {place}.'},
                       {'role': 'assistant', 'content': f'Du hast gesagt: {place}.'}]
            q, a, group = 'Wo liegt der Schlüssel? Antworte nur mit dem Ort.', place, 'context'
        else:
            q = f'{name} sucht einen Gegenstand. Der Gegenstand heißt {item}. Wie heißt der Gegenstand? Nur ein Wort.'
            a, history, group = item, None, 'reading'
        yield group, split, dialog(q, a, 'local_synthetic_grounding', key, history)


def short_reading(row):
    question = row['messages'][-2]['content']
    if '\nText: ' not in question or '\nFrage: ' not in question:
        return None
    context, question = question.split('\nText: ', 1)[1].rsplit('\nFrage: ', 1)
    answer = row['messages'][-1]['content']
    index = context.find(answer)
    if index < 0 or len(answer) > 160:
        return None
    # Keep whole sentences around the annotated answer, not an answer-only field.
    starts = [0]+[m.end() for m in re.finditer(r'(?<=[.!?])\s+', context)]
    start = max(s for s in starts if s <= max(0, index-160))
    end = next((s for s in starts if s >= index+len(answer)+160), len(context))
    excerpt = context[start:end].strip()
    if len(excerpt) > 1500:
        return None
    return dialog('Beantworte die Frage anhand des Textes. Antworte kurz.\nText: '+excerpt+
                  '\nFrage: '+question, answer, row['source'], row['source_id'])


def build():
    tokenizer = BPETokenizer.load(ROOT/'artifacts/real_v1/tokenizer.json')
    old = ROOT/'data/quality_v3'
    if not (DATA/'manifest.json').exists():
        if DATA.exists():
            raise FileExistsError('Incomplete v4 data: inspect before retrying')
        DATA.mkdir(parents=True)
        handles = {(g, s): (DATA/f'{g}_{s}.jsonl').open('w', encoding='utf-8')
                   for g in WEIGHTS for s in SPLITS}
        counts, seen, provenance = Counter(), {}, {}

        def add(group, split, row):
            ids, _ = encode_conversation(row['messages'], tokenizer)
            if len(ids)-1 > 512:
                counts[group+'_too_long'] += 1
                return
            # Ignore dates/metadata when checking duplicate conversations across splits.
            key = json.dumps(row['messages'][1:], sort_keys=True, ensure_ascii=False)
            if key in seen:
                counts[group+'_duplicate'] += 1
                return
            seen[key] = split
            handles[group, split].write(json.dumps(row, ensure_ascii=False)+'\n')
            counts[group+'_'+split] += 1

        try:
            for group in ('dialog', 'reading', 'knowledge', 'context', 'basics'):
                for split in ('validation', 'test', 'train'):
                    path = old/f'{group}_{split}.jsonl'
                    provenance[path.name] = file_hash(path)
                    for i, line in enumerate(path.open(encoding='utf-8')):
                        row = json.loads(line)
                        destination = ('dev' if i % 2 == 0 else 'test') if split == 'test' else split
                        if group == 'reading':
                            row = short_reading(row)
                            if not row:
                                continue
                        elif group == 'context' and split == 'train' and i >= 12000:
                            break
                        elif group == 'dialog':
                            answers = [m['content'] for m in row['messages'] if m['role'] == 'assistant']
                            if any(len(a) > 650 or re.search(r'ich (?:bin|heiße|heisse) (?:lily|ein mensch)|openai|chatgpt', a, re.I) for a in answers):
                                continue
                        add(group, destination, row)
            for group, split, row in synthetic_rows():
                add(group, split, row)
            for split, row in arithmetic_rows():
                add('arithmetic', split, row)
            # Explicit instruction examples: order, exact number of entries/sentences.
            nouns = 'Katze Hund Maus Pferd Vogel Fisch Hase Igel Fuchs Bär Ente Eule'.split()
            for noun in nouns:
                article = 'Die' if noun in ('Katze', 'Maus', 'Ente', 'Eule') else 'Das' if noun == 'Pferd' else 'Der'
                pronoun = {'Die': 'Sie', 'Das': 'Es', 'Der': 'Er'}[article]
                for location in ('im Garten', 'am Haus', 'im Park', 'auf einer Wiese', 'am See'):
                    sentences = [f'{article} {noun} ist {location}.', f'{pronoun} sucht etwas zu essen.', f'Danach ruht sich {pronoun.lower()} aus.']
                    key = f'format:{noun}:{location}'
                    for number in (1, 2, 3):
                        q = f'Schreibe genau {number} '+('Satz' if number == 1 else 'Sätze')+f' über folgendes Tier: {noun}. Ort: {location}.'
                        add('format', split_key(key), dialog(q, ' '.join(sentences[:number]), 'local_authored_format', key))
            for i in range(3000):
                rng = random.Random(i+1909)
                words = rng.sample(nouns, 5)
                key = 'list:'+':'.join(words)
                count = 1+i % 4
                q = 'Wörter: '+', '.join(words)+f'. Gib genau die ersten {count} Wörter aus, getrennt durch Kommas.'
                add('format', split_key(key), dialog(q, ', '.join(words[:count]), 'local_exact_format', key))
            identity = [
                ('Bist du ein Mensch?', 'Nein. Ich bin ein lokal trainiertes KI-Sprachmodell.'),
                ('Erzähl mir etwas über dich.', 'Ich bin ein kleines, lokal trainiertes Sprachmodell. Ich beantworte Fragen, kann aber Fehler machen.'),
                ('Gehst du in die Schule?', 'Nein. Ich bin ein KI-Sprachmodell und gehe nicht in die Schule.'),
                ('Hast du Freunde?', 'Ich habe kein eigenes Sozialleben. Ich bin ein KI-Sprachmodell.'),
                ('Was kannst du?', 'Ich kann kurze Texte schreiben und Fragen beantworten. Meine Fähigkeiten sind noch begrenzt.'),
                ('Du hast meine Frage nicht beantwortet.', 'Entschuldigung. Welche Frage soll ich genauer beantworten?'),
                ('Ich verstehe deine Antwort nicht.', 'Entschuldigung. Welchen Teil soll ich einfacher erklären?'),
                ('Warum redest du von einem Hund? Ich habe nach etwas anderem gefragt.', 'Entschuldigung, meine Antwort passte nicht zu deiner Frage.'),
                ('Kannst du alles wissen?', 'Nein. Mein Wissen ist begrenzt und ich kann mich irren.'),
                ('Wie ist dein Name?', 'Ich bin dein lokal trainiertes KI-Sprachmodell.'),
            ]
            for i, (q, a) in enumerate(identity):
                add('basics', 'train', dialog(q, a, 'local_authored_identity', f'identity:{i}'))
        finally:
            for stream in handles.values():
                stream.close()
        manifest = {'version': 1, 'tokenizer_hash': tokenizer.fingerprint, 'weights': WEIGHTS,
                    'stored_context': 512, 'loss_reduction': 'example', 'counts': dict(counts),
                    'parent_manifest_sha256': file_hash(old/'manifest.json'), 'parent_files': provenance,
                    'licenses': {k: v for k, v in json.loads((old/'manifest.json').read_text())['licenses'].items()
                                 if k != 'fabi2347/TinyStoriesGerman'},
                    'notes': 'Synthetic examples teach operations, not real personal facts. No story completions. '
                             'Dev/test excluded from this cycle; related facts may occur in earlier training. '
                             'Imported data is filtered heuristically, not exhaustively human-verified.',
                    'files': {p.name: file_hash(p) for p in DATA.glob('*.jsonl')}}
        (DATA/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    if manifest['tokenizer_hash'] != tokenizer.fingerprint:
        raise ValueError('Tokenizer changed')
    for filename, digest in manifest['files'].items():
        if file_hash(DATA/filename) != digest:
            raise ValueError('Data changed: '+filename)
    for group in WEIGHTS:
        for split in ('train', 'validation'):
            destination = PREPARED/f'{group}_{split}'
            if not (destination/'meta.json').exists():
                prepare_sft([DATA/f'{group}_{split}.jsonl'], destination, tokenizer, 512)
            print(json.dumps({'phase': 'prepared', 'group': group, 'split': split}), flush=True)
    print(json.dumps({'phase': 'data_ready', 'counts': manifest['counts']}), flush=True)
    return manifest


def datasets(tokenizer):
    result = []
    for split in ('train', 'validation'):
        mixture = Mixture([SFTDataset(PREPARED/f'{g}_{split}', 512, tokenizer) for g in WEIGHTS],
                          list(WEIGHTS.values()), trim=True)
        mixture.loss_reduction = 'example'
        mixture.fingerprint = hashlib.sha256((mixture.fingerprint+':example').encode()).hexdigest()
        result.append(mixture)
    return tuple(result)


def probes(split):
    if split not in ('dev', 'test'):
        raise ValueError('Only held-out generation probes')
    result = []
    tokenizer = BPETokenizer.load(ROOT/'artifacts/real_v1/tokenizer.json')
    for group, wanted in [('context', 24), ('reading', 24), ('arithmetic', 12), ('format', 6)]:
        rows = [json.loads(line) for line in (DATA/f'{group}_{split}.jsonl').open(encoding='utf-8')]
        rows = [row for row in rows if len(tokenizer.encode(row['messages'][-1]['content'])) <= 24]
        rng = random.Random(719 if split == 'dev' else 720)
        rng.shuffle(rows)
        selected = []
        if group == 'reading':
            for synthetic in (False, True):
                selected += [r for r in rows if ('local_' in r['source']) == synthetic][:wanted//2]
        else:
            selected = rows[:wanted]
        for row in selected:
            result.append({'group': group, 'messages': row['messages'][1:-1],
                           'expected': row['messages'][-1]['content']})
    open_questions = ['Stell dich in einem Satz vor.', 'Was ist ein Berg?',
              'Weshalb schmilzt Schnee in der Sonne?', 'Schreibe drei Sätze über einen Ausflug.',
              'Ich heiße Selma. Was weißt du über mich?', 'Was möchtest du von mir wissen?',
              'Wie viel sind 36 plus 29?', 'Ich suche Hilfe. Frag mich, wobei.'] if split == 'dev' else [
              'Sag mir bitte, mit wem ich spreche.', 'Was ist eine Insel?',
              'Warum wird Wasser zu Eis, wenn es sehr kalt ist?', 'Schreibe drei Sätze über einen Spaziergang.',
              'Du kannst mich Elif nennen. Wie heiße ich?', 'Kannst du mir eine Frage stellen?',
              'Wie viel sind 47 plus 16?', 'Ich verstehe das noch nicht. Bitte frage nach, was unklar ist.']
    for q in open_questions:
        result.append({'group': 'open_ended', 'question': q, 'expected': None})
    return result


if __name__ == '__main__':
    build()
