"""V5 practice: natural questions, concise factual prose, grounded answers and worked sums."""
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from data import SFTDataset, encode_conversation, file_hash, prepare_sft
from dialogue_data import split_key
from quality_data import Mixture, dialog
from tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data/improvement_v5'
PREPARED = ROOT/'prepared/improvement_v5'
GROUPS = ('natural', 'facts', 'reading', 'dialog', 'context', 'arithmetic', 'format')
VALIDATION_WEIGHTS = [.10, .25, .25, .15, .15, .05, .05]


def jsonlines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ])', text) if s.strip()]


def add_working(row):
    q = row['messages'][-2]['content']
    match = re.search(r'(\d+)\s*([+-])\s*(\d+)', q)
    if not match:
        return None
    a, op, b = int(match[1]), match[2], int(match[3])
    tens, ones = (b//10)*10, b % 10
    middle = a+tens if op == '+' else a-tens
    result = middle+ones if op == '+' else middle-ones
    if result < 0:
        return None
    word = 'plus' if op == '+' else 'minus'
    question = [f'Wie viel sind {a} {word} {b}? Zeige den Rechenweg.',
                f'Berechne {a} {op} {b} in kleinen Schritten.',
                f'Was ergibt {a} {word} {b}?'][a % 3]
    answer = f'{a} {op} {tens} = {middle}. {middle} {op} {ones} = {result}. Ergebnis: {result}.'
    return dialog(question, answer, 'local_verified_arithmetic_steps', row['source_id'])


def natural_examples():
    # Authored targets are used only as training data, never as chat-response rules.
    skills = [
        (['Wer bist du?', 'Erzähl kurz, was du bist.', 'Was für eine KI bist du?', 'Stelle dich vor.'],
         'Ich bin ein kleines Sprachmodell, das lokal auf deinem Computer läuft. Meine Antworten können Fehler enthalten.'),
        (['Wie heißt du?', 'Hast du einen Namen?', 'Wie darf ich dich nennen?'],
         'Ich habe keinen festgelegten persönlichen Namen. Du kannst mir einen Namen geben.'),
        (['Wie geht es dir?', 'Wie fühlst du dich heute?', 'Alles gut bei dir?'],
         'Ich habe kein eigenes menschliches Befinden. Ich bin bereit, mit dir zu schreiben. Wie geht es dir?'),
        (['Hast du ein eigenes Leben?', 'Wo bist du aufgewachsen?', 'In welche Schule gehst du?'],
         'Ich bin ein Computerprogramm und habe keine menschliche Lebensgeschichte.'),
        (['Hallo.', 'Hi, wie läuft es?', 'Guten Tag!', 'Hey!', 'Servus!'],
         'Hallo! Wobei kann ich dir helfen?'),
        (['Kannst du mir helfen?', 'Ich brauche etwas Hilfe.', 'Kann ich dich etwas fragen?'],
         'Ja, worum geht es? Beschreibe bitte deine Frage.'),
        (['Was soll ich dir erzählen?', 'Worüber könnten wir reden?', 'Was willst du mich fragen?'],
         'Was interessiert dich gerade? Wir können darüber sprechen.'),
        (['Danke!', 'Das hat mir geholfen.', 'Vielen Dank für deine Antwort.'], 'Gern geschehen!'),
        (['Die Antwort war falsch.', 'Das stimmt nicht.', 'Du hast etwas verwechselt.'],
         'Entschuldigung. Was genau war falsch? Ich möchte die Antwort korrigieren.'),
        (['Das ergibt keinen Sinn.', 'Du redest am Thema vorbei.', 'Du beantwortest meine Frage nicht.'],
         'Entschuldigung. Bitte nenne noch einmal die Frage, auf die ich eingehen soll.'),
        (['Erkläre es mir.', 'Was bedeutet das?', 'Ich verstehe es nicht.'],
         'Welches Thema oder welchen Satz soll ich erklären?'),
        (['Fasse das zusammen.', 'Mach eine kurze Zusammenfassung.', 'Kürze den Text.'],
         'Schick mir bitte den Text, den ich zusammenfassen soll.'),
        (['Ist das eine gute Idee?', 'Soll ich das machen?', 'Was ist besser?'],
         'Worum geht es genau? Nenne bitte die Möglichkeiten, damit ich sie vergleichen kann.'),
        (['Weißt du alles über mich?', 'Kennst du meine Vergangenheit?', 'Was weißt du über mein Leben?'],
         'Ich weiß nur, was du mir im verfügbaren Gesprächskontext mitgeteilt hast.'),
        (['Kannst du mich sehen?', 'Welche Kleidung trage ich?', 'Wie sehe ich gerade aus?'],
         'Ich kann dich in diesem Textchat nicht sehen.'),
        (['Wie ist das Wetter gerade?', 'Regnet es draußen?', 'Wie warm ist es jetzt bei mir?'],
         'Ich habe keine aktuellen Wetterdaten und kenne deinen Standort nicht.'),
        (['Kannst du im Internet nachsehen?', 'Hast du Zugriff auf Webseiten?', 'Kannst du online suchen?'],
         'Dieser lokale Chat hat keinen Internetzugriff.'),
        (['Kannst du rechnen?', 'Sind deine Rechnungen immer richtig?', 'Bist du gut in Mathematik?'],
         'Ich kann einfache Rechnungen versuchen, mache dabei aber noch Fehler. Prüfe wichtige Ergebnisse.'),
        (['Was kannst du machen?', 'Wobei kannst du helfen?', 'Welche Aufgaben kannst du bearbeiten?'],
         'Ich kann kurze Texte erzeugen und einfache Fragen beantworten. Meine Fähigkeiten sind noch begrenzt.'),
        (['Tschüss.', 'Bis morgen.', 'Ich gehe jetzt.', 'Bis bald!'], 'Bis bald!'),
    ]
    for i, (questions, answer) in enumerate(skills):
        for j, q in enumerate(questions):
            yield ('validation' if j == len(questions)-1 else 'train'), dialog(q, answer, 'local_authored_dialogue', f'skill:{i}:{j}')
    names = 'Alina Bastian Carina Dorian Elisa Fabian Georgina Henrik Ilona Justus Karina Leander Magdalena Niklas Olivia Patrick Quentin Ramona Susanne Tristan Ulrike Vanessa Wilma Xaver Yasmin Zora'.split()
    activities = ['gern Fahrrad', 'gern mit dem Bus', 'selten mit dem Zug', 'oft mit dem Auto']
    for i, name in enumerate(names):
        split = 'validation' if i % 7 == 0 else 'train'
        for j, activity in enumerate(activities):
            q = [f'Ich heiße {name}. Was kannst du über mich sagen?',
                 f'Mein Name ist {name}. Erzähle, was du über mich weißt.',
                 f'Du kannst mich {name} nennen. Was weißt du bisher?',
                 f'Ich bin {name}. Kennst du mich?'][j]
            yield split, dialog(q, f'Du hast mir gesagt, dass du {name} heißt. Mehr weiß ich bisher nicht über dich.',
                                'local_authored_context', f'name:{i}:{j}')
            history = [{'role':'user', 'content':f'Ich heiße {name} und fahre {activity}.'},
                       {'role':'assistant', 'content':'Danke für die Angaben.'}]
            yield split, dialog('Was habe ich über mich erzählt?', f'Du heißt {name} und fährst {activity}.',
                                'local_authored_context', f'profile:{i}:{j}', history)


def build():
    tokenizer = BPETokenizer.load(ROOT/'artifacts/real_v1/tokenizer.json')
    old = ROOT/'data/dialogue_v4'
    if not (DATA/'manifest.json').exists():
        if DATA.exists():
            raise FileExistsError('Incomplete improvement data; inspect before retrying')
        DATA.mkdir(parents=True)
        handles = {(g,s):(DATA/f'{g}_{s}.jsonl').open('w',encoding='utf-8')
                   for g in GROUPS for s in ('train','validation')}
        seen, counts, parent_files = {}, Counter(), {}

        def add(group, split, row):
            key = json.dumps(row['messages'][1:],sort_keys=True,ensure_ascii=False)
            if key in seen:
                return
            ids, _ = encode_conversation(row['messages'],tokenizer)
            if len(ids)-1 > 512:
                counts[group+'_too_long'] += 1
                return
            seen[key] = split
            handles[group,split].write(json.dumps(row,ensure_ascii=False)+'\n')
            counts[group+'_'+split] += 1

        try:
            for split in ('validation','train'):
                for group in ('reading','dialog','context','arithmetic','format'):
                    path = old/f'{group}_{split}.jsonl'
                    parent_files[str(path.relative_to(ROOT))] = file_hash(path)
                    for row in jsonlines(path):
                        if group == 'dialog':
                            q = row['messages'][-2]['content']
                            answers = [m['content'] for m in row['messages'] if m['role']=='assistant']
                            if len(row['messages']) != 3 or any(len(a)>450 for a in answers):
                                continue
                            if re.search(r'geschichte|gedicht|fantasie|rollenspiel|komplex|algorithmus|fiktiv|hash|programmier|quellcode',q,re.I):
                                continue
                            if re.search(r'mit (?:den )?(?:Wörtern|Worten)', q):
                                required = re.findall(r'"([^"\n]+)"', q)
                                if any(w.casefold() not in answers[0].casefold() for w in required):
                                    continue
                        add(group,split,row)
                        if group == 'arithmetic':
                            worked = add_working(row)
                            if worked:
                                add(group,split,worked)
                        elif group == 'reading' and row['source']=='deepset/germanquad':
                            q = row['messages'][-2]['content']
                            context, question = q.split('\nText: ',1)[1].rsplit('\nFrage: ',1)
                            answer = row['messages'][-1]['content'].strip()
                            containing = [s for s in sentences(context) if answer in s]
                            if containing and len(containing[0]) <= 600:
                                prompt = f'Lies diesen Satz und beantworte dann die Frage.\n{containing[0]}\nFrage: {question}'
                                add(group,split,dialog(prompt,answer,row['source'],row['source_id']))
                path = old/f'knowledge_{split}.jsonl'
                parent_files[str(path.relative_to(ROOT))] = file_hash(path)
                topic_seen = set()
                for row in jsonlines(path):
                    topic_id = row['source_id']
                    if topic_id in topic_seen:
                        continue
                    topic_seen.add(topic_id)
                    q = row['messages'][-2]['content']
                    if not q.startswith('Erkläre kurz: '):
                        continue
                    topic = q[len('Erkläre kurz: '):].rstrip('.')
                    answer = ' '.join(sentences(row['messages'][-1]['content'])[:2])
                    for q in [f'Was weißt du über {topic}?', f'Erkläre mir bitte {topic}.',
                              f'Thema: {topic}. Gib mir eine kurze Erklärung.', f'Was bedeutet der Begriff „{topic}“?',
                              f'Kannst du mir kurz etwas über {topic} sagen?']:
                        add('facts',split,dialog(q,answer,row['source'],topic_id))
                    source = row['messages'][-1]['content']
                    add('reading',split,dialog('Fasse diesen Text in zwei Sätzen zusammen:\n'+source,
                                               answer,row['source'],topic_id))
                # Existing short, authored knowledge and dialogue targets remain represented.
                path = old/f'basics_{split}.jsonl'
                parent_files[str(path.relative_to(ROOT))] = file_hash(path)
                for row in jsonlines(path):
                    add('natural',split,row)
            for split,row in natural_examples():
                add('natural',split,row)
        finally:
            for f in handles.values():
                f.close()
        manifest = {'version':1,'tokenizer_hash':tokenizer.fingerprint,'context':512,
                    'counts':dict(counts),'parent_files':parent_files,
                    'parent_manifest_sha256':file_hash(old/'manifest.json'),
                    'validation_weights':VALIDATION_WEIGHTS,'loss_reduction':'example',
                    'licenses':json.loads((old/'manifest.json').read_text(encoding='utf-8'))['licenses'],
                    'note':'Natural question variants share source topics; no claim of new facts for each variant. '
                           'Source train/validation split preserved. No former dev/test rows used for training. '
                           'Worked arithmetic is generated exactly; imported prose filtered heuristically.',
                    'files':{p.name:file_hash(p) for p in DATA.glob('*.jsonl')}}
        (DATA/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest = json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    if manifest['tokenizer_hash'] != tokenizer.fingerprint:
        raise ValueError('Tokenizer mismatch')
    for path,digest in manifest['files'].items():
        if file_hash(DATA/path) != digest:
            raise ValueError('Changed data: '+path)
    for group in GROUPS:
        for split in ('train','validation'):
            target = PREPARED/f'{group}_{split}'
            if not (target/'meta.json').exists():
                prepare_sft([DATA/f'{group}_{split}.jsonl'],target,tokenizer,512)
    print(json.dumps({'phase':'data_ready','counts':manifest['counts']}),flush=True)
    return manifest


def datasets(tokenizer, weights):
    result = []
    for split, mix in [('train',weights),('validation',VALIDATION_WEIGHTS)]:
        value = Mixture([SFTDataset(PREPARED/f'{g}_{split}',512,tokenizer) for g in GROUPS],mix,trim=True)
        value.loss_reduction = 'example'
        value.fingerprint = hashlib.sha256((value.fingerprint+':example').encode()).hexdigest()
        result.append(value)
    return tuple(result)


if __name__=='__main__':
    build()
