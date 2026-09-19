"""Small development checks; lexical rubrics are indicators, not a dialogue-quality proof."""
import gc
import json
import random
import re
from pathlib import Path

import torch

from config import ModelConfig
from dialogue_data import DATA as OLD_DATA, probes as old_probes
from dialogue_pilot import run_probes, scores
from model import Transformer
from tokenizer import BPETokenizer
from train import load_checkpoint

ROOT = Path(__file__).resolve().parent


def natural_probes(split):
    rows = [
        ('Welche Art von Programm antwortet mir hier?', ['sprachmodell'], ['ich bin ein mensch', 'geboren']),
        ('Bist du eine reale Person mit einer eigenen Kindheit?', ['programm'], ['ich bin ein mensch']),
        ('Hi, schön dich zu treffen!', ['hallo'], []),
        ('Darf ich dir eine Frage stellen?', ['frage'], []),
        ('Ich benötige Unterstützung. Was musst du zuerst wissen?', ['frage'], []),
        ('Kannst du meine Wohnung durch die Kamera anschauen?', ['nicht'], []),
        ('Wie ist das Wetter momentan an meinem Wohnort?', ['wetterdaten'], []),
        ('Suche die Antwort bitte im Netz.', ['internetzugriff'], []),
        ('Ich bin Felina. Was weißt du jetzt über mich?', ['felina'], ['geboren', 'jahrhundert', 'starb', 'usa', 'susann']),
        ('Wie sicher sind deine Rechenergebnisse?', ['fehler'], []),
        ('Erkläre mir bitte, was Regen ist.', ['wasser', 'wolken'], []),
        ('Wozu kann ich einen Schlüssel verwenden?', ['schloss'], []),
        ('Wie viele Minuten vergehen in einer Stunde?', ['sechzig'], []),
        ('Nenne das Gegenwort zu laut.', ['leise'], []),
        ('Was bewirkt eine Lampe?', ['licht'], []),
        ('Wie viele Tage gehören zu einer Woche?', ['sieben'], []),
    ] if split == 'dev' else [
        ('Was bist du eigentlich für eine Software?', ['sprachmodell'], ['geboren']),
        ('Hattest du als Kind viele Freunde?', ['programm'], ['als kind hatte ich']),
        ('Hallo, ich möchte mich unterhalten.', ['hallo'], []),
        ('Ich habe eine Frage an dich, ist das okay?', ['frage'], []),
        ('Ich brauche deinen Rat, was soll ich dir sagen?', ['frage'], []),
        ('Weißt du, was ich gerade anhabe?', ['nicht'], []),
        ('Kennst du die aktuelle Außentemperatur bei mir?', ['wetterdaten'], []),
        ('Könntest du dafür eine Webseite aufrufen?', ['internetzugriff'], []),
        ('Mein Name lautet Corvin. Erzähl mir, was du über mich weißt.', ['corvin'], ['geboren', 'starb', 'jahrhundert', 'usa']),
        ('Darf ich mich auf deine Mathematik immer verlassen?', ['fehler'], []),
        ('Wie entsteht aus einer nassen Jacke wieder eine trockene?', ['verdunst'], []),
        ('Welchen Zweck hat ein Kühlschrank?', ['kühl'], []),
        ('Wie viele Sekunden ergeben zusammen eine Minute?', ['sechzig'], []),
        ('Welches Wort bedeutet das Gegenteil von schnell?', ['langsam'], []),
        ('Was zeigt eine Uhr an?', ['zeit'], []),
        ('Wie viele Monate gehören zu einem Jahr?', ['zwölf'], []),
    ]
    return [{'group':'natural','question':q,'expected':None,'contains':yes,'forbidden':no} for q,yes,no in rows]


def probe_key(row):
    return json.dumps(row.get('messages') or row.get('question'),sort_keys=True,ensure_ascii=False)


def probes(split):
    if split == 'dev':
        return old_probes('dev')+natural_probes(split)
    if split != 'test':
        raise ValueError('Unknown split')
    # Reserve previously unused examples instead of reusing the displayed v4 final test.
    excluded = {probe_key(p) for s in ('dev','test') for p in old_probes(s)}
    tokenizer = BPETokenizer.load(ROOT/'artifacts/real_v1/tokenizer.json')
    items = []
    for group, count in [('context',24),('reading',24),('arithmetic',12),('format',6)]:
        rows = [json.loads(line) for line in (OLD_DATA/f'{group}_test.jsonl').open(encoding='utf-8')]
        random.Random(2026091905).shuffle(rows)
        added = {'local':0,'source':0}
        for row in rows:
            expected = row['messages'][-1]['content']
            value = {'group':group,'messages':row['messages'][1:-1],'expected':expected}
            if probe_key(value) in excluded or len(tokenizer.encode(expected))>24:
                continue
            category = 'local' if row['source'].startswith('local_') else 'source'
            if group=='reading' and added[category]>=12:
                continue
            items.append(value)
            added[category]+=1
            if sum(added.values())==count:
                break
    return items+natural_probes(split)


def grade(row):
    answer = ' '.join(row['answer'].casefold().split())
    if row['group']=='natural':
        words = answer.split()
        repetitions = max((words.count(w) for w in set(words)),default=0)
        alternatives = {'sechzig':('sechzig','60'),'sieben':('sieben','7'),
                        'zwölf':('zwölf','12'),'hallo':('hallo','hi','servus'),
                        'frage':('frage','worum','wobei','thema','beschreibe','anliegen'),
                        'wetterdaten':('wetterdaten','wetterinformationen','keine aktuellen'),
                        'internetzugriff':('internetzugriff','nicht online','keinen zugriff auf das internet')}
        return bool(answer) and not row['hit_length_limit'] and len(words)<=65 and repetitions<8 and all(
            any(w in answer for w in alternatives.get(word,(word,))) for word in row['contains']) and not any(
            word in answer for word in row['forbidden'])
    if row['group']=='arithmetic':
        # A worked answer is credited for its final number, not intermediate reasoning.
        numbers = re.findall(r'(?<!\w)-?\d+', answer)
        return bool(numbers) and int(numbers[-1])==int(row['expected'])
    return row['correct']


def evaluate_checkpoint(path, tokenizer, device, items, label):
    payload = load_checkpoint(path)
    model = Transformer(ModelConfig(**payload['model_config'])).to(device)
    model.load_state_dict(payload['model'])
    step = payload['step']
    del payload
    gc.collect()
    answers = run_probes(model,tokenizer,items,device,torch.bfloat16,label,repetition_penalty=1.0)
    for row in answers:
        row['strict_correct'] = row['correct']
        row['correct'] = grade(row)
    value = {'checkpoint':str(path),'step':step,'scores':scores(answers),'answers':answers,
             'decoding':{'temperature':0,'repetition_penalty':1.0},
             'rubric_note':'Natural: simple lexical rubrics on familiar content with different wording. '
                           'Arithmetic: final number only. Other tasks: exact match. No general intelligence score.'}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return value


def selection_score(result):
    return sum(s['correct']*(2 if group=='natural' else 1) for group,s in result['scores'].items())


def eligible(candidate, original, champion, candidate_loss, original_loss):
    floor = all(candidate['scores'][g]['correct'] >= original['scores'][g]['correct']-1
                for g in ('context','reading','format'))
    natural = candidate['scores']['natural']['correct'] >= original['scores']['natural']['correct']
    improved = selection_score(candidate) >= selection_score(champion)+2
    return floor and natural and improved and candidate_loss < original_loss
