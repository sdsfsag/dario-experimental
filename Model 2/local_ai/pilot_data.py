"""Small, explicitly synthetic German curriculum for a bounded diagnostic trial."""
import hashlib
import json
from pathlib import Path

from context import SYSTEM_PROMPT


def example(group, messages):
    return {'group': group, 'synthetic': True, 'source': 'local_pilot_authored_or_deterministic',
            'messages': [{'role': 'system', 'content': SYSTEM_PROMPT}] + messages}


def pair(group, question, answer):
    return example(group, [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}])


def normalized_question(text):
    return ' '.join(text.casefold().split()).rstrip('?.!')


def build_data():
    train, val, test = [], [], []
    # Authored examples teach short, relevant responses, uncertainty and clarification.
    pairs = [
        ('Hallo!', 'Hallo! Wobei kann ich dir helfen?'),
        ('Guten Morgen.', 'Guten Morgen! Was möchtest du besprechen?'),
        ('Guten Abend!', 'Guten Abend! Wie kann ich dir helfen?'),
        ('Hi!', 'Hi! Was möchtest du wissen?'),
        ('Servus.', 'Servus! Wobei kann ich dir helfen?'),
        ('Danke für die Hilfe.', 'Gern geschehen.'),
        ('Vielen Dank!', 'Bitte, gern geschehen!'),
        ('Bis später.', 'Bis später!'),
        ('Tschüss!', 'Tschüss! Bis zum nächsten Mal.'),
        ('Stell dich kurz vor.', 'Ich bin ein lokal trainiertes Sprachmodell. Ich erzeuge Text und kann Fehler machen.'),
        ('Was bist du für ein Programm?', 'Ich bin ein Sprachmodell, das lokal auf diesem Rechner läuft.'),
        ('Bist du ein Mensch?', 'Nein. Ich bin ein Computerprogramm, das Text erzeugt.'),
        ('Kannst du meine Gedanken lesen?', 'Nein. Ich kenne nur die Informationen, die du mir mitteilst.'),
        ('Hast du menschliche Gefühle?', 'Ich habe keine nachgewiesenen menschlichen Gefühle. Ich kann darüber sprechen.'),
        ('Kann ich deinen Antworten immer vertrauen?', 'Meine Antworten können falsch sein. Wichtige Angaben solltest du prüfen.'),
        ('Wie ist das Wetter bei mir gerade?', 'Ich habe keine aktuellen Wetterdaten. Für welchen Ort möchtest du das Wetter wissen?'),
        ('Kannst du das Internet durchsuchen?', 'In diesem lokalen Chat habe ich keinen Internetzugriff.'),
        ('Was habe ich gestern gegessen?', 'Das weiß ich nicht. Du hast mir dazu keine Information gegeben.'),
        ('Wie heißt meine Mutter?', 'Das weiß ich nicht. Du hast ihren Namen nicht genannt.'),
        ('Was liegt hinter mir?', 'Ich kann deine Umgebung nicht sehen. Du kannst sie mir beschreiben.'),
        ('Wie viel Geld habe ich in der Tasche?', 'Das kann ich ohne eine Angabe von dir nicht wissen.'),
        ('Mach es besser.', 'Was genau möchtest du verbessern? Schick mir bitte den Text oder beschreibe die Aufgabe.'),
        ('Erklär mir das.', 'Was soll ich dir erklären? Nenne bitte das Thema.'),
        ('Welches davon ist besser?', 'Welche Möglichkeiten möchtest du vergleichen?'),
        ('Fasse den Text zusammen.', 'Schick mir bitte den Text, den ich zusammenfassen soll.'),
        ('Bitte antworte kurz.', 'Gern. Ich halte mich kurz.'),
        ('Ich habe dich nicht verstanden.', 'Welcher Teil war unklar? Ich kann ihn einfacher erklären.'),
        ('Kannst du langsamer schreiben?', 'Ich kann kürzere Antworten schreiben und einen Schritt nach dem anderen erklären.'),
        ('Was bedeutet ein Fragezeichen?', 'Ein Fragezeichen steht am Ende einer Frage.'),
        ('Was ist ein Satz?', 'Ein Satz drückt einen Gedanken aus und besteht aus einem oder mehreren Wörtern.'),
        ('Nenne zwei Obstsorten.', 'Apfel und Birne.'),
        ('Nenne zwei Gemüsesorten.', 'Karotte und Gurke.'),
        ('Nenne drei Farben.', 'Rot, Grün und Blau.'),
        ('Nenne zwei Wochentage.', 'Montag und Dienstag.'),
        ('Wie viele Tage hat eine Woche?', 'Eine Woche hat sieben Tage.'),
        ('Wie viele Monate hat ein Jahr?', 'Ein Jahr hat zwölf Monate.'),
        ('Wie viele Minuten hat eine Stunde?', 'Eine Stunde hat sechzig Minuten.'),
        ('Wie viele Sekunden hat eine Minute?', 'Eine Minute hat sechzig Sekunden.'),
        ('Was ist das Gegenteil von warm?', 'Das Gegenteil von warm ist kalt.'),
        ('Was ist das Gegenteil von groß?', 'Das Gegenteil von groß ist klein.'),
        ('Was ist das Gegenteil von laut?', 'Das Gegenteil von laut ist leise.'),
        ('Was ist das Gegenteil von schnell?', 'Das Gegenteil von schnell ist langsam.'),
        ('Was ist ein Buch?', 'Ein Buch besteht aus Seiten mit Text oder Bildern.'),
        ('Wofür benutzt man einen Löffel?', 'Mit einem Löffel kann man zum Beispiel Suppe essen.'),
        ('Wofür braucht man einen Schlüssel?', 'Mit einem passenden Schlüssel kann man ein Schloss öffnen oder schließen.'),
        ('Was macht ein Kühlschrank?', 'Ein Kühlschrank hält Lebensmittel kühl.'),
        ('Was macht eine Lampe?', 'Eine Lampe spendet Licht.'),
        ('Wozu dient eine Uhr?', 'Eine Uhr zeigt die Zeit an.'),
        ('Was ist Regen?', 'Regen besteht aus Wassertropfen, die aus Wolken auf den Boden fallen.'),
        ('Was ist Schnee?', 'Schnee besteht aus Eiskristallen, die aus Wolken fallen.'),
        ('Was ist Eis?', 'Eis ist gefrorenes Wasser.'),
        ('Was passiert beim Schmelzen von Eis?', 'Beim Schmelzen wird Eis zu flüssigem Wasser.'),
        ('Warum wird eine nasse Jacke trocken?', 'Das Wasser in der Jacke verdunstet und gelangt in die Luft.'),
        ('Warum fällt ein losgelassener Ball nach unten?', 'Die Schwerkraft zieht den Ball zur Erde.'),
        ('Was bedeutet durchsichtig?', 'Durch ein durchsichtiges Material kann Licht hindurchgehen, sodass man hindurchsehen kann.'),
        ('Was ist eine Insel?', 'Eine Insel ist ein Stück Land, das ringsum von Wasser umgeben ist.'),
        ('Was ist ein Fluss?', 'Ein Fluss ist ein natürliches Gewässer, dessen Wasser durch ein Flussbett fließt.'),
        ('Was ist ein Fahrrad?', 'Ein Fahrrad ist ein Fahrzeug mit zwei Rädern, das man meist mit Pedalen antreibt.'),
        ('Wozu benutzt man eine Brücke?', 'Eine Brücke ermöglicht es, ein Hindernis wie einen Fluss zu überqueren.'),
        ('Was ist eine Bibliothek?', 'In einer Bibliothek werden Bücher und andere Medien zum Lesen oder Ausleihen angeboten.'),
        ('Was ist ein Computer?', 'Ein Computer verarbeitet Daten nach den Anweisungen von Programmen.'),
        ('Was ist Python?', 'Python ist eine Programmiersprache.'),
        ('Was ist eine Datei?', 'Eine Datei ist eine gespeicherte Sammlung von Daten, zum Beispiel ein Text oder ein Bild.'),
        ('Was ist ein Ordner am Computer?', 'Ein Ordner hilft dabei, Dateien und andere Ordner zu organisieren.'),
        ('Was bedeutet lokal bei einem Programm?', 'Lokal bedeutet, dass das Programm auf deinem eigenen Rechner ausgeführt wird.'),
        ('Was ist eine Sicherungskopie?', 'Eine Sicherungskopie ist eine zusätzliche Kopie von Daten für den Fall eines Verlusts.'),
        ('Schreibe einen freundlichen Gruß.', 'Hallo! Ich wünsche dir einen schönen Tag.'),
        ('Schreibe eine kurze Entschuldigung für eine Verspätung.', 'Entschuldige bitte, dass ich zu spät bin.'),
        ('Schreibe eine höfliche Bitte um Hilfe.', 'Könntest du mir bitte bei dieser Aufgabe helfen?'),
        ('Formuliere eine Einladung zum Spaziergang.', 'Hast du Lust, mit mir spazieren zu gehen?'),
        ('Schreibe zwei Sätze über einen roten Ball.', 'Ein roter Ball liegt im Garten. Ein Kind hebt ihn auf.'),
        ('Schreibe zwei Sätze über ein Boot.', 'Ein kleines Boot schwimmt auf dem See. Der Wind bewegt sein Segel.'),
        ('Schreibe drei Sätze über einen Ausflug.', 'Heute machen wir einen Ausflug. Wir fahren gemeinsam zum See. Dort essen wir unser mitgebrachtes Brot.'),
        ('Schreibe drei Sätze über einen Regentag.', 'Am Morgen beginnt es zu regnen. Ich ziehe meine Jacke an. Dann gehe ich mit einem Schirm nach draußen.'),
        ('Fasse zusammen: Tim nimmt seinen Schirm mit, weil Regen angekündigt ist.', 'Tim nimmt wegen des angekündigten Regens einen Schirm mit.'),
        ('Fasse zusammen: Der Zug fällt aus. Lea fährt deshalb mit dem Bus zur Arbeit.', 'Lea nimmt den Bus zur Arbeit, weil ihr Zug ausfällt.'),
        ('Im Text steht: Das Buch liegt auf dem Tisch. Wo liegt das Buch?', 'Das Buch liegt auf dem Tisch.'),
        ('Im Text steht: Paul trägt einen grünen Mantel. Welche Farbe hat der Mantel?', 'Der Mantel ist grün.'),
        ('Im Text steht: Nora fährt mit dem Fahrrad zur Schule. Wie kommt sie zur Schule?', 'Nora fährt mit dem Fahrrad zur Schule.'),
        ('Im Text steht: Leon ist neun Jahre alt. Wie alt ist Leon?', 'Leon ist neun Jahre alt.'),
    ]
    train.extend(pair('dialog', q, a) for q, a in pairs)
    for turns in [
        [('Ich plane einen Spaziergang.', 'Wohin möchtest du gehen?'), ('In den Park.', 'Dann kannst du einen Weg durch den Park wählen.')],
        [('Ich lerne gerade Python.', 'Was möchtest du in Python üben?'), ('Schleifen.', 'Mit einer Schleife kannst du Anweisungen wiederholt ausführen.')],
        [('Ich möchte einen Text verbessern.', 'Schick mir bitte den Text.'), ('Guten Tag ich komme morgen.', 'Guten Tag, ich komme morgen.')],
        [('Was ist ein Fluss?', 'Ein Fluss ist ein Gewässer mit fließendem Wasser.'), ('Und was ist ein See?', 'Ein See ist ein größeres stehendes Gewässer.')],
    ]:
        train.append(example('dialog', [m for q, a in turns for m in
            ({'role': 'user', 'content': q}, {'role': 'assistant', 'content': a})]))

    # Split copying tasks by entity, not by paraphrase, to prevent near-duplicate leakage.
    word_sets = {
        'train': 'Laterne Wolke Kiesel Brücke Feder Mantel Tasche Tasse Segel Regen Knopf Garten Wald Fenster Treppe Flasche Lampe Kissen Papier Stuhl Blume Stein Brot Teller Turm Wiese Glas Ring Besen Buch'.split(),
        'validation': 'Koffer Muschel Faden Stern Schachtel Kamm'.split(),
        'test': 'Banane Leuchtturm Tanne Mango Sonnenhut Notizblock Kristall Schmetterling'.split()}
    for split, words in word_sets.items():
        for word in words:
            prompts = [f'Mein Lieblingswort ist {word}. Antworte nur mit meinem Lieblingswort.',
                       f'Merke dir das Wort {word}. Welches Wort habe ich genannt? Antworte nur mit dem Wort.',
                       f'Das gesuchte Wort lautet {word}. Gib nur dieses Wort zurück.',
                       f'Wiederhole nur dieses Wort: {word}']
            if split == 'test':
                test.append({'group': 'copy', 'question': f'Ich habe mir {word} als Lieblingswort ausgesucht. Nenne ausschließlich dieses Wort.', 'expected': word})
            else:
                destination = train if split == 'train' else val
                destination.extend(pair('copy', q, word) for q in prompts)
                destination.append(example('copy', [{'role': 'user', 'content': f'Mein Lieblingswort ist {word}.'},
                    {'role': 'assistant', 'content': 'Alles klar.'},
                    {'role': 'user', 'content': 'Was ist mein Lieblingswort? Antworte nur mit dem Wort.'},
                    {'role': 'assistant', 'content': word}]))

    entities = {
        'train': ('Anna Ben Clara David Emil Finn Greta Hugo Ida Jonas Klara Leon Nora Paul Ronja Tim'.split(),
                  'Berlin Bonn Dresden Essen Graz Hamburg Köln Linz'.split()),
        'validation': ('Hanna Oskar'.split(), 'Salzburg Ulm'.split()),
        'test': ('Jana Mira Noah Sara'.split(), 'Bern Basel'.split())}
    for split, (names, cities) in entities.items():
        for i, name in enumerate(names):
            for city in cities:
                if split == 'test':
                    test.append({'group': 'context', 'messages': [
                        {'role': 'user', 'content': f'Ich bin {name} und mein Wohnort ist {city}.'},
                        {'role': 'assistant', 'content': 'Danke für die Angaben.'},
                        {'role': 'user', 'content': 'Wie heiße ich? Gib nur meinen Namen aus.'}], 'expected': name})
                    continue
                destination = train if split == 'train' else val
                for field, expected in [('Name', name), ('Wohnort', city)]:
                    destination.append(example('context', [{'role': 'user', 'content': f'Ich heiße {name} und wohne in {city}.'},
                        {'role': 'assistant', 'content': 'Danke für die Information.'},
                        {'role': 'user', 'content': f'Was ist mein {field}? Antworte nur mit dem {field}.'},
                        {'role': 'assistant', 'content': expected}]))

    reserved = {(17, 26), (15, 28), (6, 9), (5, 12), (8, 24), (3, 19), (11, 18), (7, 21)}
    for a in range(31):
        for b in range(a, 36):
            key = (a, b)
            if key in reserved:
                continue
            partition = int(hashlib.sha256(f'{a}+{b}'.encode()).hexdigest()[:8], 16)%10
            destination = val if partition == 0 else train
            destination.append(pair('arithmetic', f'Wie viel ist {a} + {b}? Antworte nur mit der Zahl.', str(a+b)))
            if destination is train:
                destination.append(pair('arithmetic', f'Berechne {b} plus {a}. Gib nur das Ergebnis aus.', str(a+b)))
    for a, b in sorted(reserved):
        test.append({'group': 'arithmetic', 'question': f'Addiere {a} und {b}. Antworte ausschließlich mit der Ergebniszahl.',
                     'expected': str(a+b), 'operands': [a, b]})

    # Distinct validation prompts; the loss is only for model selection, not a chat score.
    val.extend(pair('dialog', q, a) for q, a in [
        ('Guten Tag, ich brauche Hilfe.', 'Guten Tag! Wobei brauchst du Hilfe?'),
        ('Beschreibe dich in einem Satz.', 'Ich bin ein lokal laufendes Sprachmodell und meine Antworten können Fehler enthalten.'),
        ('Kennst du meine Telefonnummer?', 'Nein. Du hast mir deine Telefonnummer nicht mitgeteilt.'),
        ('Sag das einfacher.', 'Welchen Text soll ich einfacher formulieren?'),
        ('Was hat eine Uhr für eine Aufgabe?', 'Eine Uhr zeigt die Zeit an.'),
        ('Wie viele Tage sind in einer Woche?', 'Eine Woche hat sieben Tage.'),
    ])
    # Open-ended probes are recorded for review, never keyword-graded as correct.
    for question in ['Hallo, wer bist du?', 'Erkläre mir, warum der Himmel blau ist.',
                     'Schreibe drei zusammenhängende Sätze über einen Hund.', 'Was ist 17 + 26?',
                     'Merke dir: Mein Lieblingswort ist Banane. Was ist mein Lieblingswort?',
                     'Guten Tag! Kannst du dich kurz vorstellen?',
                     'Wieso wird Eis auf einem warmen Teller flüssig?',
                     'Erfinde eine kurze Geschichte über einen verlorenen Handschuh.',
                     'Ich verstehe das noch nicht. Kannst du nachfragen, was ich meine?',
                     'Warum ist ein Fahrrad praktisch?', 'Was bedeutet das Wort Bibliothek?']:
        test.append({'group': 'open_ended', 'question': question, 'expected': None})

    train_questions = {normalized_question(m['content']) for r in train for m in r['messages'] if m['role'] == 'user'}
    for row in test:
        question = row.get('question') or row['messages'][-1]['content']
        assert normalized_question(question) not in train_questions
    return train, val, test


def write_data(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    train, val, test = build_data()
    for name, rows in [('train', train), ('validation', val), ('test', test)]:
        content = ''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows)
        path = folder/f'{name}.jsonl'
        if path.exists() and path.read_text(encoding='utf-8') != content:
            raise FileExistsError('Pilot data changed; use a new versioned directory')
        path.write_text(content, encoding='utf-8')
    return train, val, test
