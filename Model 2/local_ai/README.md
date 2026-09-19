# Lokales Sprachmodell

Aktuell: mehrere Lernversuche innerhalb des gemeinsamen Zeitbudgets:
`Training_4_Stunden.cmd`, danach `Chat_Neues_Modell.cmd`.
Zeitgrenze, Datenmischung und Ergebnisprüfung: [VERBESSERUNG_V5.md](VERBESSERUNG_V5.md).
Der vorherige Dialoglauf ist in [DIALOG_V4.md](DIALOG_V4.md) dokumentiert.
Der vorherige V3-Lauf ist in [VIER_STUNDEN_TRAINING.md](VIER_STUNDEN_TRAINING.md) dokumentiert.

Aktueller Qualitätsstand: Auch nach rund 1,7 Milliarden verarbeiteten Tokens sind die
geprüften Chatantworten nicht zuverlässig. Ein kurzer zusätzlicher Dialog-Probelauf
ergab keine Verbesserung auf 24 unbekannten Aufgaben und wurde nicht als Standard
übernommen. Ergebnisse und weiteres Vorgehen: [DIALOG_PROBELAUF.md](DIALOG_PROBELAUF.md).

Fuer das vorbereitete Weitertraining auf insgesamt etwa 1,7 Milliarden
verarbeitete Tokens: `Training_9_Stunden.cmd` im uebergeordneten Ordner.
Das startet beim Doppelklick einen neuen, fortsetzbaren Lauf aus den vorhandenen
Gewichten mit bis zu neun Stunden pro Aufruf. Anleitung: `MORGEN_TRAINIEREN.md`.

Der echte Trainingsablauf ist jetzt in `learn.py` eingerichtet. `Training_starten.cmd` im uebergeordneten Ordner startet oder setzt einen begrenzten Lauf fort; nach Anlage von `real_v2` wird dieser neue Lauf fortgesetzt. `Training_stoppen.cmd` fordert einen Stopp mit Checkpoint an. `Chat_starten.cmd` wartet auf den ausgewaehlten Lauf und laedt danach dessen SFT-Checkpoint. Details zum ersten Lauf: `ECHTES_TRAINING.md`.

Eigenes Decoder-only-Modell aus zufälligen Gewichten: RoPE, RMSNorm, SwiGLU, GQA, geteilte Embedding-/LM-Head-Gewichte, SDPA, KV-Cache. Keine fertigen Modellgewichte, kein AutoModel, keine Netzwerkanfragen beim Training oder Chat.

Die Pipeline ist implementiert und getestet. Die Beispieldaten dienen ausschließlich dem Funktionstest. Sinnvolle Gespräche erfordern ein ausreichend vortrainiertes Modell und gute SFT-Daten; wenige Beispiele oder der Overfitting-Test reichen dafür nicht aus. Es wird kein fertig sprachfähiges Modell mitgeliefert.

## Installation

Python 3.12 empfohlen. PowerShell im Projektordner:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python hardware.py --probe
python -m pytest -q
```

Auf diesem Rechner ist außerdem bereits eine geprüfte Umgebung verfügbar:

```powershell
& '..\..\work\.venv\Scripts\Activate.ps1'
```

Ohne Aktivierung kann jeder Befehl mit `.\.venv\Scripts\python.exe` beziehungsweise `..\..\work\.venv\Scripts\python.exe` ausgeführt werden. Für CPU-only zuerst PyTorch mit dem Index `https://download.pytorch.org/whl/cpu` installieren. [Offizielle PyTorch-Installation](https://pytorch.org/get-started/locally/).

## Eigene Daten trainieren

Verzeichnisse und Dateinamen in den Befehlen sind selbst anzulegen. UTF-8-Texte: `.txt`, Absätze durch Leerzeilen getrennt. Alternativ JSONL mit `{"text":"..."}`. Code-Einrückungen bleiben erhalten; Textdateien werden beim Einlesen an Leerzeilen oder etwa 64.000 Zeichen getrennt. JSONL erhält Dokumentgrenzen genauer.

Trainings- und Validierungsdateien getrennt halten. Den Tokenizer ausschließlich auf Trainingsdaten trainieren:

```powershell
python tokenizer.py --input data/train data/sft_train.jsonl --vocab-size 32768
python data.py --input data/train --validation data/validation --output prepared/pretrain
python train.py --config configs/balanced.json --data prepared/pretrain
```

Ohne `--validation` erfolgt ein deterministischer Dokument-Split mit 5 Prozent Validierung. Exakte Duplikate werden entfernt, bei separater Validierung auch über beide Teilmengen hinweg. Varianten und Fast-Duplikate müssen bereits im Ausgangskorpus bereinigt werden. Beide Teilmengen benötigen mindestens `context_length + 1` Tokens.

Die eigentliche Vokabulargröße wird aus dem selbst trainierten Tokenizer übernommen. Ein kleiner Korpus kann weniger BPE-Einträge als angefordert erzeugen. Die Bibliothek `tokenizers` beschleunigt das BPE-Training; es wird kein bestehender Tokenizer geladen.

Vor einem neuen Pretraining prüft ein separates, zufällig initialisiertes kleines Modell dieselbe Optimierungspipeline durch absichtliches Overfitting. Bei unzureichender Loss-Abnahme wird abgebrochen.

```powershell
python train.py --config configs/balanced.json --overfit-only
python train.py --resume runs/pretrain/last.pt --data prepared/pretrain
```

Die JSON-Konfiguration enthält Modell- und Trainingsparameter. `steps` zählt Optimizer-Schritte. Batchgröße mal Gradient Accumulation mal Kontextlänge ergibt die verarbeiteten Tokens pro Schritt. Das balanced-Preset verarbeitet 16.384 Tokens pro Schritt. Die voreingestellten 10.000 Schritte sind ein Ausgangspunkt, kein Qualitätsversprechen für ein von Grund auf trainiertes Modell. Passe die Laufzeit an Korpusgröße und Validation Loss an.

`--stop-after N` beendet nach dem absoluten Optimizer-Schritt N, ohne den geplanten LR-Verlauf zu verändern. Resume übernimmt die gespeicherte Konfiguration; Scheduler, Optimizer, GradScaler, Zufallszustände und Batch-Sampling werden fortgesetzt. Geänderte Daten, Tokenizer, Geräteklasse oder Präzision werden abgewiesen. CUDA-Kernels können hardwareabhängige numerische Unterschiede erzeugen.

`last.pt` ist zum Fortsetzen, `best.pt` enthält den besten gemessenen Validation Loss. Beide enthalten den Trainingszustand und den Tokenizer. Speichern erfolgt atomar. Bei Abbruch während eines Schritts bleibt der letzte vollständige Checkpoint erhalten. Checkpoints benötigen zusammen mit Optimizerzuständen erheblich mehr Platz als reine Gewichte.

## Instruction-Tuning

Eine JSONL-Zeile enthält ein Gespräch. Optional zuerst `system`, danach abwechselnd `user` und `assistant`; das Gespräch endet mit `assistant`:

```json
{"messages":[{"role":"system","content":"Antworte sachlich."},{"role":"user","content":"Was ist 2 + 2?"},{"role":"assistant","content":"2 + 2 = 4."}]}
```

```powershell
python data.py --sft --input data/sft_train.jsonl --context-length 1024 --output prepared/sft_train
python data.py --sft --input data/sft_validation.jsonl --context-length 1024 --output prepared/sft_val
python sft_train.py --pretrained runs/pretrain/best.pt --steps 2000 --lr 0.00002
python sft_train.py --resume runs/sft/last.pt
```

`--pretrained` meint ausschließlich deinen selbst trainierten Checkpoint. Kontextlänge bei der SFT-Aufbereitung muss zum Checkpoint passen. Train/Validation dürfen keine identischen Gespräche enthalten. Lange Gespräche erhalten überlappende Fenster; jeder Assistant-Zieltoken wird einmal bewertet. System-, User- und Padding-Tokens tragen nicht zum Loss bei. Accumulation gewichtet nach der tatsächlichen Zahl bewerteter Tokens.

Gute Daten enthalten mehrstufige Gespräche, Korrekturen, begründete Unsicherheit und Antworten auf tatsächlich bereitgestellte Erinnerungen. Nutze auch Beispiele mit dem Systemtext aus `context.py` und zitierten Erinnerungen. Das bloße Vorhandensein einer Regel im Systemtext garantiert ihre Befolgung nicht.

## Terminal-Chat

```powershell
python chat.py
python chat.py --checkpoint runs/sft/best.pt --temperature 0.8 --top-k 40 --top-p 0.95 --repetition-penalty 1.1 --max-new-tokens 256 --seed 42
```

Ohne Angabe wird das Ausgabelimit auf maximal ein Viertel des Kontextfensters, höchstens 256 Tokens, gesetzt. `--temperature 0` generiert greedy. Ausgabe streamt mit Unicode-Decoding. Eingabe und Ausgabe teilen das Kontextfenster; eine einzelne zu lange Nutzernachricht wird mit einer konkreten Fehlermeldung abgewiesen.

Kommandos:

- `/remember TEXT`: explizit speichern.
- `/memories`: gespeicherte aktive Einträge mit IDs anzeigen.
- `/forget ID`: einen Eintrag löschen, einschließlich seines Suchindex-Eintrags.
- `/summary`: Nutzer-Auszüge des aktuellen Gesprächs als Zusammenfassung speichern.
- `/reset`: flüchtigen Gesprächskontext leeren; persistente Erinnerungen bleiben erhalten.
- `/quit`: beenden.

SQLite speichert ausgewählte direkte Aussagen wie Name, Wohnort, Präferenzen, Entscheidungen, Ereignisse und ausdrücklich benannte Themen. Die Erkennung ist regelbasiert und unvollständig. Neue Angaben zu Name, Wohnort oder Tätigkeit ersetzen aktive Angaben desselben Typs; ältere Einträge bleiben für Änderungsfragen recherchierbar. Allgemeine Präferenzwidersprüche werden nicht automatisch aufgelöst. `/forget` löscht jeweils die angegebene Version.

FTS5 sucht passende Einträge; Relevanz wird aus Suchrang, Wichtigkeit und Alter berechnet. Profilfragen und Fragen nach gestern werden gesondert berücksichtigt. Dies ist lexikalische Suche, keine semantische Embedding-Suche. Die Methode `MemoryStore.search()` ist die Austauschstelle für eine spätere Suchimplementierung.

Es gibt kein automatisches vollständiges Gesprächsprotokoll. `--no-auto-memory` deaktiviert die automatische Faktenspeicherung. `artifacts/memory.sqlite` bleibt nach Neustarts erhalten. Zusammenfassungen bestehen aus kenntlich gemachten Originalauszügen; sie erfinden keine neuen Fakten. Bei langen Gesprächen werden alte Nachrichten im Arbeitsspeicher verdichtet und aktuelle Nachrichten vollständig behalten. Die begrenzte Zusammenfassung und die Suche können Details verlieren. Assistentenantworten werden nicht automatisch als wahre Nutzerfakten gespeichert.

## Hardware-Presets

Gemessen auf RTX 5070 Ti, 15,92 GiB VRAM, 31,06 GiB RAM; CUDA 12.8, bf16 und fp16 verfügbar. `hardware.py` erkennt die aktuelle freie Speichermenge erneut. Keine automatische Installation, Downloads oder Änderung anderer GPU-Anwendungen.

| Preset | Parameter bei 32.768 Tokens | Kontext | Microbatch | Gemessener PyTorch-Peak |
|---|---:|---:|---:|---:|
| test | 4.563.584 | 256 | 1 | 0,139 GiB |
| balanced | 100.682.496 | 1.024 | 2 | 1,721 GiB |
| maximum_practical | 213.943.296 | 2.048 | 1 | 3,605 GiB |

Der Peak umfasst einen Forward-/Backward-/AdamW-Schritt, nicht fremde GPU-Prozesse oder den vollständigen Treiberverbrauch. Die beiden größeren Presets verwenden Gradient Checkpointing. `maximum_practical` begrenzt die Größe bewusst zugunsten der Trainingsdauer. Bei verändertem Vokabular oder Kontext erneut messen. bf16 wird bevorzugt; fp16 nutzt GradScaler, CPU nutzt fp32.

## Dateien und Tests

`model.py`: Architektur und Generierung. `train.py`: gemeinsame Trainingslogik. `sft_train.py`: zweite Trainingsstufe. `tokenizer.py` und `data.py`: Tokenisierung/Aufbereitung. `memory.py`, `context.py`, `chat.py`: Erinnerung und Terminal. `hardware.py` und `config.py`: Erkennung und Einstellungen.

`python -m pytest -q` prüft unter anderem Unicode-Roundtrips, Kausalität, Cache-Gleichheit, Gradienten, Overfitting, Resume, SFT-Maskierung, Memory-Persistenz, Kontextbudget sowie den gesamten CLI-Ablauf. Details der tatsächlich ausgeführten Tests stehen in `TEST_REPORT.md`. Logs enthalten echte Loss-Werte; Tokens/s zählt verarbeitete Eingabetokens, beim SFT einschließlich Padding.
