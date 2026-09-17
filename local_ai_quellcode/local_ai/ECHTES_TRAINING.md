# Echte Trainingsdaten und begrenzter Trainingslauf

Der erste Lauf verwendet das selbst implementierte Modell aus zufaelligen Gewichten. Es werden keine fertigen Sprachmodellgewichte heruntergeladen.

Im uebergeordneten Ordner:

- `Training_starten.cmd`: Aufbereitung, GPU-Messung, Pretraining, SFT und Ausgabe fester Testfragen. Zunaechst auf etwa zwei Stunden fuer die Trainingsphasen begrenzt. Datenaufbereitung liegt vor diesem Zeitbudget. Validierung und das sichere Schreiben des letzten Checkpoints koennen den Grenzzeitpunkt geringfuegig ueberschreiten.
- `Training_stoppen.cmd`: Nach dem laufenden Optimizer-Schritt speichern und anhalten. Waehrend der Aufbereitung wird der aktuelle Verarbeitungsschritt zuerst abgeschlossen.
- `Chat_starten.cmd`: Zeigt waehrend des Trainings automatisch den Fortschritt der aktuellen Phase. Nach Abschluss des gesamten Laufs startet der Chat mit dem eigenen SFT-Checkpoint. Das Fenster kann offen bleiben; Strg+C beendet nur das Warten. Bei pausiertem Training zuerst mit `Training_starten.cmd` fortsetzen. Die Anzeige ist kein Nachweis guter Sprachqualitaet. Es gibt keinen automatischen Rueckfall auf die alten Testgewichte.

Fortsetzen mit `Training_starten.cmd` uebernimmt vorhandene Checkpoints und beendet die noch offenen Schritte des gespeicherten Trainingsplans. Ein bereits vollstaendig absolvierter Plan wird dadurch nicht automatisch verlaengert. Ein neuer groesserer Trainingszyklus braucht einen neuen Plan; die bestehende LR-Kurve wird beim Resume nicht unbemerkt geaendert.

Andere Laufzeit fuer einen neuen Lauf:

```powershell
..\..\work\.venv\Scripts\python.exe -u learn.py --hours 8 --resume
```

`--hours` begrenzt diesen Aufruf. Wenn bereits ein Plan existiert, bleiben dessen geplante Schritte und Lernraten erhalten. Datenaufbereitung allein: `learn.py --prepare-only`. Neue GPU-Messung: `learn.py --benchmark-only`.

## Daten

- [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), Snapshot 20231101: drei deutsche und ein englischer Teil. Die Dataset-Karte nennt CC-BY-SA-3.0/GFDL. Artikel-URL, ID und Titel bleiben in den JSONL-Dateien zur Quellenzuordnung erhalten. Enthaelt enzyklopaedische Texte, keine aktuelle Wissensgarantie.
- [OpenAssistant OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2): gepruefte deutsche und englische Dialoge, Apache-2.0. Nur geeignete Bewertungen, hoechstbewertete Assistentenzweige und vollstaendige Elternketten; ein laengerer Dialog pro normalisierter Ausgangsfrage.
- [Alpaca GPT4 Deutsch](https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-deutsch): veroeffentlichte synthetische deutsche Instruktionspaare; Dataset-Karte nennt Apache-2.0. Diese Texte sind maschinell erzeugt und koennen sachliche oder sprachliche Fehler enthalten. Sie sind mit `synthetic: true` gekennzeichnet und werden nicht als uebernommenes Modell verwendet.

Der Import entfernt leere/sehr kurze Eintraege, problematische Formatierung, bestimmte falsche Selbstbeschreibungen und doppelte Ausgangsfragen. Er ersetzt keine vollstaendige redaktionelle Pruefung. Wikipedia und synthetische Instruktionen allein decken nicht alle Gespraechssituationen ab.

Artikel bleiben beim Split zusammen. Dialoge werden anhand ihrer normalisierten Ausgangsfrage getrennt; Varianten desselben Gespraechs gelangen nicht absichtlich in beide Teilmengen. Exakte Textduplikate werden bei der Tokenaufbereitung nochmals abgeglichen. Nahe semantische Duplikate koennen verbleiben.

Der eigene 32.768-Token-BPE wird auf einer ausgewogenen Stichprobe der Trainingstexte und den Trainingsdialogen erstellt. Keine Validierungsdialoge werden dafuer genutzt. Download-Dateien werden anhand ihrer SHA256-Pruefsumme verifiziert. Repository-Revisionen, Dateigroessen, Lizenzen und Aufbereitungszahlen stehen in `data/real_v1/manifest.json`. Die Original-Downloads liegen im Arbeitsordner `work/data-cache`.

## Verlauf und Ergebnisse

`runs/real_v1/status.json` zeigt die Phase. `pretrain/metrics.jsonl` und `sft/metrics.jsonl` enthalten echte Messwerte. `benchmark.json` vergleicht Microbatchgroessen und Gradient Checkpointing auf der erkannten GPU. `plan.json` enthaelt die gewaehlte Architektur und den Trainingsplan.

Die Gewichtedateien liegen in `runs/real_v1/pretrain` und `runs/real_v1/sft`. `last.pt` setzt den betreffenden Lauf fort; `best.pt` hat den bisher niedrigsten gemessenen Validation Loss. Der Tokenizer liegt in `artifacts/real_v1/tokenizer.json` und ist ausserdem in jedem Checkpoint enthalten.

Nach der SFT-Phase erzeugt das Modell Antworten auf feste Fragen in `quality_samples.json`. Der Status `finished_unreviewed` bedeutet lediglich, dass dieser begrenzte Lauf beendet ist. Die Antworten muessen gelesen und bewertet werden; das Skript vergibt keine erfundene Qualitaetsnote. Zwei Stunden sind ein erster Lernlauf und keine Zusicherung brauchbarer allgemeiner Gespraeche.

Bei einem Hintergrundstart stehen die Terminalausgaben in `runs/real_v1/training.log` und `training-error.log`. Waehrend des Trainings sollte der Rechner eingeschaltet bleiben. Ein abgeschalteter Rechner pausiert die Berechnung; Fortsetzen erfolgt vom letzten gespeicherten Checkpoint.
