# Neuer deutscher Trainingskurs

Dieser Lauf lernt mit den vorhandenen 100.682.496 Parametern weiter.
Ausgangspunkt ist `runs/real_v2/pretrain/best.pt`, der eigene Sprachmodell-Checkpoint
vor dem bisherigen Dialogtraining. Tokenizer und Modellarchitektur bleiben erhalten.
Es werden keine fremden Modellgewichte verwendet.

## Zeit und Bedienung

- `Training_4_Stunden.cmd` startet den neuen Kurs oder setzt einen unterbrochenen Lauf fort.
- Das Zeitfenster beträgt höchstens vier Stunden ab dem ersten Start. Fortsetzen verlängert
  dieses Zeitfenster nicht. Datenaufbereitung erfolgt vorher; abschließendes Speichern und
  Prüfen kann einige Minuten über das Zeitfenster hinaus dauern.
- Bis zu 2 Stunden 45 Minuten Sprachtraining, anschließend Dialogtraining innerhalb
  des Restbudgets. Ein früheres Ende wegen Schrittlimit oder ausbleibender Verbesserung ist möglich.
- `Training_stoppen.cmd` fordert einen Stopp mit Checkpoint an. PC während des Speicherns nicht ausschalten.
- `Training_Live.cmd` öffnet die laufende Konsolenausgabe ohne Neustart des Trainings.
  Schließen dieser reinen Anzeige stoppt das Training nicht.
- `Chat_Neues_Modell.cmd` wartet auf den Abschluss und öffnet dann den neuen experimentellen Chat.
- `Chat_starten.cmd` verwendet weiterhin das bisherige Modell. Beide Versionen bleiben vergleichbar.
- Der neue Chat verwendet eine separate Erinnerungsdatei und speichert Angaben nicht automatisch.
- Während des Trainings verhindert der Prozess den automatischen Windows-Ruhezustand.

Status: `runs/quality_v3/status.json`, Verlauf: `runs/quality_v3/training.log`.
Fehler: `runs/quality_v3/training-error.log`.
Ein laufender Prozess wird mit PID und Erstellungszeit gegen versehentlichen Doppelstart abgesichert.
Python-Aufruf für einen Status: `python quality_train.py --status`.

## Daten

793.661 einfache deutsche Geschichten und 2.272 einfach geschriebene Sachtexte für
das neue Sprachmaterial. Die Eingabebeispiele werden im Sprachtraining gewichtet gemischt:
70 % Geschichten, 5 % einfache Sachtexte und 25 % des bisherigen deutsch/englischen Sprachkorpus.
Trainingssequenzen sind 512 Tokens lang; das Modell behält sein Kontextfenster von 1.024 Tokens.

124.838 erzeugte Dialog- und Aufgabenbeispiele; nach Entfernung von 1.397 identischen
Kontextbeispielen bleiben **123.441 unterschiedliche Trainingsbeispiele**:

| Gruppe | Beispiele | Anteil an gezogenen Beispielen |
|---|---:|---:|
| Gefilterte deutsche Dialoge/Instruktionen | 29.632 | 35 % |
| Fragen zu gegebenen deutschen Texten | 11.169 | 20 % |
| Einfache Begriffserklärungen | 4.544 | 15 % |
| Geschichten sinnvoll fortsetzen | 49.409 | 10 % |
| Wörter/Angaben aus dem Kontext übernehmen | 28.603 | 18 % |
| Kurze grundlegende Dialoge | 84 | 2 % |

Die Aufgaben werden gewichtet zufällig gezogen. Das ist kein vollständiger Durchgang durch
alle Beispiele in fester Reihenfolge. Lange Antworten enthalten mehr überwachte Tokens als kurze;
die Prozentwerte sind Anteile der Beispiele, nicht der Loss. SFT ignoriert System- und Nutzereingaben
als Lernziele, kürzt Auffüllpositionen aus Batches und verwirft Beispiele, die den Kontext überschreiten.
Trainingszähler enthalten wiederholte Texte; SFT-Eingabetokenzähler enthalten verbleibende Auffüllpositionen.

Quellen:

- [TinyStoriesGerman](https://huggingface.co/datasets/fabi2347/TinyStoriesGerman): automatisch
  übersetzte, synthetische Geschichten. CDLA-Sharing-1.0 laut Datenkarte. Die Geschichten enthalten
  Fiktion und teils unnatürliche Übersetzungen; sie sind kein verlässliches Faktenlexikon.
- [Klexikon-Datensatz](https://huggingface.co/datasets/dennlinger/klexikon): einfacher geschriebene
  deutsche Sachtexte, CC-BY-SA-4.0 laut Datenkarte.
- [GermanQuAD](https://huggingface.co/datasets/deepset/germanquad): manuell annotierte deutsche
  Fragen mit belegten Antwortstellen, CC-BY-4.0 laut Datenkarte. Es werden nur Parquet-Daten geladen,
  keine fremden Dataset-Skripte ausgeführt.
- Bereits vorhandene deutsche Daten aus OpenAssistant/OASST2 und Alpaca-GPT4-Deutsch:
  kurze Antworten ausgewählt, erkennbare Wiederholungen, beschädigte Texte und einige unpassende
  Standardantworten entfernt. Das ist eine heuristische Filterung, keine vollständige manuelle Prüfung.
- Lokal erzeugte Kontextaufgaben mit vielen unterschiedlichen Wörtern und mehreren Formulierungen.

Dateiversionen, SHA-256-Prüfsummen, Quellkennungen und genaue Filterzahlen stehen in
`data/quality_v3/manifest.json`. Herkunft der früheren Daten: `data/real_v1/manifest.json`.
Die neuen Quellen wurden vor Verwendung anhand ihrer Download-Prüfsummen überprüft.

## Prüfung und Grenzen

Nach jeweils 1.000 Sprach- bzw. 500 Dialogupdates wird auf einem getrennten, fest ausgewerteten
Validierungssatz geprüft und gespeichert. Nach sechs Blöcken ohne ausreichende Verbesserung
endet die jeweilige Phase. Der beste Checkpoint wird nach Validierungs-Loss ausgewählt.

Während des Laufs werden echte Textproben gespeichert: alle 5.000 Sprach- bzw. 2.000 Dialogupdates.
Die Ausgabe befindet sich in `runs/quality_v3/pretrain/samples_*.json` bzw. `sft/samples_*.json`.
Diese Proben dienen der Durchsicht und wählen keine Gewichte aus.

Vorher-/Nachher-Vergleich: 24 neue Kontextaufgaben und 12 Leseverständnisfragen mit strenger
Antwortprüfung sowie acht offene Fragen zur manuellen Durchsicht. Die Aufgaben werden nicht
in diesem Lauf trainiert und wählen keinen Checkpoint aus. Eine korrekte abweichende Formulierung
kann bei der strengen Prüfung als falsch zählen. Das alte allgemeine Sprachtraining kann verwandte
Wikipedia-Inhalte enthalten haben; deshalb sind dies keine garantiert vollständig unbekannten Fakten.

Ergebnisse nach Abschluss: `runs/quality_v3/quality_results.json`. Der Status heißt bewusst
`finished_unreviewed`: automatische Einzeltests belegen keine allgemeine Gesprächsqualität.
Der Lauf wird nicht automatisch zum Standardchat. Eine bestimmte Qualitätssteigerung wird nicht zugesagt.

## Technische Prüfung

Gesamtsuite nach Einbau der ersten sechs neuen Tests: 57 Tests bestanden.
Danach alle sieben Tests des neuen Ablaufs bestanden, einschließlich eines zusätzlichen
CPU-Integrationstests: Sprachtraining stoppen, exakt fortsetzen, SFT aus den neuen Gewichten starten
und unveränderten ursprünglichen Checkpoint prüfen. Insgesamt wurden 58 unterschiedliche Tests geprüft.

Die Skripte `quality_data.py` und `quality_train.py` enthalten den neuen Ablauf.
Der echte GPU-Lauf wurde am 18.09.2026 um 17:48 Uhr Ortszeit gestartet (Prozess 13304).
Die ersten 100 Updates liefen mit endlicher Loss, etwa 58.800 Eingabetokens/s und
819.200 verarbeiteten Eingabetokens. Die anfängliche Validierungs-Loss auf der neuen Mischung
betrug 3,4530. Daraus folgt noch keine Aussage über spätere Chatqualität.
Die feste Trainingsfrist endet um etwa 21:49 Uhr; früheres Ende ist möglich.
