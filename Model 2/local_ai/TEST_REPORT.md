# Ausgeführte Prüfung

18.09.2026, neuer Vier-Stunden-Kurs: Gesamtsuite mit 57 Tests bestanden (25,74 Sekunden).
Anschließend sieben gezielte Tests bestanden (2,43 Sekunden), einschließlich eines neuen
CPU-Integrationstests für Stopp, Resume, Übergang zum Dialogtraining und unveränderte
Ausgangsgewichte. Damit sind 58 unterschiedliche Tests geprüft. Details: `VIER_STUNDEN_TRAINING.md`.

Weitertraining am 17.09.2026: Nach Einbau des neuen Zyklus bestanden alle
45 Tests der Gesamtsuite in 23,05 Sekunden. Geprueft wurden auch echter
CPU-Warmstart, Stopp nach einem abgeschlossenen Update, Resume bis zum neuen
Tokenziel, anschliessendes SFT und unveraenderte Ausgangsgewichte. Der neue
Windows-Starter bestand zusaetzlich `--check` mit dem vorhandenen
100,7-Millionen-Modell und allen Datenpruefsummen, ohne einen Trainingslauf
anzulegen. Nach Ergaenzung des Tokenzaehlers wurde die gezielte Chat-Starter-Suite
erneut ausgefuehrt. Der lange Neun-Stunden-Lauf wurde nicht gestartet.

Erweiterung am 17.09.2026: Vollstaendige Suite mit Datenimport und Zeitlimit: 34 Tests bestanden. Nach Ergaenzung von Wiederanlauf und Stopp waehrend der Aufbereitung: fuenf gezielte Tests bestanden, darunter zwei neue. Insgesamt sind damit 36 verschiedene Tests erfolgreich geprueft. Der echte GPU-Benchmark und der neue Overfitting-Test sind im Startbericht `REAL_TRAINING_REPORT.md` dokumentiert. Der nachfolgende urspruengliche Bericht bezieht sich auf die erste Projektversion.

Datum: 17.09.2026. Windows, Python 3.12.14, PyTorch 2.11.0+cu128, tokenizers 0.23.2, NumPy 2.5.3, psutil 7.2.2, pytest 9.1.1.

## Ergebnis

`python -m pytest -q -ra --junitxml=test-results.xml`

32 Tests bestanden, keine übersprungen, Laufzeit 20,67 Sekunden. Maschinenlesbares Ergebnis: `test-results.xml`.

Geprüft: Unicode und Sondertoken-Escaping, Tokenizer-Dateien, Streaming-Decoding, MHA/GQA-Shapes, kausale Maskierung, Forward/Loss/Gradienten, KV-Cache mit einzelnen und mehreren Tokens, Gradient Checkpointing, Sampling, Generierungsgrenzen, Checkpoint/Resume, Dokument-Split und Duplikate, SFT-Loss-Masken und lange Antworten, tokengewichtete Gradient Accumulation, SQLite-Persistenz/Suche/Versionen/Löschen, Datumsabfragen, extraktive Zusammenfassung, Kontextbudget, Memory-Einfügung, CPU-Presets, CPU-/CUDA-Overfitting, bf16 und fp16.

Der CLI-Test führt eigene Tokenizer-Erstellung, Datenaufbereitung, Pretraining, Resume, SFT-Aufbereitung, SFT, SFT-Resume, Chat und Memory-Neustart in einem temporären Verzeichnis aus.

## Gemessener Overfitting-Test

Separater CUDA-Lauf mit einem selbst trainierten 1.024-Token-BPE und einem zufällig initialisierten kleinen Modell:

- Anfangs-Loss: 6,911526.
- End-Loss: 0,010734 nach 40 Schritten.
- Präzision: bf16.
- Verwendet dieselbe `train_update`-Funktion wie Pretraining und SFT.

## Dialoglauf V4 (19.09.2026)

Erweiterung V5: 69 Tests bestanden in 22,92 Sekunden. Geprüft wurden zusätzlich
die gemeinsame feste Zeitgrenze mehrerer Versuche, Schutz der Ausgangsfähigkeiten
bei der Auswahl, exakt erzeugte Rechenwege sowie Stoppen/Fortsetzen eines Versuchs
ohne Änderung des Ausgangsmodells. Der Datenaudit fand 105.514 unterschiedliche
Trainingsdialoge und 5.402 Validierungsdialoge ohne identische Dialoge zwischen den
beiden Mengen. Keine bewertete Entwicklungs-/Abschlussfrage stimmt exakt mit einem
Trainingsprompt überein. Eine unbewertete alte Beobachtungsfrage kommt im Training vor;
sie beeinflusst weder Punktzahl noch Auswahl. Fakten und Formulierungsfamilien können
zwischen Training und Prüfungen verwandt sein.

64 Tests bestanden (`python -m pytest -q`, 26,16 Sekunden). Zusätzlich geprüft:
gleiches Gewicht kurzer/langer Antworten, Padding-Invarianz, gleiche Updates bei
Microbatch-Akkumulation, bitgenaues Fortsetzen der neuen Zielfunktion, sicherer
Stopp, Ablehnung bei geänderter Zielfunktion sowie Fortsetzungsentscheidung anhand
generierter Antworten statt nur Loss. Die bisherigen Trainings-/Chat-Tests bestehen weiter.

## Resume

Der CPU-Test mit Dropout und Gradient Accumulation vergleicht einen durchgehenden Vier-Schritt-Lauf mit zwei Schritten, Speichern, neuem Modell und Fortsetzen. Alle Modellparameter sind anschließend bitgenau gleich. Scheduler- und Batch-RNG-Zustände stimmen ebenfalls überein. Veränderte Trainingsdaten werden beim Resume abgewiesen.

Zusätzlich wurden Pretraining und SFT auf CUDA über separate Terminalprozesse gespeichert und fortgesetzt. Der 256-Token-Testlauf erreichte im Pretraining Schritt 4 und im SFT Schritt 4. Der SFT-Checkpoint ließ sich im Terminal laden und streamte tatsächlich aus den Modellgewichten erzeugte Tokens. Erinnerungen wurden gespeichert und gelesen.

## Hardware

RTX 5070 Ti: 15,92 GiB VRAM. RAM: 31,06 GiB. bf16 und fp16 durch echte GPU-Rechenoperationen mit Backward bestätigt. Alle drei Presets bestanden einen Forward-/Backward-/AdamW-Schritt bei Vokabulargröße 32.768. Gemessene Details stehen in `configs/hardware.json`.

## Aussagegrenzen

Die Prüfungen belegen technische Funktion und Lernfähigkeit auf einem winzigen Datensatz. Sie belegen keine allgemeine Sprach-, Wissens- oder Gesprächsqualität. Der nur wenige Schritte trainierte CLI-Checkpoint erzeugte noch unbrauchbaren Text. Es wurden keine fertigen Modellgewichte genutzt und kein großes Sprachtraining durchgeführt. Dafür wurden keine eigenen umfangreichen Trainingsdaten bereitgestellt.

Die Beispiele sind ausschließlich Testmaterial. Die beiliegenden Presets und die vollständige Pipeline sind für anschließendes Training auf eigenen Korpora vorgesehen. Testgewichte und temporäre Trainingsdaten sind nicht im Projektarchiv enthalten.
