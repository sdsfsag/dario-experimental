# Startbericht des echten Trainings

Erstellt: 2026-09-17T16:34:07.041518+00:00.

Das echte Training wurde gestartet. Dieser Bericht dokumentiert den Start und ist kein Abschluss- oder Qualitaetsnachweis. Der aktuelle Zustand steht in `runs/real_v1/status.json`, der Verlauf in `runs/real_v1/training.log`.

- Eigener BPE: 32.768 Tokens, nur auf Trainingsmaterial erstellt.
- Pretraining: 565,191,850 Trainingstokens, 11,482,058 Validierungstokens.
- Deduplizierte Textdokumente insgesamt: 504,524.
- SFT: 52.637 Trainingsgespraeche, 1.075 Validierungsgespraeche; nach Fensterbildung 53.103 beziehungsweise 1.082 Beispiele.
- Modell: 100.682.496 Parameter, zufaellig initialisiert, Kontext 1.024.
- Geplant: 16,083 Optimizer-Schritte im Pretraining; danach begrenztes SFT.
- Zeitbudget dieses Aufrufs: zwei Stunden fuer die Trainingsphasen, plus geringfuegige Abschlusszeit fuer einen vollstaendigen Checkpoint.
- Neuer Overfitting-Test mit dem echten Tokenizer: Loss von 10,408185 auf 0,014719 nach 40 Schritten, bf16/CUDA.

## Gemessene GPU-Einstellungen

| Microbatch | Gradient Checkpointing | Tokens/s | Peak GiB |
|---|---|---:|---:|
| 4 | False | 42928 | 4.855 |
| 8 | False | 46922 | 8.299 |
| 2 | True | 25068 | 2.065 |

Die schnellste gemessene Einstellung wurde verwendet. Die Messung besteht aus echten Forward-/Backward-/AdamW-Schritten auf den aufbereiteten Trainingsdaten. Sie garantiert keinen identischen Durchsatz waehrend des gesamten Laufs.

Quellen, Lizenzen, Download-Pruefsummen und Filter sind in `data/real_v1/manifest.json` und `ECHTES_TRAINING.md` dokumentiert. Die deutschen Alpaca-Instruktionen sind synthetisch; es wurden ausschliesslich Texte, keine Modellgewichte uebernommen.

Der Chat-Starter verwendet nach dem Dialogtraining den echten Checkpoint. Fuer die Sprachqualitaet ist zunaechst eine Pruefung der erzeugten Antworten erforderlich. Ein sinkender Loss ist kein Nachweis korrekter oder konsistenter Antworten.
