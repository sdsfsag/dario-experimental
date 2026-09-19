# Weitertraining am 18.09.2026

Um 06:00 Uhr `Training_9_Stunden.cmd` im uebergeordneten Ordner doppelt
anklicken. Das Skript startet sofort beim Anklicken; es ist kein automatischer
Zeitplan eingerichtet. Den bestehenden Chat vorher mit `/quit` beenden und
andere GPU-intensive Programme schliessen. Der Rechner muss eingeschaltet
bleiben. Waehrend dieses Trainings verhindert das Skript unter Windows den
automatischen Ruhezustand; die vorherige Einstellung gilt nach Beendigung
wieder. Manuelles Ausschalten oder manuell ausgeloester Ruhezustand wird
dadurch nicht verhindert. Der Bildschirm darf ausgehen.

## Ziel und Zeitlimit

- Modell: vorhandene 100.682.496 Parameter, gleicher eigener Tokenizer und
  gleiches Kontextfenster von 1.024 Tokens.
- Neuer Lauf: `runs/real_v2`, aus den Gewichten von
  `runs/real_v1/pretrain/best.pt`. Die bisherigen Dateien bleiben erhalten.
- Ausgangspunkt: 263.503.872 verarbeitete Sprachtrainingstokens.
- Weiteres Sprachtraining: 84.677 Optimizer-Schritte mit je 16.384 Tokens.
- Anschliessendes Dialogtraining: 3.000 Schritte mit je 16.384 Eingabetokens.
- Geplante Summe: 1.700.003.840 verarbeitete Tokens. Die leichte Ueberschreitung
  entsteht durch das Speichern vollstaendiger Lernschritte.
- Pro Aufruf: bis zu neun Stunden fuer die Trainingsphasen. Bei Start um
  06:00 Uhr endet der Aufruf ungefaehr bis 15:00 Uhr. Vorbereitung, Validierung,
  Abschluss eines laufenden Schritts und Checkpoint-Speicherung koennen eine
  geringe Zusatzzeit benoetigen. Fuer das Dialogtraining wird Zeit reserviert;
  falls das Sprachtraining nicht rechtzeitig fertig wird, pausiert der Lauf
  bereits etwas vor dem Ende der neun Stunden.

Das Tokenziel und eine feste Laufzeit lassen sich nicht beide garantieren.
Die gemessene Geschwindigkeit macht etwa neun Stunden plausibel; andere
GPU-Nutzung oder ein langsamerer Lauf koennen weiteres Fortsetzen erfordern.

## Nach dem Lauf

`Chat_starten.cmd` zeigt den Stand. Solange der neue Lauf rechnet, zeigt es
Fortschritt und Tokenzaehler. Nach vollstaendigem Abschluss startet es den
neuen Chat. Bei einer Pause denselben `Training_9_Stunden.cmd`-Starter erneut
oeffnen; er uebernimmt den Checkpoint und die verbleibenden Schritte. Dabei
gilt erneut ein Zeitlimit von neun Stunden, der bestehende Plan wird nicht
verlaengert. Auch `Training_starten.cmd` setzt nach Anlage des neuen Plans
diesen Lauf fort.

Zum vorzeitigen Anhalten `Training_stoppen.cmd` verwenden. Der laufende
Schritt wird abgeschlossen und gespeichert. Fortschritt: `runs/real_v2/status.json`,
`pretrain/metrics.jsonl`, `sft/metrics.jsonl` sowie `training.log`.
Fehler stehen in `training-error.log`. Alte Gewichte sind weiter unter
`runs/real_v1` vorhanden. Automatisch verwendeter neuer Chat-Checkpoint:
`runs/real_v2/sft/best.pt`.

## Was der Zaehler bedeutet

Es werden die vorhandenen Daten weiterverwendet, keine neuen Datensaetze
heruntergeladen. Mehrfach gelesene Texte werden mehrfach gezaehlt. Beim
Dialogtraining zaehlen die Eingabetokens inklusive Padding; fuer den
Trainingsfehler werden nur Assistentenantworten bewertet. Der alte Dialoglauf
wird nicht zur neuen Gewichtelinie addiert: Der neue Lauf verzweigt vorher
vom Sprachcheckpoint und erhaelt ein neues Dialogtraining.

`processed_tokens` im Abschlussstatus zaehlt die Ausgangstokens plus alle
Schritte des neuen Laufs. Fuer den Chat wird jeweils ein Checkpoint mit dem
niedrigsten gemessenen Validierungsfehler verwendet. Dieser kann aus einem
frueheren Schritt stammen. `selected_checkpoint_tokens` dokumentiert deshalb
separat die verarbeiteten Tokens der tatsaechlich ausgewaehlten Gewichtelinie.

Der neue Zyklus startet AdamW und eine neue Lernratenkurve mit Warmup bei
maximal 0,0001. Innerhalb dieses Zyklus werden beim Fortsetzen Optimizer,
Lernrate und Zufallszustaende exakt wiederhergestellt. Ausgangsgewichte,
Tokenizer und Daten werden auf Konsistenz geprueft. Der beste Sprachcheckpoint
wird schon vor dem ersten neuen Update gesichert, um einen Rueckfall bei der
Validierung erkennbar zu machen.

## Pruefung und Grenzen

Die CPU-Pruefung mit den echten Dateien wurde am 17.09.2026 bestanden:
`Training_9_Stunden.cmd --check`. Sie kontrolliert Datenpruefsummen, Tokenizer,
Train/Validation-Trennung und das vollstaendige Laden der Ausgangsgewichte,
ohne Training oder einen neuen Lauf anzulegen.

Automatisierte Tests pruefen unter anderem Warmstart, tokenbasierte Planung,
Stopp nach einem Update, Fortsetzen, anschliessendes Dialogtraining,
Quellcheckpoint-Erhaltung und die Auswahl des Chat-Laufs.

Mehr Training ist kein Nachweis brauchbarer Dialogqualitaet. Die nach dem
Dialogtraining erzeugten `quality_samples.json` muessen gelesen werden.
Der Status `finished_unreviewed` bedeutet Abschluss des Plans, keine
bestandene Sprachpruefung. Speicherbedarf des neuen Laufs: grob weitere
5 GB fuer vier Checkpoints, zusaetzlicher Platz fuer temporaeres Speichern.
