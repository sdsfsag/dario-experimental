# Dialog-Probelauf vom 18. September 2026

Ergebnis: **keine nachgewiesene Verbesserung bei den geprüften neuen Aufgaben**.
Das bestehende Chatmodell bleibt `runs/real_v2/sft/best.pt`.
Seine SHA-256-Prüfsumme war vor und nach dem Versuch identisch.
Es wurde kein neuer mehrstündiger Trainingslauf gestartet.

## Versuch

- Ausgangspunkt: eigener real_v2-SFT-Checkpoint, Schritt 2.900.
- Unverändert 100.682.496 Parameter, eigener Tokenizer, Kontext 1.024.
- 1.666 lokal erstellte synthetische Trainingsbeispiele: 84 Dialoge, 150 Wortaufgaben,
  256 Kontextaufgaben und 1.176 Additionsaufgaben.
- 99 getrennte Validierungsbeispiele. Ein Datum im Systemtext entspricht dem Chatformat.
- Aufgaben wurden gewichtet gezogen: Dialog 45 %, Wörter 25 %, Kontext 15 %, Rechnen 15 %.
- Variable Sequenzlängen mit korrekter Maskierung der Nutzereingaben und Auffüllpositionen.
  Die Modellarchitektur wurde nicht verändert. Antworten stammen ausschließlich aus den Gewichten.
- Höchstens 300 Sekunden oder 1.200 Updates vorgesehen. Nach steigender Validierungs-Loss
  vorzeitig über STOP-Datei angehalten: 780 Updates in 72,08 Sekunden einschließlich
  Zwischenauswertung und Speichern. Vorher-/Nachher-Generierung und Laden kommen zeitlich hinzu.
- 1.141.516 verarbeitete Eingabetokens ohne Auffüllpositionen und 102.910 überwachte
  Antworttokens einschließlich Antwort-Endmarken. Wiederholungen sind mitgezählt.
- Bester Zwischenstand allein nach Validierungs-Loss: Schritt 200.
  Loss auf diesem kleinen Validierungssatz: 3,9886 vorher, 2,4297 bei Schritt 200,
  2,8277 bei Schritt 400 und 3,1977 bei Schritt 600.
  Diese Werte sind nicht direkt mit der Loss auf dem früheren großen Datensatz vergleichbar.

## Unbekannte Aufgaben

Gleiche 35 Prüfaufgaben vor und nach dem Training, jeweils frischer Kontext, ohne persistente
Erinnerungen, Temperatur 0 und Wiederholungsstrafe 1,1. Testantworten wurden weder trainiert
noch zur Auswahl des Checkpoints verwendet. Zahlenpaare einschließlich vertauschter Summanden
sowie die geprüften Wörter und Namen wurden aus den entsprechenden Trainingsaufgaben ausgeschlossen.
Die Formulierungen der Testfragen waren ebenfalls nicht im Training.

| Aufgabe | Vorher richtig | Nachher richtig |
|---|---:|---:|
| Genanntes Wort wiedergeben | 0/8 | 0/8 |
| Namen aus dem Gespräch wiedergeben | 0/8 | 0/8 |
| Ungeübte Additionen | 0/8 | 0/8 |
| Insgesamt automatisch geprüft | **0/24** | **0/24** |

Die Bewertung verlangt die angeforderte einzelne Zahl oder das einzelne Wort, mit tolerierter
Groß-/Kleinschreibung und abschließender Satzinterpunktion. Auch bei inhaltlicher Durchsicht
waren die 24 Antworten falsch; es handelt sich nicht nur um Formatfehler.
Weitere elf offene Aufgaben wurden aufgezeichnet und gelesen, nicht automatisch als richtig gezählt.

Beispiele aus dem besten Zwischenstand:

| Frage | Tatsächliche Antwort |
|---|---|
| Hallo, wer bist du? | Hallo! Wobei kann ich dir helfen? |
| Erkläre mir, warum der Himmel blau ist. | Was soll ich dir erklären? Nenne bitte das Thema. |
| Schreibe drei zusammenhängende Sätze über einen Hund. | Ein kleines Boot schwimmt auf dem See. Der Wind bewegt sein Segel. |
| Was ist 17 + 26? | 29 ist eine Programmiersprache. |
| Merke dir: Mein Lieblingswort ist Banane. Was ist mein Lieblingswort? | Alles klar. |

Der kleine Datensatz fördert kurze gelernte Antworten, aber keine brauchbare Übertragung
auf die neuen Aufgaben. Die fallende Trainings-Loss allein täuscht hier einen Fortschritt vor.
Der Test beweist weder, dass das Modell grundsätzlich nicht lernen kann, noch eine einzelne
Ursache für die schlechte Qualität. Er rechtfertigt keine weitere lange Wiederholung dieses Kurses.

## Weiteres Vorgehen

1. Das Sprachenlernen mit einem breiteren, geprüften deutschen Korpus aus einfachen,
   zusammenhängenden Texten und vielfältigen passenden Fragen/Antworten verbessern.
   Datensatzumfang anhand unterschiedlicher Texte und überwachten Antworttokens dokumentieren.
2. Wortübernahme, Kontextnutzung, einfache Erklärungen und Textzusammenhang getrennt messen.
   Neue Testaufgaben zurückhalten, sobald die bisherigen Ergebnisse zur Entwicklung genutzt werden.
3. Erst mit einem weiteren begrenzten Versuch prüfen, ob unbekannte Aufgaben tatsächlich besser
   beantwortet werden. Größere Modelle oder stundenlange Läufe sind erst danach begründbar.

Das ist eine geplante nächste Untersuchung, kein bereits vorbereiteter großer Trainingslauf
und keine Zusage einer bestimmten späteren Sprachqualität.

## Dateien und technische Prüfung

- `pilot_data.py`: nachvollziehbare Erstellung der kleinen Übungs- und Prüfdaten.
- `dialogue_pilot.py`: begrenzter Versuch in einem neuen Verzeichnis, ohne Änderung des Standardchats.
- `runs/dialogue_pilot_v1/manifest.json`: Konfiguration und Herkunft.
- `runs/dialogue_pilot_v1/results.json`: sämtliche Vorher-/Nachher-Antworten und Messwerte.
- `runs/dialogue_pilot_v1/sft/metrics.jsonl`: Trainingsverlauf.
- `runs/dialogue_pilot_v1/sft/best.pt`: experimenteller Zwischenstand, nicht als Verbesserung freigegeben.
- Gesamte Testsuite: **51 bestanden**. Darunter fünf neue Tests für Datentrennung,
  richtige Rechenergebnisse, Zielverschiebung/Maskierung, strenge Auswertung und Checkpoint-Fortsetzung.

Die technischen Tests belegen die geprüften Softwarefunktionen; sie sind kein Sprachqualitätstest.
