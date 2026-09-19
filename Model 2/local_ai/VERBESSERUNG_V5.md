# Gemeinsam genutztes Zeitbudget

V5 beginnt beim besten Dialog-V4-Checkpoint (100.682.496 Parameter). Das Zeitbudget
gilt für die gesamte Verbesserungssitzung: mehrere Versuche, Vergleiche und Abschlussprüfung.
Ohne ausdrücklich neue Zeitgrenze gilt das ursprüngliche Ende am 19.09.2026 um 17:48 Uhr.

Schwerpunkte: natürlich formulierte Fragen, kurze Sachantworten aus Klexikon,
Fragen zu kurzen Texten, Gesprächsbezug und Rechenbeispiele mit Zwischenschritten.
Die früheren Testdaten werden nicht zum Training hinzugefügt. Neue Fragevarianten
bedeuten keine entsprechend große Menge neuer Fakten.

Bei ausbleibendem Fortschritt wechselt ein Versuch zum nächsten Schwerpunkt.
Gespeichert wird jeweils der Stand mit dem niedrigsten festen Validierungs-Loss.
Ein Entwicklungstest vergleicht anschließend die Antworten mit V4. Er muss zusätzliche
richtige Antworten zeigen und die bisherigen Fähigkeiten weitgehend erhalten.
Falls kein Kandidat diese Bedingungen erfüllt, bleibt V4 ausgewählt.
Ein anderer Abschlusstest wird nur berichtet. Die einfachen Wortprüfungen für natürliche
Antworten sind eine Orientierung; die ausgegebenen Texte müssen weiterhin beurteilt werden.

Die Tests und der neue Chat nutzen dieselbe deterministische Ausgabe: Temperatur 0,
Repetition Penalty 1.0. Ein kurzer Vergleich am V4-Entwicklungssatz ergab damit 15 statt
14 exakt richtige Leseantworten, aber weiterhin schwache freie Antworten. Das allein
belegt keine allgemeine Verbesserung.

- `Training_Live.cmd`: aktuelle Konsole öffnen.
- `Training_stoppen.cmd`: nach vollständigem Schritt speichern und anhalten.
- `Training_4_Stunden.cmd`: innerhalb derselben Zeitgrenze fortsetzen.
- `Chat_Neues_Modell.cmd`: nach Abschluss den ausgewählten Stand verwenden.

Alte Checkpoints bleiben unverändert. Status, Kandidatenvergleiche und Ergebnisse liegen
in `runs/improvement_v5`; Datenherkunft und Prüfsummen in `data/improvement_v5/manifest.json`.
