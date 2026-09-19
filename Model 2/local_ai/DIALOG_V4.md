# Dialogtraining V4

- Vorhandene 100.682.496 Parameter; Start beim besten SFT-Checkpoint von quality_v3.
- Bis zu vier Stunden ab Trainingsstart. Nach ca. 20 Minuten prüft ein separater Entwicklungstest neue Antworten. Nur bei mindestens vier zusätzlichen richtigen Antworten, ohne Rückschritt bei Kontext und Lesen und mit besserem Validierungs-Loss, wird weitertrainiert. Das ist kein Nachweis allgemeiner Chatqualität.
- Kurze Dialoge, Textverständnis, Namen/Korrekturen im Gespräch, kleine Rechenaufgaben und Formatanweisungen. Keine weiteren Geschichten-Fortsetzungen im SFT.
- Jede Unterhaltung bekommt gleiches Gewicht, unabhängig von ihrer Antwortlänge. Loss-Werte sind deshalb nicht direkt mit V3 vergleichbar.
- Die kleinste feste Validierungsabweichung bestimmt den Checkpoint. Nach zwölf Prüfabschnitten ohne relevante Verbesserung endet der Lauf früher. Ein separater Abschlusstest wird nur berichtet, nicht zur Gewichtsauswahl verwendet.

`Training_Live.cmd` zeigt den Fortschritt. `Training_stoppen.cmd` speichert und pausiert; `Training_4_Stunden.cmd` setzt innerhalb desselben Vier-Stunden-Zeitfensters fort. Ein Neustart verlängert das Zeitfenster nicht. Speichern und Abschlusstests können wenige Minuten zusätzlich benötigen.

Danach `Chat_Neues_Modell.cmd`. Falls der kurze Versuch scheitert, bleibt der Chat bei V3. Die bisherigen Checkpoints werden nicht überschrieben. Ein fertig trainierter V4-Checkpoint bleibt experimentell und muss anhand seiner Antworten beurteilt werden.

Details: `runs/dialogue_v4/status.json`, `trial_decision.json`, `quality_results.json`. Datenherkunft und Dateiprüfsummen stehen in `data/dialogue_v4/manifest.json`. Die Tests sind aus diesem Trainingszyklus ausgeschlossen; verwandte Fakten können in früheren Daten vorkommen. Gemeldete verarbeitete Tokens sind Eingabepositionen einschließlich Padding und Wiederholungen, keine Zahl neuer Fakten.
