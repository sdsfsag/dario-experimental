# Gespraechsqualitaet nach model 2

Pruefung am 18.09.2026. Es wurde kein weiterer Trainingslauf gestartet und
keine Gewichtedatei oder Erinnerungsdatenbank geaendert.

## Ergebnis

Die fuenf vom Nutzer gezeigten Aufgaben werden weder von real_v1 noch von
real_v2 sinnvoll und vollstaendig erfuellt. Der verbesserte Validation Loss
belegt hier keinen erfolgreichen Gespraechsassistenten. Auch die neuen,
isolierten Proben zeigen falsche Aussagen, starke Wiederholungen, fehlenden
Bezug zur Frage und unzureichendes Nutzen des gegebenen Kontexts.

## Direkte Pruefungen

- Eigene Checkpoints direkt geladen: real_v1 SFT Schritt 3000, real_v2 SFT
  Schritt 2900, real_v2 Pretraining Schritt 84500. Tokenizer-Fingerprints
  stimmen mit den Checkpoints ueberein.
- Dieselben fuenf Nutzerfragen einzeln ohne Gespraechshistorie und ohne
  gespeicherte Erinnerungen gestellt. Greedy-Decoding (temperature=0),
  maximal 64 neue Tokens. Diese Einstellung weicht absichtlich vom
  zufaelligen Sampling des interaktiven Chats ab.
- real_v2 antwortet beispielsweise auf die Rechenfrage, 17 + 26 sei der
  erste Tag in der Geschichte der Menschheit. Das Lieblingswort Banane wird
  faelschlich mit Apfel beantwortet.
- Weglassen des Datumszusatzes stellt die Gespraechsqualitaet nicht her.
- Cache und vollstaendige Neuberechnung ergeben fuer die ersten 32 Tokens
  der untersuchten Begruessung dieselben IDs, bei beiden SFT-Versionen.
  Dies ist eine gezielte Probe und kein Beweis fuer Fehlerfreiheit aller
  denkbaren Eingaben.
- Erste vorbereitete SFT-Zeile mit Quelldialog verglichen: Eingabetokens und
  um eins versetzte, maskierte Labels stimmen exakt ueberein. Weitere
  Maskierungs-, Kausalitaets- und Cache-Tests existieren in test_core.py.
- Das reine real_v2-Sprachmodell erzeugt bereits bei drei einfachen
  Textfortsetzungen falsche und repetitive Inhalte. Die Schwaeche betrifft
  somit auch das Grundmodell, nicht allein das Chat-Fenster.

## Daten und Lernumfang

- 48.381 synthetische deutsche Alpaca-Gespraeche.
- 418 deutsche und 3.838 englische OASST2-Gespraeche.
- Zusammen 52.637 Gespraeche, aufbereitet zu 53.103 SFT-Fenstern.
- Pro SFT-Lauf: 3.000 Updates mal 16 Fenster = 48.000 zufaellige Ziehungen.
  Das entspricht 0,904 Datensatzdurchlaeufen nach Umfang. Es ist keine
  vollstaendige Abdeckung: Ziehungen koennen sich wiederholen.
- Der gesamte vorbereitete SFT-Trainingssatz hat 54.377.472 Eingabepositionen,
  davon 14.081.710 ohne Padding und 9.342.678 bewertete Antworttokens.
  Diese Korpuszaehler sind keine Messung der tatsaechlich gezogenen Tokens.
- Die 49.152.000 Eingabetokens pro SFT-Lauf enthalten Padding. Diese bereits
  dokumentierte Zaehlerkonvention ist kein Mass fuer die Menge neu gelernter
  Informationen.

Diese Befunde sprechen fuer Schwaechen bei Grundmodell, Datenzusammenstellung
und Dialogtraining. Sie beweisen keine einzelne Ursache. Mehr Parameter oder
einfaches Wiederholen desselben langen Laufs ist damit noch nicht begruendet.
Ein naechster Versuch sollte separat bleiben, gezielt Datenqualitaet und
Dialoglernen pruefen und vor weiterem langen Training an getrennten
Gespraechsaufgaben bewertet werden.

Rohdaten der Diagnose: CHAT_DIAGNOSE.json. Eine kleine Diagnose mit diesen
bekannten Fragen ersetzt keinen unabhaengigen allgemeinen Qualitaetsbenchmark.
