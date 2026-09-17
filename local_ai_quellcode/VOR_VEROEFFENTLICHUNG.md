# Quellcode-Paket

Enthalten: Python-Quellcode und Tests, Startskripte, Konfigurationen,
selbst erstellte kleine Testbeispiele und Dokumentation.
Der Rechnername wurde aus dem XML-Testbericht entfernt.

Nicht enthalten: heruntergeladene Trainingsdaten, aufbereitete Daten,
trainierter Tokenizer, Modellgewichte, Optimizerzustaende, Laufprotokolle,
Chat-Erinnerungen, virtuelle Python-Umgebung und Download-Cache.
Das Paket enthaelt daher kein sofort nutzbares trainiertes Chatmodell.
Installation und eigenes Training sind in local_ai/README.md beschrieben.

## Lizenz vor der Veroeffentlichung festlegen

Fuer den Projektcode wurde noch keine Lizenz ausgewaehlt. Soll das Projekt
als Open Source weiterverwendbar sein, muss der Herausgeber eine passende
Lizenz waehlen und deren Text als LICENSE beilegen. Dieses Paket vergibt
selbst keine Lizenz.

Hinweise: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository

## Externe Daten und Bibliotheken

Die Download-Funktion kann externe Daten beziehen. Die Dataset-Karten nennen:

- Wikimedia Wikipedia: CC-BY-SA 3.0/GFDL; manche Texte nur unter CC-BY-SA.
  https://huggingface.co/datasets/wikimedia/wikipedia
- OpenAssistant OASST2: Apache-2.0.
  https://huggingface.co/datasets/OpenAssistant/oasst2
- Alpaca GPT4 Deutsch: Apache-2.0 laut Anbieter; synthetische Dialoge.
  https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-deutsch

Diese Datensaetze werden hier nicht mitgeliefert. Bei einer spaeteren
Weitergabe von Daten muessen die jeweiligen Bedingungen und Quellenhinweise
beruecksichtigt werden. Eine Lizenz fuer den Projektcode ersetzt sie nicht.
Fuer eine separate Modellveroeffentlichung sind Gewichte, zugehoeriger
Tokenizer, Konfiguration, Trainingsquellen, Lizenzentscheidung und eine
ehrliche Beschreibung der gemessenen Qualitaet zusammenzustellen.
Die rechtliche Einordnung einer solchen Modellveroeffentlichung wird durch
dieses Quellcode-Paket nicht entschieden.

Python-Abhaengigkeiten werden vom Nutzer installiert und unterliegen ihren
eigenen Lizenzen. Es werden keine Bibliotheksdateien mitgeliefert.
