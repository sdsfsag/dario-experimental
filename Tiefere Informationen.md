# Dario Experimental

> **Auch wichtig:** Prompts wurden gelöscht und es existieren weiterhin **keine Sicherheitsfilter**.

# Modell- und Trainingshistorie

---

## Modell 1 – Erster Trainingslauf

### Architektur

* **100.682.496 Parameter**
* **12 Transformer-Schichten**
* **768 Hidden-Dimension**
* **12 Attention-Köpfe / 4 KV-Köpfe**
* Eigener **Byte-Level-BPE-Tokenizer** mit **32.768 Tokens**
* **1.024 Tokens Kontextfenster**
* Sprachen: **Deutsch und Englisch**
* Entwickelt mit **Python und PyTorch**

### Training

Der erste Trainingslauf wurde erfolgreich abgeschlossen.

* **19.083 Trainingsschritte insgesamt**

  * **16.083 Schritte Sprachtraining**
  * **3.000 Schritte Dialogtraining / SFT**
* **312,7 Millionen verarbeitete Tokens**
* Trainingsdauer: ca. **1 Stunde und 57 Minuten**

### Trainingsdaten

* Wikipedia
* OpenAssistant OASST2
* Synthetische deutsche Instruktionsdaten

### Trainingseinstellungen

* Optimizer: **AdamW**
* Mixed Precision: **BF16**

### Hardware und Leistung

Trainiert auf einer **NVIDIA RTX 5070 Ti mit 16 GB VRAM**.

* Trainingsgeschwindigkeit: ca. **45.000–47.000 Tokens/s**
* VRAM-Peak: ca. **8,3 GiB**
* Modellgewichte:

  * ca. **403 MB in FP32**
  * ca. **201 MB in BF16**
* Vollständiger Trainingscheckpoint: ca. **1,21 GB**

---

## Prototyp 1 – Modell 2

### Architektur

* **100.682.496 Parameter**
* **12 Transformer-Schichten**
* **768 Hidden-Dimension**
* **12 Attention-Köpfe / 4 KV-Köpfe**
* Eigener **Byte-Level-BPE-Tokenizer** mit **32.768 Tokens**
* **1.024 Tokens Kontextfenster**
* Sprachen: **Deutsch und Englisch**
* Entwickelt mit **Python und PyTorch**
* Weitertrainiert auf den bereits vorhandenen eigenen Modellgewichten

### Training

Der zweite Trainingslauf wurde erfolgreich abgeschlossen.

* **87.677 zusätzliche Trainingsschritte**

  * **84.677 Schritte Sprachtraining**
  * **3.000 Schritte Dialogtraining / SFT**
* **1.436.499.968 Tokens** im neuen Trainingslauf verarbeitet
* **1.700.003.840 Tokens** einschließlich des verwendeten Ausgangs-Sprachtrainings
* Trainingsdauer: ca. **7 Stunden und 38 Sekunden**
* Beginn: **18. September, ca. 06:13 Uhr**
* Abschluss: **18. September, ca. 13:14 Uhr**

### Trainingsdaten

* Wikipedia
* OpenAssistant OASST2
* Synthetische deutsche Instruktionsdaten

### Trainingseinstellungen

* Optimizer: **AdamW**
* Mixed Precision: **BF16**

> **Hinweis:** Mehrfach verwendete Texte werden mehrfach gezählt. Beim SFT enthält der Tokenzähler außerdem Padding. Der vorherige Dialogtrainingslauf ist in der genannten Summe von **1,7 Milliarden Tokens** nicht enthalten.

### Abschließende Trainingswerte

| Trainingsphase       | Training Loss | Validation Loss |
| -------------------- | ------------: | --------------: |
| Sprachtraining       |     **2,789** |       **2,909** |
| Dialogtraining / SFT |     **2,768** |        **3,05** |

Der automatisch ausgewählte Chat-Checkpoint stammt aus den Trainingsschritten mit den besten Validierungswerten.

Seine Trainingslinie umfasst **1.695.465.472 verarbeitete Tokens**. Dadurch liegt dieser Wert etwas unter dem gesamten Trainingszähler.

### Hardware und Leistung

Trainiert auf einer **NVIDIA RTX 5070 Ti mit 16 GB VRAM**.

* Zuletzt ca. **48.000–52.000 Tokens/s**
* Früher gemessener PyTorch-VRAM-Peak: ca. **8,3 GiB**
* Modellgewichte:

  * ca. **403 MB in FP32**
  * ca. **201 MB in BF16**
* Vollständiger Trainingscheckpoint: ca. **1,21 GB**

---

## Prototyp 2 – Modell 2

### Architektur

* **100.682.496 Parameter**
* **12 Transformer-Schichten**
* **768 Hidden-Dimension**
* **12 Attention-Köpfe / 4 KV-Köpfe**
* Eigener **Byte-Level-BPE-Tokenizer** mit **32.768 Tokens**
* **1.024 Tokens Kontextfenster**
* Sprachen: **Deutsch und Englisch**
* Entwickelt mit **Python und PyTorch**

### Training

Der nächste größere Trainingslauf wurde erfolgreich abgeschlossen.

* **67.404 Trainingsschritte insgesamt**

  * **62.904 Schritte Sprachtraining**
  * **4.500 Schritte Dialogtraining / SFT**
* **541,9 Millionen Tokens** im neuesten Trainingslauf verarbeitet
* **2,17 Milliarden Trainingstokens insgesamt** für die ausgewählte Chatversion
* Trainingsdauer: ca. **2 Stunden und 58 Minuten**
* Verwendeter Chat-Checkpoint: bester Dialogstand bei **Schritt 1.500**

### Trainingsdaten

* Wikipedia
* TinyStoriesGerman
* Klexikon
* GermanQuAD
* OpenAssistant OASST2
* Synthetische Instruktionsdaten

### Trainingseinstellungen

* Optimizer: **AdamW**
* Mixed Precision: **BF16**

### Hardware und Leistung

Trainiert auf einer **NVIDIA RTX 5070 Ti mit 16 GB VRAM**.

* Sprachtraining: ca. **58.600 Tokens/s** im Median
* Dialogtraining: ca. **34.500 Tokens/s** im Median
* VRAM-Peak dieses Trainingslaufs: **nicht aufgezeichnet**
* Modellgewichte:

  * ca. **403 MB in FP32**
  * ca. **201 MB in BF16**
* Vollständiger Trainingscheckpoint: ca. **1,21 GB**

---

## Prototyp 3 – Modell 2

### Release Candidate

### Architektur

* **100.682.496 Parameter**
* **Decoder-only-Transformer**
* **12 Transformer-Schichten**
* **768 Hidden-Dimension**
* **12 Attention-Köpfe / 4 KV-Köpfe**
* **RoPE** für Positionsinformationen
* **RMSNorm**
* **SwiGLU**
* Eigener **BPE-Tokenizer** mit **32.768 Tokens**
* **1.024 Tokens Kontextfenster**
* Sprachen: **Deutsch und Englisch**
* Zuletzt stärkerer Schwerpunkt auf **Deutsch**
* Entwickelt mit **Python und PyTorch**

### Training

Für den Release Candidate wurden mehrere neue Dialogtrainingsvarianten getestet.

* Ca. **2,190 Milliarden verarbeitete Tokens** in der Trainingshistorie des ausgewählten Modells
* Letzte Runde: **6 Dialogtrainingsversuche**
* **25.577 zusätzliche Trainingsschritte** über alle sechs Versuche
* **68,6 Millionen Tokens** über alle sechs Versuche verarbeitet
* Ausgewählter Stand:

  * **Versuch 1**
  * **Schritt 1.000**
  * ca. **2,6 Millionen zusätzliche Tokens**
* **105.514 Trainingsdialoge**
* **5.402 Validierungsdialoge**
* Laufzeit der letzten Runde: ca. **1 Stunde und 8 Minuten inklusive Tests**

### Trainingsdaten

* Klexikon
* GermanQuAD
* OpenAssistant
* Synthetische deutsche Dialogdaten
* Synthetische Kontextaufgaben
* Synthetische Rechenaufgaben

### Trainingseinstellungen

* Optimizer: **AdamW**
* Mixed Precision: **BF16**

### Ergebnis

* Validierungs-Loss des ausgewählten Release-Candidate-Standes: **1,699**

> **Hinweis:** Die angegebenen Tokenzahlen enthalten Wiederholungen und teilweise Padding. Die Anzahl der verarbeiteten Tokens entspricht daher nicht der Anzahl einzigartiger Texte, Informationen oder Fakten.

### Hardware und Leistung

Trainiert auf einer **NVIDIA RTX 5070 Ti mit 16 GB VRAM**.

* Trainingsgeschwindigkeit der letzten Runde: ca. **16.200–21.900 Tokens/s**
* Median: ca. **19.500 Tokens/s**
* VRAM-Peak dieser Runde: **nicht erfasst**
* Gespeicherte Modellgewichte: ca. **403 MB in FP32**
* Vollständiger Trainingscheckpoint: ca. **1,21 GB**

---

# Entwicklungsübersicht

| Version                            |                      Trainingsschritte |                                  Tokenstand | Besonderheit                                   |
| ---------------------------------- | -------------------------------------: | ------------------------------------------: | ---------------------------------------------- |
| **Modell 1**                       |                             **19.083** |                              **312,7 Mio.** | Erster vollständiger Trainingslauf             |
| **Prototyp 1 – Modell 2**          |                 **87.677 zusätzliche** |                               **1,70 Mrd.** | Massives Weitertraining                        |
| **Prototyp 2 – Modell 2**          |                             **67.404** |                               **2,17 Mrd.** | Neue Datenquellen und stärkeres Dialogtraining |
| **Prototyp 3 – Release Candidate** | **25.577 zusätzliche** über 6 Versuche | **ca. 2,190 Mrd.** beim ausgewählten Modell | Auswahl des besten Dialog-Checkpoints          |

---

# Aktueller Stand

Der **Release Candidate von Modell 2** besitzt weiterhin **100.682.496 Parameter**, basiert jedoch auf einer deutlich längeren Trainingshistorie als das ursprüngliche Modell.

Vom ersten Trainingslauf mit **312,7 Millionen verarbeiteten Tokens** entwickelte sich das Modell über mehrere Trainings- und Dialogphasen bis zu einer Trainingshistorie von rund **2,190 Milliarden verarbeiteten Tokens**.

Der aktuell ausgewählte Release-Candidate-Checkpoint stammt aus **Dialogtrainingsversuch 1 bei Schritt 1.000** und erreicht einen **Validierungs-Loss von 1,699**.
