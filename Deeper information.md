# Dario Experimental

> **Also important:** Prompts were removed and there are still **no safety filters**.

# Model and Training History

---

## Model 1 – First Training Run

### Architecture

* **100,682,496 parameters**
* **12 Transformer layers**
* **768 hidden dimension**
* **12 attention heads / 4 KV heads**
* Custom **Byte-Level BPE tokenizer** with **32,768 tokens**
* **1,024-token context window**
* Languages: **German and English**
* Developed with **Python and PyTorch**

### Training

The first training run was successfully completed.

* **19,083 training steps in total**

  * **16,083 language training steps**
  * **3,000 dialogue training / SFT steps**
* **312.7 million processed tokens**
* Training duration: approx. **1 hour and 57 minutes**

### Training Data

* Wikipedia
* OpenAssistant OASST2
* Synthetic German instruction data

### Training Settings

* Optimizer: **AdamW**
* Mixed Precision: **BF16**

### Hardware and Performance

Trained on an **NVIDIA RTX 5070 Ti with 16 GB VRAM**.

* Training speed: approx. **45,000–47,000 tokens/s**
* VRAM peak: approx. **8.3 GiB**
* Model weights:

  * approx. **403 MB in FP32**
  * approx. **201 MB in BF16**
* Full training checkpoint: approx. **1.21 GB**

---

## Prototype 1 – Model 2

### Architecture

* **100,682,496 parameters**
* **12 Transformer layers**
* **768 hidden dimension**
* **12 attention heads / 4 KV heads**
* Custom **Byte-Level BPE tokenizer** with **32,768 tokens**
* **1,024-token context window**
* Languages: **German and English**
* Developed with **Python and PyTorch**
* Further trained on the already existing custom model weights

### Training

The second training run was successfully completed.

* **87,677 additional training steps**

  * **84,677 language training steps**
  * **3,000 dialogue training / SFT steps**
* **1,436,499,968 tokens** processed in the new training run
* **1,700,003,840 tokens** including the initial language training that was used
* Training duration: approx. **7 hours and 38 seconds**
* Start: **September 18, approx. 06:13**
* Completion: **September 18, approx. 13:14**

### Training Data

* Wikipedia
* OpenAssistant OASST2
* Synthetic German instruction data

### Training Settings

* Optimizer: **AdamW**
* Mixed Precision: **BF16**

> **Note:** Repeatedly used texts are counted multiple times. During SFT, the token counter also includes padding. The previous dialogue training run is not included in the stated total of **1.7 billion tokens**.

### Final Training Values

| Training Phase          | Training Loss | Validation Loss |
| ----------------------- | ------------: | --------------: |
| Language training       |     **2.789** |       **2.909** |
| Dialogue training / SFT |     **2.768** |        **3.05** |

The automatically selected chat checkpoint comes from the training steps with the best validation values.

Its training line includes **1,695,465,472 processed tokens**. This is why this value is slightly below the total training counter.

### Hardware and Performance

Trained on an **NVIDIA RTX 5070 Ti with 16 GB VRAM**.

* Most recently approx. **48,000–52,000 tokens/s**
* Previously measured PyTorch VRAM peak: approx. **8.3 GiB**
* Model weights:

  * approx. **403 MB in FP32**
  * approx. **201 MB in BF16**
* Full training checkpoint: approx. **1.21 GB**

---

## Prototype 2 – Model 2

### Architecture

* **100,682,496 parameters**
* **12 Transformer layers**
* **768 hidden dimension**
* **12 attention heads / 4 KV heads**
* Custom **Byte-Level BPE tokenizer** with **32,768 tokens**
* **1,024-token context window**
* Languages: **German and English**
* Developed with **Python and PyTorch**

### Training

The next larger training run was successfully completed.

* **67,404 training steps in total**

  * **62,904 language training steps**
  * **4,500 dialogue training / SFT steps**
* **541.9 million tokens** processed in the latest training run
* **2.17 billion training tokens in total** for the selected chat version
* Training duration: approx. **2 hours and 58 minutes**
* Chat checkpoint used: best dialogue state at **step 1,500**

### Training Data

* Wikipedia
* TinyStoriesGerman
* Klexikon
* GermanQuAD
* OpenAssistant OASST2
* Synthetic instruction data

### Training Settings

* Optimizer: **AdamW**
* Mixed Precision: **BF16**

### Hardware and Performance

Trained on an **NVIDIA RTX 5070 Ti with 16 GB VRAM**.

* Language training: approx. **58,600 tokens/s** median
* Dialogue training: approx. **34,500 tokens/s** median
* VRAM peak for this training run: **not recorded**
* Model weights:

  * approx. **403 MB in FP32**
  * approx. **201 MB in BF16**
* Full training checkpoint: approx. **1.21 GB**

---

## Prototype 3 – Model 2

### Release Candidate

### Architecture

* **100,682,496 parameters**
* **Decoder-only Transformer**
* **12 Transformer layers**
* **768 hidden dimension**
* **12 attention heads / 4 KV heads**
* **RoPE** for positional information
* **RMSNorm**
* **SwiGLU**
* Custom **BPE tokenizer** with **32,768 tokens**
* **1,024-token context window**
* Languages: **German and English**
* Recently with a stronger focus on **German**
* Developed with **Python and PyTorch**

### Training

Several new dialogue training variants were tested for the Release Candidate.

* Approx. **2.190 billion processed tokens** in the training history of the selected model
* Latest round: **6 dialogue training attempts**
* **25,577 additional training steps** across all six attempts
* **68.6 million tokens** processed across all six attempts
* Selected state:

  * **Attempt 1**
  * **Step 1,000**
  * approx. **2.6 million additional tokens**
* **105,514 training dialogues**
* **5,402 validation dialogues**
* Runtime of the latest round: approx. **1 hour and 8 minutes including tests**

### Training Data

* Klexikon
* GermanQuAD
* OpenAssistant
* Synthetic German dialogue data
* Synthetic context tasks
* Synthetic arithmetic tasks

### Training Settings

* Optimizer: **AdamW**
* Mixed Precision: **BF16**

### Result

* Validation loss of the selected Release Candidate state: **1.699**

> **Note:** The stated token counts include repetitions and partially padding. Therefore, the number of processed tokens does not correspond to the number of unique texts, pieces of information, or facts.

### Hardware and Performance

Trained on an **NVIDIA RTX 5070 Ti with 16 GB VRAM**.

* Training speed in the latest round: approx. **16,200–21,900 tokens/s**
* Median: approx. **19,500 tokens/s**
* VRAM peak for this round: **not recorded**
* Stored model weights: approx. **403 MB in FP32**
* Full training checkpoint: approx. **1.21 GB**

---

# Development Overview

| Version                             |                          Training Steps |                               Token Count | Key Feature                                     |
| ----------------------------------- | --------------------------------------: | ----------------------------------------: | ----------------------------------------------- |
| **Model 1**                         |                              **19,083** |                                **312.7M** | First complete training run                     |
| **Prototype 1 – Model 2**           |                   **87,677 additional** |                                 **1.70B** | Major continued training                        |
| **Prototype 2 – Model 2**           |                              **67,404** |                                 **2.17B** | New data sources and stronger dialogue training |
| **Prototype 3 – Release Candidate** | **25,577 additional** across 6 attempts | **approx. 2.190B** for the selected model | Selection of the best dialogue checkpoint       |

---

# Current Status

The **Release Candidate of Model 2** still contains **100,682,496 parameters**, but is based on a significantly longer training history than the original model.

Starting from the first training run with **312.7 million processed tokens**, the model developed through multiple training and dialogue phases into a training history of around **2.190 billion processed tokens**.

The currently selected Release Candidate checkpoint comes from **dialogue training attempt 1 at step 1,000** and achieves a **validation loss of 1.699**.
