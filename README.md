# CallWise: End-to-End ASR Pipeline for Agent–Customer Call Transcription
This repository provides an end-to-end pipeline for fine-tuning Automatic Speech Recognition (ASR) models, integrated with Voice Activity Detection (VAD) to accurately separate agent and customer speech in call recordings. The workflow emphasizes parameter-efficient fine-tuning using LoRA and supports structured, reproducible experimentation.

The project also systematically analyzes the effect of key hyperparameters on Word Error Rate (WER), including:

- Learning Rate  
- LoRA Adapter Rank  
- Model Size (Small vs. Medium)

These experiments enable data-driven optimization of ASR performance and informed model selection decisions.

## Project Highlights
- End-to-end ASR fine-tuning pipeline covering data preparation, model training, and inference
- Support for both Full Fine-Tuning and Parameter-Efficient Fine-Tuning (LoRA)
- Systematic experimentation and comparison of key hyperparameters
- Notebook-driven workflow for transparent and iterative experimentation
- Model evaluation using Weighted / Micro Word Error Rate (WER)
- Modular, reproducible, and production-aligned project structure

## Repository Structure
```text
.
├── 1. Preprocessing_Script.ipynb
├── 2. Train_test_split.ipynb
├── 3. Training_Data_set_preparation.ipynb
├── 3b. Training_Full_training.ipynb
├── 3b. Training_LORA_fine_tuning.ipynb
├── 4. Inference_setup.ipynb
├── utils/
│   └── Preprocessing.py               # For training data preprocessing
│   └── Preprocessing_inference.py     # For inference preprocessing
├── Training_stats/                    # comparison of model training stats across various hyperparameters
│   └── wer_comparison.csv
└── README.md
```



## Notebook Walkthrough (Execution Order)

### Preprocessing  
**1. Preprocessing_Script.ipynb**

- Audio loading
- Silence removal and speech segmentation using VAD
- Text cleanup and alignment
- Ensures model-compatible input format

---

### Train–Test Split  
**2. Train_test_split.ipynb**

- Splits dataset into training and validation sets
- Prevents data leakage between splits

---

### Dataset Preparation  
**3. Training_Data_set_preparation.ipynb**

- Tokenization using the ASR tokenizer
- Feature extraction (log-Mel spectrograms)
- Padding and masking
- Prepares dataset for `Trainer` / `DataLoader`

---

### LoRA Fine-Tuning  
**3b. Training_LORA_fine_tuning.ipynb**

- Parameter-efficient fine-tuning using LoRA
- Experiments across:
  - Multiple learning rates
  - Different LoRA ranks
- Faster training with significantly fewer trainable parameters

---

### Inference Setup  
**4. Inference_setup.ipynb**

- Loads fine-tuned model checkpoints
- Performs ASR inference on actual audio
- Generates call-level transcripts separating calling agent and customer speech




