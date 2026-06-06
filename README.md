# EN→DE Machine Translation: Europarl Domain

A comparative study of three machine translation approaches for English-to-German parliamentary translation, built as part of the *Applied Machine Learning for Language Processing* course at Hochschule Campus Wien.

## Project Overview

This project investigates whether fine-tuning an open-source MT model on domain-specific data (EU parliamentary proceedings) can outperform general-purpose translation systems. We compare three approaches:

| Model | Type | BLEU | METEOR | chrF |
|---|---|---|---|---|
| NLLB-200-distilled-600M | Multilingual foundation model | 26.16 | 0.459 | 56.51 |
| Helsinki-NLP/opus-mt-en-de | Pretrained MarianMT pipeline | 30.21 | 0.503 | 59.76 |
| Helsinki-NLP/opus-mt-en-de (fine-tuned) | Fine-tuned on Europarl | 30.57 | 0.504 | 60.05 |

## Project Structure

```
machine-learning-project/
├── src/
│   ├── __init__.py
│   ├── config.py              # Hyperparameters & model names
│   ├── data.py                # Dataset loading & cleaning
│   ├── preprocessing.py       # Tokenization & train/val/test split
│   ├── evaluate.py            # Metrics (BLEU, METEOR, chrF) & baseline evaluation
│   └── train.py               # Fine-tuning with Seq2SeqTrainer
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   └── 02_experiments.ipynb    # Hyperparameter search & final evaluation
├── predictions/
│   └── translations.json       # All test set translations with sentence-level scores
├── app.py                      # Streamlit comparison UI
├── .streamlit/
│   └── config.toml             # Streamlit theme config
├── pyproject.toml
├── .gitignore
└── README.md
```

## Setup

### NLTK Data (required for METEOR)

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Usage

### Run Streamlit UI

Click [here](https://translation-comparison.streamlit.app/)

### Reproduce Experiments

The full training and evaluation pipeline is documented in 
`notebooks/02_experiments.ipynb`. This notebook was executed on 
Google Colab (T4 GPU). To reproduce, upload the notebook to Colab 
and run all cells.

For local execution, see `src/` for the modular implementation.

## Dataset

Europarl parallel corpus (EN-DE) via [Hugging Face](https://huggingface.co/datasets/Helsinki-NLP/europarl). 10,000 sentence pairs subsampled (seed=42), cleaned to 9,760 pairs after filtering short sentences, duplicates, and misaligned pairs. Split: 80/10/10 (train/val/test).

## Configuration

All hyperparameters are centralized in `src/config.py`:

| Parameter | Value |
|---|---|
| Model | Helsinki-NLP/opus-mt-en-de |
| Learning Rate | 1e-5 |
| Epochs | 3 |
| Batch Size | 16 |
| Max Length | 128 |
| Weight Decay | 0.01 |
| Seed | 42 |


*Group D — Hochschule Campus Wien - MODULE Applied Machine Learning for Language Processing*