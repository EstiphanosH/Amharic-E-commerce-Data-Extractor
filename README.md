
# 🛍️ Amharic E-commerce Data Extractor

A Telegram-based Amharic data ingestion, annotation, and model training pipeline for e-commerce and micro-lending insights. This project supports Named Entity Recognition (NER) in Amharic and vendor analytics using scraped messages from Ethiopian-based Telegram vendors.

---

## 📚 Table of Contents

- [🛍️ Amharic E-commerce Data Extractor](#️-amharic-e-commerce-data-extractor)
  - [📚 Table of Contents](#-table-of-contents)
  - [📌 Project Overview](#-project-overview)
  - [💻 Technology Stack](#-technology-stack)
  - [⚙️ System Requirements](#️-system-requirements)
  - [📁 Project Structure](#-project-structure)
  - [🧠 Task Summaries](#-task-summaries)
    - [✅ Task 1: Data Collection \& Preprocessing](#-task-1-data-collection--preprocessing)
    - [✅ Task 2: Labeling in CoNLL Format](#-task-2-labeling-in-conll-format)
    - [✅ Task 3: Fine-Tune NER Model](#-task-3-fine-tune-ner-model)
    - [✅ Task 4: Model Comparison](#-task-4-model-comparison)
    - [✅ Task 5: Model Interpretability](#-task-5-model-interpretability)
    - [✅ Task 6: Vendor Scorecard for Lending](#-task-6-vendor-scorecard-for-lending)
  - [🔖 Annotation Schema](#-annotation-schema)
  - [📊 Demo Notebooks](#-demo-notebooks)
  - [🚀 Getting Started](#-getting-started)
  - [📌 License](#-license)
  - [This project is intended for academic and research use.](#this-project-is-intended-for-academic-and-research-use)
  - [🙌 Acknowledgements](#-acknowledgements)

---

## 📌 Project Overview

This project was built to:
- Ingest and preprocess Amharic-language Telegram e-commerce data
- Annotate and fine-tune models for Named Entity Recognition (NER)
- Evaluate and compare multilingual models for Amharic NER
- Interpret and explain NER model predictions
- Build a FinTech-ready vendor analytics and lending scorecard

---

## 💻 Technology Stack

- **Python 3.10+**
- **Telethon** – Telegram data scraping
- **HuggingFace Transformers** – Fine-tuning multilingual NER models
- **spaCy, Pandas, Regex** – Preprocessing and evaluation
- **SHAP & LIME** – Model interpretability
- **Google Colab / Jupyter Notebooks** – Development environment
- **GitHub Actions** – CI/CD and version control

---

## ⚙️ System Requirements

- Python 3.10+
- pip / conda
- Telegram API credentials
- Access to GPU (Colab Pro, local CUDA, or AWS)

---

## 📁 Project Structure

```
├── .github/workflows/            # CI/CD workflows
├── .vscode/                      # VSCode workspace config
├── app/                          # Web app backend (future use)
├── config/
│   └── config.yaml               # Global parameters and channel lists
├── notebooks/
│   ├── NER_Labeling_Task.ipynb       # Manual labeling in CoNLL
│   ├── preprocessing.ipynb           # Cleaning and tokenizing messages
│   ├── telegram_scraping.ipynb       # Data scraping pipeline
│   ├── finetune_NER_model.ipynb      # Model training
│   ├── model_comparison.ipynb        # Side-by-side model evaluation
│   ├── model_interpretability.ipynb  # SHAP/LIME explanations
│   └── vendor_scorecard.ipynb        # FinTech analytics + lending score
├── scripts/
│   ├── telegram_scraper.py
│   ├── preprocessor.py
│   ├── ner_trainer.py
│   ├── ner_data_utils.py
│   ├── model_interpret.py
│   ├── coll_annotator.py
│   └── vendor_analytics.py
├── src/                       # Internal packages
├── tests/                     # Unit and functional tests
├── utils/                     # Utility modules
├── main.py
└── README.md
```

---

## 🧠 Task Summaries

### ✅ Task 1: Data Collection & Preprocessing
- Scraped Amharic messages, images, and metadata using `telethon`.
- Normalized text, tokenized Amharic, separated metadata.
- Output structured dataset saved for annotation.

### ✅ Task 2: Labeling in CoNLL Format
- Manually annotated 50+ messages with entity tags: `B/I-Product`, `B/I-LOC`, `B/I-PRICE`, `O`.
- Saved labels in standard CoNLL format for training.

### ✅ Task 3: Fine-Tune NER Model
- Trained XLM-Roberta, AfriBERTa, and mBERT using Hugging Face.
- Used `ner_trainer.py` to handle training loop, tokenizer, and CoNLL loader.
- Tuned learning rate, epochs, and batch sizes in `finetune_NER_model.ipynb`.

### ✅ Task 4: Model Comparison
- Benchmarked models by F1 score and training speed.
- Used `model_comparison.ipynb` to visualize performance.
- Selected XLM-Roberta for highest accuracy.

### ✅ Task 5: Model Interpretability
- Used SHAP and LIME to explain why models tagged tokens as entities.
- Difficult cases were manually reviewed in `model_interpretability.ipynb`.

### ✅ Task 6: Vendor Scorecard for Lending
- Created `vendor_analytics.py` to calculate:
  - Avg. Posts/Week
  - Avg. Views/Post
  - Avg. Price
  - Top Performing Product
- Final vendor score:  
  `Score = (Avg Views × 0.5) + (Post Freq × 0.5)`

---

## 🔖 Annotation Schema

| Tag        | Description                      |
|------------|----------------------------------|
| B-Product  | Start of product name            |
| I-Product  | Continuation of product name     |
| B-PRICE    | Start of price entity            |
| I-PRICE    | Continuation of price entity     |
| B-LOC      | Start of location                |
| I-LOC      | Continuation of location         |
| O          | Outside any named entity         |

---

## 📊 Demo Notebooks

- `notebooks/NER_Labeling_Task.ipynb`
- `notebooks/telegram_scraping.ipynb`
- `notebooks/finetune_NER_model.ipynb`
- `notebooks/model_comparison.ipynb`
- `notebooks/vendor_scorecard.ipynb`

---

## 🚀 Getting Started

```bash
git clone https://github.com/EstiphanosH/Amharic-Ecommerce-Data-Extractor.git
cd Amharic-Ecommerce-Data-Extractor
pip install -r requirements.txt

# Example: Run scraper
python scripts/telegram_scraper.py

# Preprocess messages
python scripts/preprocessor.py
```

---

## 📌 License

This project is intended for academic and research use. 
---

## 🙌 Acknowledgements

- Built as part of 10 Academy’s NLP track
- Special thanks to open-source communities supporting low-resource languages
