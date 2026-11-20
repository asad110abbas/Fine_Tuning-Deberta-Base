# 🎯 BRIHA · Fine-tuning DeBERTa for Review Classification

![Part of BRIHA](https://img.shields.io/badge/Part_of-BRIHA-8B5CF6?style=for-the-badge&labelColor=0B1024)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) ![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat-square&logo=huggingface&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white) ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white) ![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=flat-square) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

> ⚡ **Part of [BRIHA](https://github.com/asad110abbas)** · Briha Research, built by Syed Asad Abbas.

End-to-end NLP project: label and explore a customer-review dataset, then **fine-tune Microsoft's
DeBERTa-base transformer** to classify reviews automatically.

## Two notebooks
| Notebook | What it does |
|---|---|
| `Reviews_Classification.ipynb` | **Data preparation & EDA:** text cleaning (NLTK, regex), sentiment labelling, word clouds, Plotly / Seaborn charts, baseline Keras model |
| `Fine_Tuning_Deberta_Base.ipynb` | **Transformer fine-tuning:** tokenization, `AutoModelForSequenceClassification`, Hugging Face `Trainer`, evaluation with scikit-learn metrics |

## Fine-tuning pipeline
```mermaid
flowchart LR
    A[Labelled reviews] --> B[Train / test split]
    B --> C[DeBERTa tokenizer]
    C --> D[DeBERTa-base + classification head]
    D --> E[Hugging Face Trainer]
    E --> F[Accuracy · F1 · confusion matrix]
    F --> G[Saved model for inference]
```

## Run
```bash
pip install transformers datasets torch scikit-learn pandas
```
Run `Reviews_Classification.ipynb` first if your data isn't labelled yet, then `Fine_Tuning_Deberta_Base.ipynb`.
A GPU (e.g. Google Colab) is recommended.
