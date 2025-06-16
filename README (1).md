
# Citation Prediction Between Papers

🏆 **2nd Place Winner** in the 2025 NLP Kaggle Competition at the University of Ioannina, out of 100+ competing teams!

---

## Project Overview

This project involved creating a model to predict citation relationships between scientific papers, leveraging both textual data and citation graph structure. The task was completed as part of the NLP 053 course at the University of Ioannina.

Team Name: **Holy cite**

---

## Key Goals

- Accurately predict if a citation link exists between pairs of scientific papers.
- Utilize both natural language processing (NLP) and graph-based analytical methods.
- Minimize data leakage and ensure robust validation procedures.

---

## Data Preprocessing

- Text cleaning: Removed punctuation, symbols, and stop words.
- Standardized text to lowercase.
- Ensured abstract and author data integrity and completeness.

---

## Feature Engineering

### Text-based Features
- Fine-tuned **DistilBERT** and **SciBERT** models to extract semantic embeddings from abstracts.
- Generated customized embeddings (CLS tokens) for deeper semantic insights.
- Calculated TF-IDF vectors and cosine similarity between abstracts.

### Graph-based Features
- **Node2Vec** embeddings tailored for citation graph structures.
- Extracted additional metrics: Common Neighbors, Preferential Attachment, Adamic-Adar Index.

### Author Similarity
- Computed Jaccard similarity to evaluate author collaboration likelihood.

---

## Modeling & Optimization

### Logistic Regression (Baseline)
- Feature set: Combined embeddings (PCA-reduced), similarity scores, and graph metrics.
- Metrics:
  - AUROC: **0.9981**
  - F1-score: **0.9862**
  - Log-loss: **0.0448**

### CNN with PyTorch
- Multi-channel CNN processing embeddings and metadata in real-time.
- Metrics:
  - AUROC: **0.9962**
  - F1-score: **0.9810**
  - Log-loss: **0.0563**

### XGBoost (Best Performing Model)
- Comprehensive feature set including semantic, graph-based, and similarity metrics.
- Rigorous hyperparameter tuning and early stopping to prevent overfitting.
- Metrics:
  - AUROC: **0.9986**
  - F1-score: **0.9861**
  - Log-loss: **0.0409**

---

## Results & Achievements

The combination of advanced embeddings and careful feature engineering enabled exceptional predictive accuracy, securing **2nd place** among over 100 competing teams.

---

## Repository Structure

```
.
├── notebooks/
│   ├── bert-finetuning_cleaned.ipynb
│   ├── embedding-gen-feature-extraction_cleaned.ipynb
│   └── model-training-and-submission-generation_cleaned.ipynb
├── data/             # Raw and processed datasets
├── models/           # Saved models and preprocessing artifacts
└── README.md
```

---

## References

- [HuggingFace LLM Course](https://huggingface.co/learn)
- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Fine-tuning Transformers](https://www.youtube.com/watch?v=V1-Hm2rNkik)
- [Deep Learning on Graphs](https://yaoma24.github.io/dlg_book/dlg_book.pdf)

---

## Contributors

- **Vasileios Papadimitriou** (4759)
- **Konstantinos Kourkakis** (4089)

Supervisor: **K. Skianis**

Team Name: **Holy cite**
