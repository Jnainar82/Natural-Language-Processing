# Customer Feedback Intelligence — NLP Pipeline for Yelp Reviews

An end-to-end NLP pipeline for customer feedback analysis, built on the [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full) dataset. Covers preprocessing, classical ML, semantic embeddings, and transformer fine-tuning for binary sentiment classification (positive/negative).

## Overview

| Task | Notebook | What it covers |
|---|---|---|
| 1, 2, 3, 5 | `NLP_Assignment_Tasks1_3_5.ipynb` | Preprocessing & EDA, classical ML baseline, Word2Vec embeddings, responsible NLP discussion |
| 4 | `Task4_Transformer_Yelp.ipynb` | DistilBERT fine-tuning |

**Dataset:** Yelp Review Full — 650K train / 50K test reviews, 1–5 star ratings. Collapsed to binary sentiment for this project: 1–2 stars → negative, 4–5 stars → positive, 3-star reviews dropped as ambiguous.

## Results Summary

### Classical baseline (TF-IDF + linear models, 20K train / 4K test)

| Model | Accuracy | Precision | Recall | F1 | Train time |
|---|---|---|---|---|---|
| Naive Bayes | 0.899 | 0.901 | 0.899 | 0.900 | 0.01s |
| Logistic Regression | **0.920** | 0.919 | 0.922 | **0.921** | 1.6s |
| Linear SVM | 0.918 | 0.922 | 0.915 | 0.918 | 0.1s |

### Transformer (DistilBERT, 4K train / 500 val / 500 test)

| Metric | Value |
|---|---|
| Accuracy | 0.93 |
| F1 (macro/weighted) | 0.93 |
| Best validation F1 | 0.912 |
| Training time | 1.07 min (GPU) |

The transformer outperforms the classical baseline by ~1 point of accuracy/F1 on a fraction of the labelled data (4K vs 20K examples), consistent with transfer learning from pretraining — at the cost of longer training time, a much larger model (~66M parameters vs a sparse linear model), and reduced interpretability.

### Semantic embeddings (Word2Vec, Skip-Gram, 8K reviews)

Vocabulary size: 6,350 words. Nearest-neighbour retrieval for domain terms (e.g. "staff" → *accommodating*, *enthusiastic*) is semantically sensible for frequent words but noisier for others (e.g. "delivery" → *ontrac*, *domino's*), reflecting the limits of a small, single-domain training corpus.

### Stopword ablation (optional)

Removing stopwords outperformed keeping them — 0.773 vs 0.667 accuracy — on a fixed 10K-feature TF-IDF budget, likely because keeping stopwords lets high-frequency function words crowd out the feature budget that would otherwise go to content words.

## Repository Structure

```
.
├── NLP_Assignment_Tasks1_3_5.ipynb   # Tasks 1, 2, 3, 5
├── Task4_Transformer_Yelp.ipynb      # Task 4
├── README.md
└── outputs/                          # saved figures, entity CSVs, model files (generated on run)
```

## Running the Notebooks

Both notebooks are built for Google Colab (GPU recommended for Task 4: *Runtime → Change runtime type → T4 GPU*).

```bash
# Core dependencies installed at the top of each notebook:
pip install datasets nltk spacy gensim wordcloud scikit-learn pandas matplotlib seaborn
python -m spacy download en_core_web_sm

# Task 4 additionally requires:
pip install transformers accelerate torch
```

Run top to bottom. Sample sizes are capped for reasonable runtime in each notebook (e.g. 20K/4K for the classical baseline, 4K/500/500 for transformer fine-tuning) — increase them if you have the compute budget.

## Key Techniques

- **Preprocessing:** NLTK (POS-aware lemmatisation) and spaCy (integrated pipeline), compared directly
- **NER:** spaCy statistical NER (ORG, GPE, DATE, MONEY) plus a keyword-based service-term tagger
- **Classical ML:** TF-IDF features with Naive Bayes, Logistic Regression, and Linear SVM
- **Embeddings:** Gensim Word2Vec (Skip-Gram), cosine similarity, PCA/t-SNE visualisation, semantic retrieval
- **Transformer:** DistilBERT fine-tuned via Hugging Face `Trainer`, evaluated with accuracy/precision/recall/F1/macro-F1/weighted-F1

## Dataset Citation

Zhang, X., Zhao, J. and LeCun, Y. (2015) *Character-level Convolutional Networks for Text Classification*. Advances in Neural Information Processing Systems 28 (NeurIPS 2015). Dataset: https://huggingface.co/datasets/Yelp/yelp_review_full

## License

[Add your license here, e.g. MIT]
