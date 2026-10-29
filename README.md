# 🚨 Natural Language Processing with Disaster Tweets

Este repositório contém uma solução para a competição **"Natural Language Processing with Disaster Tweets"** do Kaggle. O objetivo do projeto é construir um modelo de Machine Learning/Deep Learning capaz de classificar se um tweet está relatando um desastre/emergência real (`1`) ou se utiliza termos conotativos/figurados (`0`).

---

## 📌 Visão Geral do Projeto

- **Plataforma:** Kaggle
- **Métrica de Avaliação:** $F_1\text{-score}$
- **Linguagem:** Python 3.x
- **Principais Tecnologias:** `pandas`, `scikit-learn`, `PyTorch`, `transformers` (Hugging Face)

---

## ⚙️ Abordagem e Modelos

### Baseline: TF-IDF + Logistic Regression
- **Pré-processamento:** Limpeza de URLs e tags HTML, preservação de hashtags e números, e concatenação do campo `keyword` ao texto do tweet.
- **Vetorização:** `TfidfVectorizer` (unigramas e bigramas, `max_features=10000`).
- **Algoritmo:** Regressão Logística.
- **Resultado ($F_1\text{-score}$ em Validação Cruzada 5-Fold):** ~0.785

---

## 📂 Estrutura do Repositório

```text
├── data/
│   ├── train.csv          # Dataset de treino
│   └── test.csv           # Dataset de teste (Kaggle)
├── notebooks/
│   ├── 01_baseline_tfidf.ipynb     # Modelo baseline (TF-IDF + Logistic Regression)
│   └── 02_deberta_finetuning.ipynb # Fine-tuning com Transformer (DeBERTa-v3)
├── submissions/
│   ├── submission_baseline.csv
│   └── submission_deberta.csv
├── README.md
└── requirements.txt
