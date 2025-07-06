# NLP-Projekt – Sentimentanalyse

Dieses Repository enthält den finalen Code.

## Projektstruktur

nlp-projekt-final/
├── src/
│ ├── sample_data.py # Sampling (1 000 Reviews × 5 Klassen × 3 Kategorien)
│ ├── preprocess.py # Tokenisierung, Stopwords, Lemmatisierung
│ ├── data_exploration.py # Erste Statistiken (Wortlängen, Sterne-Verteilung)
│ ├── train_model.py # NB + LR + RF (jeweils GridSearch)
│ └── train_model_neuronal.py # TF-IDF + MLP (Early-Stopping)
├── data
│ └──   raw
├── requirements.txt
└── .gitignore

> **Hinweis:** Die großen Rohdaten (`data/raw/`) sind nicht Teil des Repos.


## Installation
pip install -r requirements.txt          # Abhängigkeiten
python src/sample_data.py                # Daten ziehen + samplen
