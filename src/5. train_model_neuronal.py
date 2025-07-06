# Datei: src/train_model_neuronal.py

import json
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CATEGORIES   = ["Books", "Baby_Products", "Home_and_Kitchen"]
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

def load_data(categories, data_dir=PROCESSED_DIR):
    texts, labels = [], []
    for cat in categories:
        path = os.path.join(data_dir, f"{cat}_processed.jsonl")
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                texts.append(" ".join(rec.get("tokens", [])))
                labels.append(rec.get("rating"))
    return texts, labels

def print_metrics(y_true, y_pred, labels=[1,2,3,4,5]):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred, labels=labels), "\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

def main():
    # 1) Daten laden & splitten
    texts, labels = load_data(CATEGORIES)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # 2) Pipeline: TF-IDF + MLPClassifier (mit EarlyStopping)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=5)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(100,),       # ein versteckter Layer mit 100 Neuronen
            alpha=0.0005,                    # L2-Regularisierung
            max_iter=100,                    # max. Epochen
            early_stopping=True,             # Stoppt, wenn Validierungs-Score stagniert
            n_iter_no_change=5,              # Geduld in Epochen
            tol=1e-3,                        # minimaler Verbesserungs-Tolerenzwert
            random_state=42,
            verbose=True                     # zeigt den Trainingsfortschritt
        ))
    ])

    # 3) Trainieren & evaluieren
    print("=== Schnelles Neural Network Training ===")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print_metrics(y_test, y_pred)

if __name__ == "__main__":
    main()
