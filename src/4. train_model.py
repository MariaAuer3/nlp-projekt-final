# Datei: src/train_model.py

import json
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

CATEGORIES = ["Books", "Baby_Products", "Home_and_Kitchen"]
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_data(categories, data_dir=PROCESSED_DIR):
    """
    Lädt tokenisierte Texte und Labels aus den processed JSONL-Dateien.
    Rückgabe: texts (List[str]), labels (List[int])
    """
    texts, labels = [], []
    for cat in categories:
        path = os.path.join(data_dir, f"{cat}_processed.jsonl")
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                texts.append(" ".join(rec.get("tokens", [])))
                labels.append(rec.get("rating"))
    return texts, labels


def print_metrics(y_true, y_pred, labels=[1, 2, 3, 4, 5]):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    print("Confusion Matrix:")
    print(cm, "\n")
    print("Classification Report:")
    print(report)


def main():
    # Daten laden und splitten
    texts, labels = load_data(CATEGORIES)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # 1) Baseline: CountVectorizer + MultinomialNB
    print("=== Baseline: Naive Bayes ===")
    pipeline_nb = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB())
    ])
    pipeline_nb.fit(X_train, y_train)
    y_pred_nb = pipeline_nb.predict(X_test)
    print_metrics(y_test, y_pred_nb)

    # 2) Optimierung: TF-IDF + Logistic Regression + GridSearch
    print("=== Optimierung: TF-IDF + Logistic Regression + GridSearch ===")
    pipeline_lr = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial'))
    ])
    param_grid_lr = {
        'tfidf__ngram_range': [(1,1), (1,2)],
        'tfidf__max_df': [0.8, 0.9, 1.0],
        'clf__C': [0.1, 1, 10]
    }
    grid_lr = GridSearchCV(pipeline_lr, param_grid_lr, cv=3, n_jobs=-1, verbose=1)
    grid_lr.fit(X_train, y_train)
    print("Best Params (LR):", grid_lr.best_params_, "\n")
    y_pred_lr = grid_lr.predict(X_test)
    print_metrics(y_test, y_pred_lr)

    # 3) Erweiterung: CountVectorizer + Random Forest + GridSearch
    print("=== Erweiterung: CountVectorizer + Random Forest + GridSearch ===")
    pipeline_rf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1,2), max_df=0.9)),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    param_grid_rf = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    }
    grid_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=3, n_jobs=-1, verbose=1)
    grid_rf.fit(X_train, y_train)
    print("Best Params (RF):", grid_rf.best_params_, "\n")
    y_pred_rf = grid_rf.predict(X_test)
    print_metrics(y_test, y_pred_rf)

if __name__ == "__main__":
    main()
