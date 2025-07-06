# Datei: src/preprocess.py

import json
import os
import gzip
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Stelle sicher, dass du Stopwords und WordNet-Data hast:
# python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

CATEGORIES = ["Books", "Baby_Products", "Home_and_Kitchen"]
# Regex-Tokenizer: nur alphanumerische Zeichen, keine Satzzeichen
TOKENIZER   = RegexpTokenizer(r'\w+')
# Stopwords Deutsch und Englisch
STOPWORDS   = set(stopwords.words("german")) | set(stopwords.words("english"))
LEMMA       = WordNetLemmatizer()

def clean_text(text):
    # 1) Kleinschreibung
    text = text.lower()
    # 2) Tokenisierung per Regex
    tokens = TOKENIZER.tokenize(text)
    # 3) Stopwords und reine Zifferntokens rausfiltern, dann Lemmatisieren
    clean_tokens = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        # Nur Wörter, keine reinen Zahlen
        if t.isdigit():
            continue
        # Lemmatisieren
        lemma = LEMMA.lemmatize(t)
        clean_tokens.append(lemma)
    return clean_tokens

def preprocess_category(cat):
    in_path  = os.path.join("..", "data", "raw",   f"{cat}_sample.jsonl")
    out_dir  = os.path.join("..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cat}_processed.jsonl")

    print(f"→ Preprocessing {cat}…")
    with open(in_path, encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            rec    = json.loads(line)
            text   = rec.get("text", "") or ""
            tokens = clean_text(text)
            # Nur speichern, wenn mindestens 3 Tokens übrig sind
            if len(tokens) < 3:
                continue
            out = {
                "rating": rec["rating"],
                "tokens": tokens
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"✓ {cat} fertig: {out_path}")

def main():
    for c in CATEGORIES:
        preprocess_category(c)

if __name__ == "__main__":
    main()
