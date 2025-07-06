import gzip
import json
import random
import os

# Dieses Skript erzeugt aus großen JSONL.gz-Dateien pro Kategorie (Books, Baby_Products, Home_and_Kitchen)
# jeweils eine Sample-Datei mit 1000 Reviews pro Sterne-Klasse (1–5).

MAX_PER_CLASS = 1000
CATEGORIES = ["Books", "Baby_Products", "Home_and_Kitchen"]

for cat in CATEGORIES:
    counts = {str(i): 0 for i in range(1, 6)}
    in_path = os.path.join("..", "data", "raw", f"{cat}.jsonl.gz")
    out_path = os.path.join("..", "data", "raw", f"{cat}_sample.jsonl")

    # Öffnet komprimierte Datei zeilenweise und schreibt nur MAX_PER_CLASS Einträge pro Klasse
    with gzip.open(in_path, "rt", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        for line in fin:
            review = json.loads(line)
            raw = review.get("star_rating", review.get("rating"))

            # Konvertiere zu Float, dann zu Int; überspringe ungültige Werte
            try:
                rating_int = int(float(raw))
            except (TypeError, ValueError):
                continue

            # Nur Ratings zwischen 1 und 5 zulassen
            if rating_int < 1 or rating_int > 5:
                continue

            rating_str = str(rating_int)

            # Mit 5%-Chance auswählen und nur, solange Klasse nicht voll ist
            if counts[rating_str] < MAX_PER_CLASS and random.random() < 0.05:
                out_record = {
                    "rating": rating_int,
                    "text": review.get("review_body", review.get("text", ""))
                }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                counts[rating_str] += 1

            # Stop, wenn alle Klassen gefüllt sind
            if all(c >= MAX_PER_CLASS for c in counts.values()):
                break

    print(f"{cat} Samples erzeugt:", counts)
