import json
import pandas as pd
import os

def load_samples(category):
    path = os.path.join("..", "data", "raw", f"{category}_sample.jsonl")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            review = json.loads(line)
            text = review.get("text", "") or ""
            records.append({
                "Kategorie": category,
                "Sterne":    review.get("rating", None),
                "Wortanzahl": len(text.split())
            })
    return records

def main():
    categories = ["Books", "Baby_Products", "Home_and_Kitchen"]
    all_data = []

    for cat in categories:
        print(f"→ Lade Samples für {cat}...")
        all_data.extend(load_samples(cat))

    df = pd.DataFrame(all_data)

    # 1) Verteilung
    verteilung = df.groupby(["Kategorie", "Sterne"])["Sterne"].count()
    print("\n=== Verteilung der Bewertungen ===")
    print(verteilung)

    # 2) Durchschnittliche Wortanzahl
    laengen = df.groupby(["Kategorie", "Sterne"])["Wortanzahl"].mean()
    print("\n=== Durchschnittliche Wortanzahl ===")
    print(laengen)

if __name__ == "__main__":
    main()