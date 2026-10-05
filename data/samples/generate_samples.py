"""Generate small synthetic demo datasets (no downloads needed).

    python data/samples/generate_samples.py

Creates:
  customer_churn.csv   - tabular binary classification (target: churn)
  house_prices.csv     - tabular regression            (target: price)
  product_reviews.csv  - text classification           (target: sentiment, text: review_text)
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
rng = np.random.default_rng(7)


def churn(n: int = 1500) -> pd.DataFrame:
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.2]
    )
    tenure = rng.integers(1, 72, n)
    charges = rng.normal(70, 25, n).clip(18, 150).round(2)
    calls = rng.poisson(1.5, n)
    internet = rng.choice(["dsl", "fiber", "none"], n, p=[0.35, 0.45, 0.2])
    logit = (
        -1.0
        + 1.4 * (contract == "month-to-month")
        - 0.035 * tenure
        + 0.018 * (charges - 70)
        + 0.45 * calls
        + 0.5 * (internet == "fiber")
    )
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))
    df = pd.DataFrame(
        {
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "age": rng.integers(18, 80, n),
            "tenure_months": tenure,
            "monthly_charges": charges,
            "contract": contract,
            "payment_method": rng.choice(
                ["card", "bank_transfer", "e_check", "mailed_check"], n
            ),
            "internet_service": internet,
            "support_calls": calls,
            "churn": np.where(churned, "yes", "no"),
        }
    )
    df.loc[rng.random(n) < 0.04, "monthly_charges"] = np.nan
    df.loc[rng.random(n) < 0.03, "payment_method"] = np.nan
    return df


def houses(n: int = 1000) -> pd.DataFrame:
    sqft = rng.normal(1800, 600, n).clip(450, 5000).round()
    beds = np.clip((sqft / 600 + rng.normal(0, 0.8, n)).round(), 1, 7).astype(int)
    baths = np.clip((beds * 0.6 + rng.normal(0, 0.5, n)).round(1), 1, 5)
    age = rng.integers(0, 90, n)
    hood = rng.choice(["downtown", "suburb", "riverside", "hills", "industrial"], n)
    hood_mult = (
        pd.Series(hood)
        .map(
            {
                "downtown": 1.35,
                "suburb": 1.0,
                "riverside": 1.2,
                "hills": 1.5,
                "industrial": 0.75,
            }
        )
        .to_numpy()
    )
    garage = rng.choice([0, 1, 2], n, p=[0.2, 0.5, 0.3])
    price = (
        60_000 + 145 * sqft + 9_000 * baths - 650 * age + 12_000 * garage
    ) * hood_mult
    price = (price * rng.lognormal(0, 0.08, n)).round(-2)
    return pd.DataFrame(
        {
            "sqft": sqft,
            "bedrooms": beds,
            "bathrooms": baths,
            "age_years": age,
            "neighborhood": hood,
            "garage_spaces": garage,
            "price": price,
        }
    )


POS = [
    "love",
    "excellent",
    "works perfectly",
    "great value",
    "highly recommend",
    "exceeded my expectations",
    "fast shipping",
    "sturdy and well made",
    "five stars",
    "very happy with",
]
NEG = [
    "broke after a week",
    "terrible",
    "waste of money",
    "stopped working",
    "very disappointed",
    "poor quality",
    "would not recommend",
    "arrived damaged",
    "customer service ignored me",
    "refund",
]
NEU = [
    "it is okay",
    "does the job",
    "average quality",
    "as described",
    "nothing special",
    "fine for the price",
    "mixed feelings",
    "decent but",
    "not bad not great",
    "expected more features",
]
ITEMS = {
    "electronics": ["headphones", "charger", "keyboard", "speaker"],
    "home": ["blender", "lamp", "vacuum", "kettle"],
    "outdoor": ["tent", "backpack", "water bottle", "camping chair"],
}


def reviews(n: int = 1500) -> pd.DataFrame:
    rows = []
    for i in range(n):
        cat = rng.choice(list(ITEMS))
        item = rng.choice(ITEMS[cat])
        label = rng.choice(["positive", "negative", "neutral"], p=[0.45, 0.35, 0.2])
        bank = {"positive": POS, "negative": NEG, "neutral": NEU}[label]
        phrases = list(rng.choice(bank, 2, replace=False))
        if rng.random() < 0.25:  # label noise: borrow a phrase from another class
            phrases.append(rng.choice(NEU if label != "neutral" else POS))
        text = f"Bought this {item} last month. {phrases[0].capitalize()}, {' and '.join(phrases[1:])}."
        rows.append(
            {
                "review_id": i + 1,
                "product_category": cat,
                "review_text": text,
                "sentiment": label,
            }
        )
    return pd.DataFrame(rows)


def churn_leaky(n: int = 1500) -> pd.DataFrame:
    df = churn(n)
    # 1. Leaked post-outcome column with near-perfect predictive power
    df["cancellation_confirmation"] = np.where(df["churn"] == "yes", 1, 0)
    # 2. Near-unique identifier column
    df["account_guid"] = [f"acc_{i:06d}_{rng.integers(1000, 9999)}" for i in range(n)]
    # 3. Duplicate rows (duplicate 50 rows)
    dup_rows = df.iloc[:50].copy()
    return pd.concat([df, dup_rows], ignore_index=True)


if __name__ == "__main__":
    churn().to_csv(OUT / "customer_churn.csv", index=False)
    churn_leaky().to_csv(OUT / "customer_churn_leaky.csv", index=False)
    houses().to_csv(OUT / "house_prices.csv", index=False)
    reviews().to_csv(OUT / "product_reviews.csv", index=False)
    print("wrote samples to", OUT)
