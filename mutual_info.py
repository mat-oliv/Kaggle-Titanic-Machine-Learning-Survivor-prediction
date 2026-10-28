from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

DATA_DIR = Path(__file__).parent / "data"
SEEDS = range(10)

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

full = pd.concat([train.assign(is_test=0), test.assign(is_test=1)], ignore_index=True)

full["Title"] = full.Name.str.extract(r" ([A-Za-z]+)\.", expand=False)
full["Title"] = full["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
common = ["Mr", "Mrs", "Miss", "Master"]
full.loc[~full["Title"].isin(common), "Title"] = "Rare"

full["AgeMissing"] = full["Age"].isna().astype(int)
full["Age"] = full["Age"].fillna(full.groupby("Title").Age.transform("median"))
full["Fare"] = full["Fare"].fillna(full.groupby("Pclass").Fare.transform("median"))
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

full["FamilySize"] = full.SibSp + full.Parch + 1
full["IsAlone"] = (full.FamilySize == 1).astype(int)
full["HasCabin"] = full.Cabin.notna().astype(int)
full["Deck"] = full.Cabin.str[0].fillna("U")
full["TicketGroup"] = full.groupby("Ticket").Ticket.transform("count")

discrete = ["Sex", "Pclass", "Title", "Embarked", "SibSp", "Parch", "FamilySize",
            "IsAlone", "HasCabin", "Deck", "TicketGroup", "AgeMissing"]
continuous = ["Age", "Fare"]
features = discrete + continuous

X_full = full[features].copy()
for col in ["Sex", "Title", "Embarked", "Deck"]:
    X_full[col] = X_full[col].astype("category").cat.codes

is_train = full.is_test == 0
X = X_full[is_train]
y = full.loc[is_train, "Survived"].astype(int)
discrete_mask = [f in discrete for f in features]


def mi_with(X, target):
    runs = [mutual_info_classif(X, target, discrete_features=discrete_mask, random_state=s)
            for s in SEEDS]
    return np.mean(runs, axis=0)


p = y.mean()
h_y = -(p * np.log(p) + (1 - p) * np.log(1 - p))

mi = pd.Series(mi_with(X, y), index=features)

rng = np.random.default_rng(0)
null = np.array([mi_with(X, rng.permutation(y.values)) for _ in range(20)])
null_p95 = pd.Series(np.percentile(null, 95, axis=0), index=features)

report = pd.DataFrame({
    "MI (nats)": mi,
    "% de H(Survived)": 100 * mi / h_y,
    "ruído p95": null_p95,
    "nº valores": X.nunique(),
}).sort_values("MI (nats)", ascending=False)

print(f"Entropia de Survived: {h_y:.4f} nats  (taxa de sobrevivência = {p:.1%})\n")
print("=== Mutual information com Survived (train.csv) ===")
print(report.round(4).to_string())

binned = X_full.copy()
for col in continuous:
    binned[col] = pd.qcut(binned[col], q=10, labels=False, duplicates="drop")

nmi = pd.DataFrame(
    [[normalized_mutual_info_score(binned[a], binned[b]) for b in features] for a in features],
    index=features, columns=features,
)
pairs = (
    nmi.where(np.triu(np.ones(nmi.shape, dtype=bool), k=1))
    .stack()
    .sort_values(ascending=False)
)
print("\n=== Pares de features mais redundantes (NMI, train+test) ===")
print(pairs.head(10).round(3).to_string())

shift = pd.Series(
    {f: mutual_info_score(binned[f], full.is_test) for f in features}
).sort_values(ascending=False)
print("\n=== MI de cada feature com 'está no test.csv' (≈0 = mesma distribuição) ===")
print(shift.round(4).to_string())

combo = X.Sex.astype(str) + "_" + X.Pclass.astype(str)
print("\n=== MI de combinações com Survived ===")
print(f"Sex sozinho:         {mutual_info_score(X.Sex, y):.4f}")
print(f"Pclass sozinho:      {mutual_info_score(X.Pclass, y):.4f}")
print(f"Sex + Pclass juntos: {mutual_info_score(combo, y):.4f}")
title_cls = X.Title.astype(str) + "_" + X.Pclass.astype(str)
print(f"Title + Pclass:      {mutual_info_score(title_cls, y):.4f}")
