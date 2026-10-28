"""
Titanic - Machine Learning from Disaster
Convertido do notebook do Kaggle (notebook5c61d3a726, versão 2) para script do VS Code.

Antes de rodar:
  1. Instale as bibliotecas:  pip install pandas numpy scikit-learn
  2. Baixe train.csv e test.csv da competição Titanic no Kaggle
     (https://www.kaggle.com/competitions/titanic/data)
     e coloque numa pasta "data" ao lado deste arquivo.

Estrutura esperada:
  app.py
  data/
    train.csv
    test.csv
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Pasta dos dados (no Kaggle era /kaggle/input/competitions/titanic/)
DATA_DIR = Path(__file__).parent / "data"

TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
COMMON_TITLES = ["Mr", "Mrs", "Miss", "Master"]


# Extrai o título do nome (Mr, Mrs, Miss, Master...)
def extract_title(names):
    title = names.str.extract(r" ([A-Za-z]+)\.", expand=False).replace(TITLE_MAP)
    return title.where(title.isin(COMMON_TITLES), "Rare")


# ---------------------------------------------------------------
# 1. Carregar e preparar os dados de treino
# ---------------------------------------------------------------
train_data = pd.read_csv(DATA_DIR / "train.csv")
test_data = pd.read_csv(DATA_DIR / "test.csv")
print(train_data.head())

for df in (train_data, test_data):
    df["Title"] = extract_title(df.Name)
    df["FamilySize"] = df.SibSp + df.Parch + 1
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

age_by_title = train_data.groupby("Title").Age.median()
fare_median = train_data["Fare"].median()

for df in (train_data, test_data):
    # Preenche idades faltantes com a mediana da idade de cada título
    df["Age"] = df["Age"].fillna(df["Title"].map(age_by_title))
    df["Fare"] = df["Fare"].fillna(fare_median)

title_cols = [f"Title_{t}" for t in COMMON_TITLES + ["Rare"]]


def build_features(df):
    dummies = pd.get_dummies(df["Title"], prefix="Title", dtype=int)
    dummies = dummies.reindex(columns=title_cols, fill_value=0)
    base = df[["Age", "Sex", "Pclass", "SibSp", "Parch", "Fare", "FamilySize"]]
    return pd.concat([base, dummies], axis=1)


X = build_features(train_data)
y = train_data["Survived"]
X_test = build_features(test_data)

# ---------------------------------------------------------------
# 2. Treinar o modelo
# ---------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
scores = cross_val_score(rf, X, y, cv=cv, scoring="accuracy")
print(f"Acurácia por fold: {scores.round(4)}")
print(f"Acurácia média (CV): {scores.mean():.4f} ± {scores.std():.4f}")

rf.fit(X, y)

# ---------------------------------------------------------------
# 3. Prever no conjunto de teste
# ---------------------------------------------------------------
result = rf.predict(X_test)

# ---------------------------------------------------------------
# 4. Gerar o arquivo de submissão
# ---------------------------------------------------------------
out_put = pd.DataFrame({"PassengerId": test_data.PassengerId, "Survived": result})
out_put.to_csv(Path(__file__).parent / "submission.csv", index=False)
print("Success")
