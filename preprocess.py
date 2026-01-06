import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Charger dataset brut
df = pd.read_csv("data/raw/heart.csv", sep=";")

# Nettoyer colonnes
df.columns = df.columns.str.strip().str.lower()

# Séparer features et cible
X = df.drop("target", axis=1)
y = df["target"]

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer dossier processed
os.makedirs("data/processed", exist_ok=True)

# Sauvegarder fichiers
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("✅ Données traitées et sauvegardées")
