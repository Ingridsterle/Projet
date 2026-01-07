# train.py
import pandas as pd
from sklearn.neural_network import MLPRegressor

# Charger les données
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Entraîner le modèle
model = MLPRegressor(hidden_layer_sizes=(10,10), max_iter=500, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Prédictions
y_pred = model.predict(X_test)

# Sauvegarde temporaire pour l'étape suivante
import pickle, os
os.makedirs("models", exist_ok=True)
with open("models/mlp_model.pkl", "wb") as f:
    pickle.dump(model, f)

pd.DataFrame({'y_true': y_test.values.ravel(), 'y_pred': y_pred}).to_csv("data/processed/predictions_raw.csv", index=False)

print("✅ Modèle entraîné et prédictions sauvegardées")
