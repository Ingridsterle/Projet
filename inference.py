import pandas as pd
import joblib

# Charger modèle
model = joblib.load("models/heart_model.pkl")

# Charger test
X_test = pd.read_csv("data/processed/X_test.csv")
y_pred = model.predict(X_test)

print("✅ Prédictions effectuées")
print(y_pred[:10])
