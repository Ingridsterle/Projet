# inference.py
import pandas as pd
import pickle

with open("models/mlp_model.pkl", "rb") as f:
    model = pickle.load(f)

X_test = pd.read_csv("data/processed/X_test.csv")
y_pred = model.predict(X_test)

print("Prédictions effectuées")
print(y_pred[:10])
