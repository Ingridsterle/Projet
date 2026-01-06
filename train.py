import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import os
import matplotlib.pyplot as plt

# Charger données
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

os.makedirs("models", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

mlflow.set_experiment("HeartDiseasePipeline")

with mlflow.start_run():

    # Entraîner modèle
    model = MLPRegressor(hidden_layer_sizes=(10,10), max_iter=500, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # Prédictions et métriques
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")

    mlflow.log_param("hidden_layers", (10,10))
    mlflow.log_param("max_iter", 500)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)

    # Sauvegarder modèle
    model_path = "models/heart_model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    # CSV de prédictions
    df_pred = pd.DataFrame({'y_true': y_test.values.ravel(), 'y_pred': y_pred})
    csv_path = "artifacts/predictions.csv"
    df_pred.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

print("✅ Training terminé et loggé")
