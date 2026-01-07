# mlflow_log.py
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Charger CSV comparatif
df = pd.read_csv("artifacts/predictions_comparison.csv")

# Séparer sans et avec ZKP
df_sans = df[df["mode"]=="sans_zkp"]
df_avec = df[df["mode"]=="avec_zkp"]

# Calcul métriques
mse = mean_squared_error(df_sans["y_true"], df_sans["y_pred"])
r2 = r2_score(df_sans["y_true"], df_sans["y_pred"])
ll_sans = df_sans["log_likelihood"].mean()
ll_avec = df_avec["log_likelihood"].mean()

mlflow.set_experiment("HeartDiseasePipeline_ZKP")
with mlflow.start_run():
    # Paramètres
    mlflow.log_param("hidden_layers", (10,10))
    mlflow.log_param("max_iter", 500)
    # Métriques
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("log_likelihood_sans_zkp", ll_sans)
    mlflow.log_metric("log_likelihood_avec_zkp", ll_avec)
    # Artefacts
    mlflow.log_artifact("artifacts/predictions_comparison.csv")
    mlflow.log_artifact("artifacts/log_likelihood_comparison.png")

print(" Tout loggé dans MLflow")
