# run_trust.py
import pandas as pd
from trust_middleware import TrustMiddleware

df_pred = pd.read_csv("data/processed/predictions_raw.csv")
y_true = df_pred["y_true"].values
y_pred = df_pred["y_pred"].values

trust = TrustMiddleware(y_true, y_pred)
trust.apply_zkp()
csv_file, graph_file = trust.export_artifacts()
