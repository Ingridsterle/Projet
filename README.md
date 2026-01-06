# HeartDisease MLOps & Zero-Knowledge Proof Simulation

## 1. Contexte et Motivation

L'utilisation croissante de l'intelligence artificielle dans le domaine médical, notamment pour la prédiction des maladies cardiovasculaires, nécessite des pipelines fiables, traçables et conformes aux régulations européennes (EU AI Act).  

Les systèmes de ML “à haut risque” doivent démontrer :
- Robustesse et performance stable,
- Qualité et représentativité des données,
- Traçabilité complète du cycle de vie du modèle,
- Capacité d’audit et de supervision humaine.

Ce projet illustre la mise en place d’un **pipeline MLOps complet**, intégrant la **simulation de Zero-Knowledge Proof (ZKP)** pour vérifier l’intégrité et la fiabilité des prédictions, sans divulguer les données sensibles des patients.

---

## 2. Architecture du Projet
│
├── data/
│ ├── raw/ # Dataset original (heart.csv)
│ └── processed/ # Données train/test générées automatiquement
├── models/ # Modèles entraînés (heart_model.pkl)
├── artifacts/ # Graphiques, CSV, preuves ZKP
├── preprocess.py # Nettoyage et préparation des données
├── train.py # Entraînement modèle + log MLflow
├── inference.py # Prédictions sur le jeu de test
├── trust_graphs.py # Visualisation des scores de vraisemblance
├── zkp_proof.py # Simulation de preuve ZKP
└── README.md

---

## 3. Méthodologie

1. **Prétraitement des données** (`preprocess.py`)  
   - Nettoyage et standardisation des colonnes,
   - Séparation en jeux `train` et `test` (80/20),
   - Sauvegarde automatique dans `data/processed`.

2. **Entraînement du modèle** (`train.py`)  
   - Réseau de neurones MLP pour régression,
   - Entraînement avec logging MLflow (paramètres, métriques, artefacts),
   - Sauvegarde du modèle (`models/heart_model.pkl`) et prédictions (`artifacts/predictions.csv`).

3. **Simulation de vraisemblance & ZKP** (`trust_graphs.py` + `zkp_proof.py`)  
   - Calcul des scores de vraisemblance **avec et sans ZKP**,
   - Génération automatique de graphiques comparatifs :
     - `trust_bar.png` : score global comparatif,
     - `trust_radar.png` : radar multi-critères (robustesse, traçabilité, intégrité),
     - `trust_hist.png` : distribution des scores,
   - Simulation d’une **preuve ZKP** dans `zkp_proof.py`, stockée en JSON (`artifacts/zkp_proof.json`).

4. **Prédiction et visualisation** (`inference.py`)  
   - Application du modèle sur `X_test`,
   - Affichage des premières prédictions pour validation rapide.

---

## 4. Artefacts générés

| Type                   | Emplacement                    | Contenu |
|------------------------|--------------------------------|---------|
| Modèle ML              | `models/heart_model.pkl`       | Modèle entraîné |
| Prédictions            | `artifacts/predictions.csv`    | y_true / y_pred |
| Graphiques vraisemblance| `artifacts/*.png`             | Comparatif score ZKP vs non-ZKP |
| Preuve ZKP simulée     | `artifacts/zkp_proof.json`     | Empreinte + scores simulés |
| MLflow                 | `mlruns/`                      | Logging complet des runs et artefacts |

---

## 5. Exécution du Pipeline

```bash
# 1️ Préprocessing
python preprocess.py

# 2️ Entraînement du modèle
python train.py

# 3️ Graphiques de vraisemblance
python trust_graphs.py

# 4️ Génération de la preuve ZKP
python zkp_proof.py

# 5️ Prédictions finales
python inference.py


Auteur : Ingrid Sterle – Ingénierie MLOps appliquée à la santé
Date : Janvier 2026