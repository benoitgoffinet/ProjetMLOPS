
print(f"Executing train.py from: {__file__}")

import re
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.base import BaseEstimator
from typing import Dict, Any
import mlflow
import mlflow.sklearn
import os
import time
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import joblib



def split_tags(tag_str):
    return tag_str.strip("<>").split("><")
    
# 📌 1. Chargement et vectorisation
def prepare_data():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "data.pkl"))

    # Ouverture du fichier
    with open(data_path, 'rb') as f:
        data = pickle.load(f)


    titles = data['Title']
    bodies = data['Body']
    tags_list = [split_tags(tag) for tag in data['Tags']]

    titles_train, titles_test, bodies_train, bodies_test, tags_train, tags_test = train_test_split(
    titles, bodies, tags_list, test_size=0.2, random_state=42
    )
    
         # 2. Préparation des tags (filtrage des top N)
    tag_counts = Counter(tag for tags in tags_train for tag in tags)
    N = 30
    most_common_tags = [tag for tag, _ in tag_counts.most_common(N)]

    filtered_tags_train = [[tag for tag in tags if tag in most_common_tags] for tags in tags_train]
    filtered_tags_test = [[tag for tag in tags if tag in most_common_tags] for tags in tags_test]

    mlb = MultiLabelBinarizer(classes=most_common_tags)
    print(mlb)
    
    y_train = mlb.fit_transform(filtered_tags_train)
    y_test = mlb.transform(filtered_tags_test)
  

    vectorizer_title_count = CountVectorizer(max_features=5000, ngram_range=(1,2))
    X_title_train_count = vectorizer_title_count.fit_transform(titles_train)
    X_title_test_count = vectorizer_title_count.transform(titles_test)

    vectorizer_body_count = CountVectorizer(max_features=10000, ngram_range=(1,2))
    X_body_train_count = vectorizer_body_count.fit_transform(bodies_train)
    X_body_test_count = vectorizer_body_count.transform(bodies_test)

    X_train_count = hstack([X_title_train_count, X_body_train_count])
    X_test_count = hstack([X_title_test_count, X_body_test_count])

    return X_train_count, X_test_count, y_train, y_test, mlb, vectorizer_title_count, vectorizer_body_count

# 📌 2. Tracking MLflow
def track_training_run(model, X_train, y_train, X_test, y_test, params, experiment_name, model_name):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Avant tout
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"🚀 Run MLflow lancé : {run_id}")

        mlflow.log_params(params)

        start_train = time.time()
        model.fit(X_train, y_train)
        end_train = time.time()
        mlflow.log_metric("train_time", end_train - start_train)

        start_pred = time.time()
        y_pred = model.predict(X_test)
        end_pred = time.time()
        mlflow.log_metric("predict_time", end_pred - start_pred)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        jacc = jaccard_score(y_test, y_pred, average="samples")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("jaccard_score", jacc)



        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.register_model(f"runs:/{run_id}/model", name=model_name)

        print("✅ Entraînement terminé et suivi par MLflow.")
        print(f"📍 Voir : http://127.0.0.1:5000/#/experiments/{run.info.experiment_id}/runs/{run_id}")

# 📌 3. Script principal
def train():
    X_train, X_test, y_train, y_test, mlb, vec_title, vec_body = prepare_data()

    # Modèle avec SGDClassifier dans un OneVsRest
    model = OneVsRestClassifier(SGDClassifier(loss='log_loss', max_iter=1000, random_state=42))

    # Entraînement
    model.fit(X_train, y_train)

    params = {
    "max_iter": 1000,
    "classifier": "SGDClassifiermlops",
    "strategy": "OneVsRest"
    }

    track_training_run(
    model=model,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    params=params,
    experiment_name="MyExperiment",
    model_name="OneVsRest_LogRegmlops")


    # 📦 Sauvegarder le modèle et les vectorizers dans ../modele/
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    
    joblib.dump((model, mlb, vec_title, vec_body), model_path)
    print(f"📦 Modèle sauvegardé dans '{model_path}'.")

if __name__ == "__main__":
    train()