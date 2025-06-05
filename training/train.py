
print(f"Executing train.py from: {__file__}")


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
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import joblib

# 📌 1. Chargement et vectorisation
def prepare_data():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "data.pkl"))

    # Ouverture du fichier
    with open(data_path, 'rb') as f:
        newquestion = pickle.load(f)

    tags_list = newquestion['Tags']
    N = 50
    tag_counts = Counter(tag for tags in tags_list for tag in tags)
    print(most_common_tags)
    most_common_tags = [tag for tag, _ in tag_counts.most_common(N)]
    filtered_tags_list = [[tag for tag in tags if tag in most_common_tags] for tags in tags_list]

    mlb = MultiLabelBinarizer(classes=most_common_tags)
    y = mlb.fit_transform(filtered_tags_list)

    titles = newquestion['NewTitle']
    bodies = newquestion['NewBody']

    vectorizer_title = CountVectorizer(max_features=5000)
    vectorizer_body = CountVectorizer(max_features=10000)
    X_title = vectorizer_title.fit_transform(titles)
    X_body = vectorizer_body.fit_transform(bodies)

    X = hstack([X_title, X_body])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, mlb, vectorizer_title, vectorizer_body

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
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mcm = multilabel_confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        global_cm = np.sum(mcm, axis=0)
        ax.imshow(global_cm, cmap='Blues')
        plt.title("Confusion Matrix")
        cm_path = "artifacts/confusion_matrix.png"
        os.makedirs("artifacts", exist_ok=True)
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.register_model(f"runs:/{run_id}/model", name=model_name)

        print("✅ Entraînement terminé et suivi par MLflow.")
        print(f"📍 Voir : http://127.0.0.1:5000/#/experiments/{run.info.experiment_id}/runs/{run_id}")

# 📌 3. Script principal
def train():
    X_train, X_test, y_train, y_test, mlb, vec_title, vec_body = prepare_data()

    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    params = {
        "max_iter": 1000,
        "classifier": "LogisticRegression",
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
        model_name="OneVsRest_LogReg"
    )

    # 📦 Sauvegarder le modèle et les vectorizers dans ../modele/
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pkl")
    
    joblib.dump((model, mlb, vec_title, vec_body), model_path)
    print(f"📦 Modèle sauvegardé dans '{model_path}'.")

if __name__ == "__main__":
    train()