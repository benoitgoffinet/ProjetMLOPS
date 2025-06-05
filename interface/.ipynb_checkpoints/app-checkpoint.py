import streamlit as st
import requests
import mlflow
from datetime import datetime

API_URL = "http://localhost:8000"

st.title("🔍 Prédiction de mots clés")

text_input = st.text_area("Entrez un texte à analyser")
if st.button("Prédire"):
    response = requests.post(f"{API_URL}/predict", json={"text": text_input})
    st.write(response.json())
    if response.ok:
        keywords = response.json()["keywords"]
        st.success(f"Mots clés : {', '.join(keywords)}")
    else:
        try:
            # Essaye de lire une erreur JSON propre
            error_message = response.json().get("detail", "Erreur inconnue.")
        except ValueError:
            # Si ce n’est pas du JSON, utilise le texte brut
            error_message = response.text
        st.error(f"Erreur : {error_message}")

st.header("⚙️ Réentraînement du modèle")
if st.button("Relancer l'entraînement"):
    response = requests.post(f"{API_URL}/retrain")
    if response.ok:
        st.info("Réentraînement lancé.")
    else:
        st.error("Erreur lors du réentraînement.")

def afficher_evaluation(run_id):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    st.subheader("📊 Évaluation du modèle")
    st.write("Run ID:", run_id)
    
    # Affichage des métriques principales
    st.write("### Metrics")
    st.json(run.data.metrics)

    # Affichage des paramètres (optionnel)
    st.write("### Parameters")
    st.json(run.data.params)

def afficher_artifacts(run_id):
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)

    st.write("### Artifacts")
    for artifact in artifacts:
        if artifact.path.endswith(".png"):
            st.image(f"{client._tracking_client._tracking_uri}/artifacts/{run_id}/{artifact.path}")
        elif artifact.path.endswith(".txt") or artifact.path.endswith(".json"):
            local_path = client.download_artifacts(run_id, artifact.path)
            with open(local_path, "r") as f:
                st.text(f.read())

run_id = st.text_input("🔍 Entrez un Run ID pour voir son évaluation")
if run_id:
    afficher_evaluation(run_id)
    afficher_artifacts(run_id)

with mlflow.start_run():
    ...
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    # Exemple : log d’un fichier texte de classification_report
    with open("report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact("report.txt")