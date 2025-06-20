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

