import streamlit as st
import requests
import os

# Met ici l'URL publique de ton API Azure (avec / à la fin)
API_URL = "https://monprojetmlops-dfc9bqh8hkefh8g8.canadaeast-01.azurewebsites.net/"

DATA_PATH = "data/data.skl"

st.title("🔍 Prédiction de mots clés")

text_input = st.text_area("Entrez un texte à analyser")
if st.button("Prédire"):
    try:
        response = requests.post(f"{API_URL}predict", json={"text": text_input})
        if response.ok:
            keywords = response.json().get("keywords", [])
            if keywords:
                st.success(f"Mots clés : {', '.join(keywords)}")
            else:
                st.warning("Aucun mot clé trouvé.")
        else:
            error_message = response.json().get("detail", response.text)
            st.error(f"Erreur : {error_message}")
    except Exception as e:
        st.error(f"Erreur de connexion à l'API : {e}")

