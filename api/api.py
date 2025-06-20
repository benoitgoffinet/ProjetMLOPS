from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import joblib
import os
import sys
from scipy.sparse import hstack, issparse

app = FastAPI()

# Permet d'importer le fichier train.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))
from train import train  # La fonction d'entraînement

class PredictRequest(BaseModel):
    text: str

def load_model():
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Le modèle n'a pas encore été entraîné.")
    return joblib.load(model_path)  # Retourne le tuple (model, mlb, vec_title, vec_body)

@app.get("/")
def read_root():
    return {"message": "API is running 🚀"}

@app.post("/predict")
def predict(data: PredictRequest):
    try:
        model, mlb, vec_title, vec_body = load_model()
        X_title = vec_title.transform([data.text])
        X_body = vec_body.transform([data.text])
        X = hstack([X_title, X_body])

        if issparse(X) and X.nnz == 0:
            return {"keywords": ["aucun mot clé"]}

        y_pred = model.predict(X)
        labels_list = mlb.inverse_transform(y_pred)

        if not labels_list or not labels_list[0]:
            return {"keywords": ["aucun mot clé"]}

        if isinstance(labels_list[0], str):
            keywords = [labels_list[0]]
        else:
            keywords = list(labels_list[0])

        return {"keywords": keywords}
    except Exception as e:
        print(f"Erreur dans la prédiction : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(train)
    return {"message": "Training started in background"}
