
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import joblib
import os
import sys
from scipy.sparse import hstack
from scipy.sparse import issparse
import mlflow.pyfunc


app = FastAPI()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))
from train import train  # pour relancer l’entraînement

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "model.pkl"))

def load_model():
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError("Le modèle n'a pas encore été entraîné.")
    return joblib.load(model_path)  # Retourne le tuple complet (model, mlb, vec_title, vec_body)


model, mlb, vec_title, vec_body = load_model()

class PredictRequest(BaseModel):
    text: str



@app.get("/")
def read_root():
    return {"message": "API is running 🚀"}



@app.post("/predict")
def predict(data: PredictRequest):
    try:
        model, mlb, vec_title, vec_body = load_model()
        X_title = vec_title.transform([data.text])
        X_body = vec_body.transform([data.text])
        from scipy.sparse import hstack
        X = hstack([X_title, X_body])
         # Vérifier si X est vide (aucune valeur non nulle)
        if issparse(X) and X.nnz == 0:
            return {"keywords": ["aucun mot clé"]} 
        y_pred = model.predict(X)
        labels_list = mlb.inverse_transform(y_pred)
        print("data", data.text)
        print("X", X)
        print("y_pred:", y_pred)
        print("labels_list:", labels_list)
        print("labels_list[0]:", labels_list[0])
        print("mlb classes:", mlb.classes_)
        if not labels_list or not labels_list[0]:
            return {"keywords": ["aucun mot clé"]}

        # Ici, vérifie si labels_list[0] est une string ou un tuple/list
        if isinstance(labels_list[0], str):
            keywords = [labels_list[0]]  # transformer string en liste d’un seul élément
        else:
            keywords = list(labels_list[0])

        return {"keywords": keywords}
        

    except Exception as e:
        print(f"Erreur dans la prédiction : {e}")  # <-- Ajoute ça pour débugger
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks):
    def retrain_and_reload():
        train()
        global vectorizer, model
        vectorizer, model = load_model()
    background_tasks.add_task(retrain_and_reload)
    return {"message": "Réentraînement lancé"}
