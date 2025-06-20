# Utilise une image de base Python
FROM python:3.10

# Crée un répertoire de travail
WORKDIR /app

# Copie les fichiers du projet
COPY . /app

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Spécifie le port écouté par l'application
EXPOSE 80

# Lance l'API FastAPI avec Uvicorn sur le port 80
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "80"]