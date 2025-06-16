# Utilise une image de base Python
FROM python:3.10

# Crée un répertoire de travail
WORKDIR /app

# Copie les fichiers du projet
COPY . /app

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Lance le script principal
CMD ["python", "main.py"]