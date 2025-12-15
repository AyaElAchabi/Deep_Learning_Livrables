FROM python:3.10-slim

# Métadonnées
LABEL maintainer="teachable-machine-team@example.com"
LABEL version="1.0.0"
LABEL description="Teachable Machine Streamlit - Application complète d'IA"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Créer un utilisateur non-root
RUN useradd --create-home --shell /bin/bash app

# Définir le répertoire de travail
WORKDIR /app

# Copier les requirements et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Changer le propriétaire des fichiers
RUN chown -R app:app /app

# Créer les dossiers nécessaires
RUN mkdir -p artifacts logs .cache samples && \
    chown -R app:app artifacts logs .cache samples

# Basculer vers l'utilisateur non-root
USER app

# Exposer le port
EXPOSE 8501

# Vérification de santé
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Point d'entrée
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
