.PHONY: setup run test lint clean

# Variables
PYTHON = python3
PIP = pip
STREAMLIT = streamlit
PYTEST = pytest

# Installation des dépendances
setup:
	$(PIP) install -r requirements.txt

# Installation en mode développement
setup-dev: setup
	$(PIP) install pytest pytest-cov black flake8 mypy

# Lancement de l'application
run:
	$(STREAMLIT) run app.py

# Exécution des tests
test:
	$(PYTEST) tests/ -v

# Tests avec couverture
test-cov:
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term

# Vérification du style de code
lint:
	black --check src/ tests/
	flake8 src/ tests/
	mypy src/

# Formatage du code
format:
	black src/ tests/

# Nettoyage des fichiers temporaires
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .cache/

# Nettoyage complet (inclut artifacts et logs)
clean-all: clean
	rm -rf artifacts/*/
	rm -rf logs/
	rm -rf .cache/

# Création d'un environnement virtuel
venv:
	$(PYTHON) -m venv venv
	@echo "Activez l'environnement avec: source venv/bin/activate"

# Export des requirements
freeze:
	$(PIP) freeze > requirements.txt

# Vérification de la santé du projet
health-check: lint test
	@echo "✅ Projet en bonne santé !"

# Démarrage rapide avec données d'exemple
demo:
	@echo "🚀 Démarrage de la démo avec données d'exemple..."
	$(STREAMLIT) run app.py

# Aide
help:
	@echo "Commandes disponibles:"
	@echo "  make setup      - Installer les dépendances"
	@echo "  make setup-dev  - Installer en mode développement"
	@echo "  make run        - Lancer l'application"
	@echo "  make test       - Exécuter les tests"
	@echo "  make test-cov   - Tests avec couverture"
	@echo "  make lint       - Vérifier le code"
	@echo "  make format     - Formater le code"
	@echo "  make clean      - Nettoyer les fichiers temporaires"
	@echo "  make clean-all  - Nettoyage complet"
	@echo "  make venv       - Créer un environnement virtuel"
	@echo "  make demo       - Démarrer avec les données d'exemple"
	@echo "  make health-check - Vérification complète"
