# 🤖 Teachable Machine Streamlit

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.13+-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Une application Streamlit complète et modulaire pour créer, entraîner, évaluer et déployer des modèles de classification et de régression d'images, inspirée de Teachable Machine de Google mais avec plus de contrôle et de fonctionnalités avancées.

## 🚀 Démo en ligne

Vous pouvez essayer l'application directement : [🔗 **Lancer la démo**](https://teachable-machine-streamlit.streamlit.app) *(lien à ajouter après déploiement)*

## ✨ Fonctionnalités principales

### 📁 Gestion des données
- Import de dossiers d'images organisés par classes
- Support des archives ZIP
- Datasets de régression via CSV
- Labelling et organisation automatique
- Split automatique train/validation/test (stratifié)
- Validation et statistiques des données

### 🧪 Entraînement
- **Modèles pré-entraînés** : MobileNetV3, EfficientNet, ResNet50
- **Transfer learning** optimisé avec fine-tuning configurable
- **Augmentation de données** : rotation, flip, luminosité, zoom, mixup/cutmix
- **Optimiseurs** : Adam, SGD, RMSprop avec schedulers
- **Callbacks** : Early stopping, réduction LR, sauvegarde automatique
- **Suivi en temps réel** : métriques et courbes d'apprentissage
- **Presets** : configurations rapide/équilibré/précis

### 📊 Évaluation et explicabilité
- **Classification** : Accuracy, Precision, Recall, F1, ROC/AUC, matrice de confusion
- **Régression** : MAE, MSE, RMSE, R², MAPE, graphiques résiduels
- **Explicabilité** : Grad-CAM et Score-CAM (en cours d'implémentation)
- **Comparaison de modèles** et recommandations
- **Export des résultats** en CSV/PDF/HTML

### 🚀 Déploiement
- **Inférence temps réel** : upload, webcam, batch, URL
- **Export multi-format** : Keras, ONNX, TensorFlow Lite
- **Génération d'API** FastAPI automatique
- **Guide de déploiement** : local, cloud, mobile

### ⚙️ Configuration avancée
- Configuration centralisée via YAML
- Interface graphique pour tous les paramètres
- Système de presets et sauvegarde
- Cache intelligent et gestion des logs
- Support multilingue (FR/EN)

## 🚀 Installation et démarrage rapide

### Prérequis
- Python 3.10 ou supérieur
- 4GB RAM minimum (8GB recommandé)
- GPU optionnel mais recommandé pour l'entraînement

### Installation

1. **Cloner le repository**
```bash
git clone https://github.com/your-repo/teachable-machine-streamlit.git
cd teachable-machine-streamlit
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\\Scripts\\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Lancer l'application**
```bash
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`

### Démarrage avec Make
```bash
make setup    # Installation des dépendances
make run      # Lancement de l'application
make demo     # Démarrage avec données d'exemple
```

## 📖 Guide d'utilisation

### 1. Classification d'images

1. **📁 Données** : 
   - Organisez vos images en dossiers par classe :
   ```
   mon_dataset/
   ├── chats/
   │   ├── chat1.jpg
   │   └── chat2.jpg
   └── chiens/
       ├── chien1.jpg
       └── chien2.jpg
   ```
   - Ou utilisez les données d'exemple intégrées

2. **🧪 Entraînement** :
   - Choisissez un modèle (MobileNetV3 recommandé pour débuter)
   - Sélectionnez un preset "Rapide" pour tester
   - Lancez l'entraînement et observez les métriques en temps réel

3. **📊 Évaluation** :
   - Analysez les métriques de performance
   - Examinez la matrice de confusion
   - Visualisez les explications Grad-CAM

4. **🚀 Déploiement** :
   - Testez sur de nouvelles images
   - Exportez le modèle au format souhaité
   - Générez une API REST automatiquement

### 2. Régression d'images

1. **📁 Données** :
   - Préparez un CSV avec chemins d'images et valeurs cibles :
   ```csv
   image_path,target
   images/img1.jpg,23.5
   images/img2.jpg,45.2
   ```

2. **🧪 Entraînement** :
   - Le type "régression" sera détecté automatiquement
   - Configurez selon vos besoins (MSE, MAE...)

3. **📊 Évaluation** :
   - Analysez R², RMSE, graphiques de résidus
   - Vérifiez la distribution des erreurs

### 3. Utilisation des données d'exemple

Pour tester rapidement l'application :

1. Cliquez sur "🎮 Charger la démo" sur la page d'accueil
2. Ou utilisez les boutons de création de datasets synthétiques
3. Les données d'exemple incluent :
   - Classification : 3 classes avec images synthétiques colorées
   - Régression : images avec luminosité corrélée à la valeur cible

## 🏗️ Architecture

### Structure du projet
```
teachable_machine_streamlit/
├── app.py                          # Application principale
├── pages/                          # Pages Streamlit
│   ├── 1_📁_Data_&_Labelling.py   # Gestion des données
│   ├── 2_🧪_Experiment_&_Train.py # Entraînement
│   ├── 3_📊_Evaluate_&_Explain.py # Évaluation
│   ├── 4_🚀_Deploy_&_Realtime.py  # Déploiement
│   └── 5_⚙️_Settings_&_Logs.py    # Configuration
├── src/                            # Code source modulaire
│   ├── data/                       # Chargement et transformation
│   ├── models/                     # Architectures et heads
│   ├── training/                   # Boucles d'entraînement
│   ├── evaluation/                 # Métriques et explicabilité
│   ├── serving/                    # Inférence et export
│   ├── utils/                      # Configuration, cache, logs
│   └── schemas/                    # Types et validation
├── artifacts/                      # Modèles et résultats sauvegardés
├── samples/                        # Données d'exemple
├── tests/                          # Tests unitaires
├── config.yaml                     # Configuration par défaut
└── requirements.txt                # Dépendances
```

### Modules principaux

- **src.data** : Chargement, validation et transformation des données
- **src.models** : Registre des modèles, transfer learning, têtes de classification/régression
- **src.training** : Entraînement avec callbacks, schedulers, et optimiseurs
- **src.evaluation** : Métriques, rapports, et explicabilité
- **src.serving** : Inférence, export multi-format, génération d'API
- **src.utils** : Configuration, cache, logs, et utilitaires

## 🔧 Configuration

### Configuration via l'interface

Utilisez la page "⚙️ Configuration" pour :
- Modifier les paramètres via interface graphique
- Éditer directement le YAML
- Appliquer des presets prédéfinis
- Exporter/importer des configurations

### Configuration manuelle

Éditez `config.yaml` pour personnaliser :

```yaml
# Exemple de configuration personnalisée
data:
  image_size: [224, 224]
  batch_size: 32
  validation_split: 0.2

training:
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"

model:
  backbone: "MobileNetV3Small"
  pretrained: true
  trainable_layers: 20

augmentation:
  enabled: true
  horizontal_flip: true
  rotation_range: 15
  brightness_range: [0.9, 1.1]
```

## 🚀 Déploiement

### Local avec FastAPI

1. Entraînez votre modèle dans l'application
2. Allez dans "🚀 Déploiement" > "Génération d'API"
3. Configurez et générez l'API
4. Lancez avec :
```bash
python serve_api.py
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Cloud (Heroku, AWS, GCP)

1. Utilisez le `Dockerfile` fourni
2. Configurez les variables d'environnement
3. Déployez via votre plateforme préférée

## 🧪 Tests

Exécutez les tests unitaires :

```bash
# Tests complets
make test

# Tests avec couverture
make test-cov

# Test d'un module spécifique
pytest tests/test_loaders.py -v
```

### Tests inclus

- **test_loaders.py** : Chargement et validation des données
- **test_trainer.py** : Entraînement sur données synthétiques
- **test_inference.py** : Prédictions et formats de sortie

## 📊 Métriques et monitoring

### Logs structurés

Les logs sont sauvegardés en JSON dans `logs/` avec :
- Timestamp, niveau, module
- Événements d'entraînement trackés
- Métriques et erreurs contextuelles

### Cache intelligent

- Cache automatique des datasets et images préprocessées
- Optimisation des performances avec `tf.data`
- Gestion intelligente de la mémoire

### Artefacts

Chaque entraînement sauvegarde dans `artifacts/run_YYYYMMDD_HHMMSS/` :
- Modèle final (`.keras`, `.h5`)
- Configuration complète
- Historique d'entraînement
- Métriques et graphiques
- Logs détaillés

## 🤝 Contribution

### Développement

1. Forkez le repository
2. Créez une branche feature : `git checkout -b feature/ma-nouvelle-fonctionnalite`
3. Installez les dépendances de développement : `make setup-dev`
4. Respectez le style de code : `make lint`
5. Ajoutez des tests : `make test`
6. Soumettez une Pull Request

### Standards de code

- **Black** pour le formatage
- **Flake8** pour la qualité
- **MyPy** pour le typage
- **Pytest** pour les tests
- Documentation des fonctions publiques

## 📝 Roadmap

### Version 1.1
- [ ] Grad-CAM et Score-CAM complets
- [ ] Support PyTorch via adaptateur
- [ ] Augmentation avancée (Albumentations)
- [ ] Interface de labelling interactif

### Version 1.2
- [ ] MLflow tracking optionnel
- [ ] Batch inference avec export CSV
- [ ] Calibration des modèles (ECE)
- [ ] Support modèles personnalisés

### Version 2.0
- [ ] Support multimodal (texte + images)
- [ ] AutoML et recherche d'hyperparamètres
- [ ] Déploiement edge (TensorRT, etc.)
- [ ] Interface collaborative multi-utilisateurs

## 🐛 Problèmes connus

- La fonctionnalité webcam nécessite `streamlit-webrtc` (optionnel)
- Les très gros datasets (>10GB) peuvent nécessiter plus de RAM
- GPU requis pour les modèles ResNet sur de gros datasets

## 📄 License

MIT License - voir [LICENSE](LICENSE) pour les détails.

## 🙏 Remerciements

- **Google Teachable Machine** pour l'inspiration
- **Streamlit** pour le framework UI fantastique
- **TensorFlow/Keras** pour les modèles pré-entraînés
- La communauté open source pour les nombreuses bibliothèques utilisées

## 📞 Support

- 🐛 **Issues** : [GitHub Issues](https://github.com/your-repo/issues)
- 📧 **Email** : support@teachable-machine-streamlit.com
- 💬 **Discord** : [Serveur communautaire](https://discord.gg/teachable-machine)
- 📖 **Documentation** : [Wiki détaillé](https://github.com/your-repo/wiki)

---

**Créé avec ❤️ et Streamlit | Version 1.0.0**
