# Document Intelligence Assistant

Assistant intelligent basé sur RAG (Retrieval-Augmented Generation) avec une interface moderne développée avec Streamlit et Cohere.

## Fonctionnalités

-   **Téléchargement de Documents**: Chargez plusieurs documents PDF simultanément.
-   **Traitement Intelligent**: Découpe et analyse automatique du texte des documents.
-   **Recherche Vectorielle**: Utilise FAISS pour une recherche par similarité efficace.
-   **Interface Conversationnelle**: Discutez avec vos documents via une interface chat moderne et intuitive.
-   **Mémoire Contextuelle**: Maintient le contexte de la conversation pour des réponses cohérentes.
-   **Design Moderne**: Interface utilisateur élégante avec des animations et un design responsive.

## Prérequis

-   Python 3.8+
-   Clé API Cohere

## Installation

1.  Naviguez vers le répertoire du projet :
    ```bash
    cd RAG_Chatbot
    ```

2.  Installez les dépendances requises :
    ```bash
    pip install -r requirements.txt
    ```

3.  Configurez vos variables d'environnement :
    -   Renommez `.env.example` en `.env`.
    -   Ajoutez votre clé API Cohere dans le fichier `.env` :
        ```
        COHERE_API_KEY=...
        ```
    -   Alternativement, vous pouvez entrer votre clé API directement dans la barre latérale de l'application.

## Utilisation

1.  Lancez l'application Streamlit :
    ```bash
    streamlit run app.py
    ```

2.  Ouvrez votre navigateur et accédez à l'URL fournie (généralement `http://localhost:8501`).

3.  **Barre Latérale** :
    -   Entrez votre clé API Cohere (si non configurée dans `.env`).
    -   Téléchargez vos documents PDF.
    -   Cliquez sur "🚀 Analyser les documents" pour créer la base vectorielle.

4.  **Chat** :
    -   Une fois le traitement terminé, commencez à poser des questions sur vos documents dans le champ de saisie du chat.

## Structure du Projet

-   `app.py`: Fichier principal de l'application Streamlit gérant l'interface utilisateur et la logique d'interaction.
-   `utils.py`: Contient les fonctions auxiliaires pour le chargement, le découpage, l'embedding des documents et la création de la chaîne de conversation.
-   `requirements.txt`: Liste des dépendances Python.
-   `.env.example`: Modèle pour les variables d'environnement.

## Personnalisations

L'interface a été personnalisée avec :
- **Thème de couleurs moderne** : Palette bleue avec des dégradés violets
- **Animations et effets** : Boutons interactifs avec effets de survol
- **Avatars personnalisés** : Émojis distincts pour l'utilisateur (👤) et l'assistant (🤖)
- **Messages en français** : Interface entièrement traduite
- **Design responsive** : S'adapte à différentes tailles d'écran
