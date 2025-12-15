"""
Application Streamlit principale - Teachable Machine.
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.config import config_manager, setup_tensorflow_config, setup_global_seed
from src.utils.logging import logger
from src.schemas.dataclasses import SessionState


def setup_page_config():
    """Configuration de la page Streamlit."""
    st.set_page_config(
        page_title="Teachable Machine Streamlit",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def initialize_app():
    """Initialisation de l'application."""
    # Charger la configuration
    config = config_manager.load_config()
    
    # Configurer TensorFlow
    setup_tensorflow_config(config)
    
    # Configurer les seeds
    setup_global_seed(config.app.seed)
    
    # Initialiser l'état de session
    if 'session_state' not in st.session_state:
        st.session_state.session_state = SessionState()
    
    # Sauvegarder la config dans la session
    config_manager.save_to_streamlit(config)
    
    return config


def create_sidebar():
    """Crée la barre latérale avec navigation."""
    st.sidebar.title("🤖 Teachable Machine")
    st.sidebar.markdown("---")
    
    # Indicateurs de progression
    st.sidebar.subheader("📋 Progression")
    
    # Vérifier l'état des étapes
    session_state = st.session_state.get('session_state')
    
    steps = {
        "📁 Données": bool(session_state and session_state.dataset_info),
        "🧪 Entraînement": bool(session_state and session_state.training_state),
        "📊 Évaluation": bool(session_state and session_state.selected_model_path),
        "🚀 Déploiement": bool(session_state and session_state.selected_model_path),
        "⚙️ Configuration": True
    }
    
    for step, completed in steps.items():
        if completed:
            st.sidebar.success(f"✅ {step}")
        else:
            st.sidebar.info(f"⏳ {step}")
    
    st.sidebar.markdown("---")
    
    # Informations sur la session
    if session_state and session_state.dataset_info:
        st.sidebar.subheader("📊 Dataset actuel")
        dataset_info = session_state.dataset_info
        st.sidebar.info(f"**Nom:** {getattr(dataset_info, 'name', 'N/A')}")
        st.sidebar.info(f"**Type:** {getattr(dataset_info, 'task_type', 'N/A')}")
        st.sidebar.info(f"**Échantillons:** {getattr(dataset_info, 'num_samples', 'N/A')}")
    
    if session_state and session_state.selected_model_path:
        st.sidebar.subheader("🤖 Modèle actuel")
        model_path = Path(session_state.selected_model_path)
        st.sidebar.info(f"**Modèle:** {model_path.parent.name}")
    
    st.sidebar.markdown("---")
    
    # Modèles sauvegardés
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        model_runs = list(artifacts_dir.glob("run_*"))
        if model_runs:
            st.sidebar.subheader(f"💾 Modèles ({len(model_runs)})")
            
            # Afficher les 3 plus récents
            recent_runs = sorted(model_runs, reverse=True)[:3]
            for run_dir in recent_runs:
                metadata_file = run_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        import json
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        accuracy = metadata.get('final_metrics', {}).get('accuracy', 0)
                        dataset_name = metadata.get('dataset_name', 'Unknown')
                        
                        st.sidebar.write(f"**{run_dir.name[-8:]}** ({accuracy:.1%})")  # Derniers 8 caractères
                        st.sidebar.caption(f"{dataset_name}")
                    except:
                        st.sidebar.write(f"**{run_dir.name[-8:]}**")
                else:
                    st.sidebar.write(f"**{run_dir.name[-8:]}**")
            
            if len(model_runs) > 3:
                st.sidebar.caption(f"... et {len(model_runs) - 3} autres")
            
            if st.sidebar.button("🗂️ Gérer les modèles", use_container_width=True):
                st.switch_page("pages/5_⚙️_Settings_&_Logs.py")
    
    st.sidebar.markdown("---")
    
    # Liens rapides
    st.sidebar.subheader("🔗 Navigation rapide")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("📁 Données", use_container_width=True):
            st.switch_page("pages/1_📁_Data_&_Labelling.py")
        if st.button("📊 Évaluation", use_container_width=True):
            st.switch_page("pages/3_📊_Evaluate_&_Explain.py")
    
    with col2:
        if st.button("🧪 Entraînement", use_container_width=True):
            st.switch_page("pages/2_🧪_Experiment_&_Train.py")
        if st.button("🚀 Déploiement", use_container_width=True):
            st.switch_page("pages/4_🚀_Deploy_&_Realtime.py")
    
    if st.sidebar.button("⚙️ Configuration", use_container_width=True):
        st.switch_page("pages/5_⚙️_Settings_&_Logs.py")


def main():
    """Fonction principale."""
    setup_page_config()
    config = initialize_app()
    create_sidebar()
    
    # Contenu principal
    st.title("🤖 Teachable Machine Streamlit")
    st.markdown("### Créez, entraînez et déployez vos modèles d'IA facilement")
    
    # Description
    st.markdown("""
    **Teachable Machine Streamlit** est une application complète pour créer des modèles de classification 
    et de régression d'images sans code complexe. Inspirée de Teachable Machine de Google, cette version 
    vous offre plus de contrôle et de fonctionnalités avancées.
    """)
    
    # Fonctionnalités principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📁 **Gestion des données**
        - Import de dossiers d'images
        - Support ZIP et CSV
        - Labelling interactif
        - Split automatique train/val/test
        """)
    
    with col2:
        st.markdown("""
        #### 🧪 **Entraînement**
        - Modèles pré-entraînés (MobileNet, EfficientNet, ResNet)
        - Transfer learning optimisé
        - Augmentation de données
        - Suivi en temps réel
        """)
    
    with col3:
        st.markdown("""
        #### 🚀 **Déploiement**
        - Prédictions en temps réel
        - Export ONNX
        - API FastAPI générée
        - Explicabilité (Grad-CAM)
        """)
    
    st.markdown("---")
    
    # Guide de démarrage rapide
    st.subheader("🚀 Démarrage rapide")
    
    tab1, tab2, tab3 = st.tabs(["Classification", "Régression", "Démo"])
    
    with tab1:
        st.markdown("""
        **Pour créer un modèle de classification :**
        
        1. **📁 Données** : Importez un dossier avec vos images organisées par classes
        2. **🧪 Entraînement** : Choisissez un modèle et lancez l'entraînement
        3. **📊 Évaluation** : Analysez les performances et l'explicabilité
        4. **🚀 Déploiement** : Testez et exportez votre modèle
        """)
        
        if st.button("🎯 Commencer avec la classification", type="primary"):
            st.switch_page("pages/1_📁_Data_&_Labelling.py")
    
    with tab2:
        st.markdown("""
        **Pour créer un modèle de régression :**
        
        1. **📁 Données** : Importez un CSV avec les chemins d'images et valeurs cibles
        2. **🧪 Entraînement** : Configurez pour la régression et entraînez
        3. **📊 Évaluation** : Analysez MAE, MSE, R² et graphiques résiduels
        4. **🚀 Déploiement** : Prédisez des valeurs continues
        """)
        
        if st.button("📈 Commencer avec la régression", type="primary"):
            st.switch_page("pages/1_📁_Data_&_Labelling.py")
    
    with tab3:
        st.markdown("""
        **Tester avec les données d'exemple :**
        
        Le dossier `samples/` contient un mini-dataset de démonstration pour tester rapidement 
        l'application sans avoir à préparer vos propres données.
        """)
        
        if st.button("🎮 Charger la démo", type="primary"):
            # Charger automatiquement les données d'exemple
            st.session_state.demo_mode = True
            st.switch_page("pages/1_📁_Data_&_Labelling.py")
    
    st.markdown("---")
    
    # Statistiques et informations système
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Informations système")
        
        # Vérifier TensorFlow
        try:
            import tensorflow as tf
            tf_version = tf.__version__
            gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            
            st.success(f"✅ TensorFlow {tf_version}")
            if gpu_available:
                st.success("✅ GPU disponible")
            else:
                st.info("ℹ️ CPU uniquement")
        except ImportError:
            st.error("❌ TensorFlow non installé")
        
        # Vérifier les autres dépendances
        dependencies = [
            ("streamlit", "streamlit"),
            ("pandas", "pandas"), 
            ("numpy", "numpy"),
            ("scikit-learn", "sklearn"),
            ("PIL", "PIL")
        ]
        
        for name, module in dependencies:
            try:
                __import__(module)
                st.success(f"✅ {name}")
            except ImportError:
                st.error(f"❌ {name}")
    
    with col2:
        st.subheader("💾 Cache et stockage")
        
        # Informations sur le cache
        from src.utils.cache import default_cache_manager
        
        cache_files, cache_size = default_cache_manager.get_cache_size()
        cache_size_mb = cache_size / (1024 * 1024)
        
        st.metric("Fichiers en cache", cache_files)
        st.metric("Taille du cache", f"{cache_size_mb:.1f} MB")
        
        if st.button("🗑️ Vider le cache"):
            deleted = default_cache_manager.clear()
            st.success(f"Cache vidé : {deleted} fichiers supprimés")
            st.rerun()
        
        # Informations sur les artifacts
        artifacts_dir = Path("artifacts")
        if artifacts_dir.exists():
            runs = list(artifacts_dir.glob("run_*"))
            st.metric("Entraînements sauvegardés", len(runs))
        else:
            st.metric("Entraînements sauvegardés", 0)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🤖 Teachable Machine Streamlit - Créé avec ❤️ et Streamlit<br>
        <small>Version 1.0.0 | MIT License</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
