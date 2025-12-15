"""
Page 5 - Configuration et logs.
"""

import streamlit as st
import sys
from pathlib import Path
import yaml

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import config_manager, get_config_schema, validate_config
from src.utils.cache import default_cache_manager
from src.utils.logging import logger
from src.schemas.dataclasses import ExperimentPreset


def setup_page():
    """Configuration de la page."""
    st.set_page_config(
        page_title="Configuration - Teachable Machine",
        page_icon="⚙️",
        layout="wide"
    )


def configuration_editor():
    """Éditeur de configuration."""
    st.subheader("⚙️ Configuration de l'application")
    
    # Charger la configuration actuelle
    current_config = config_manager.load_config()
    config_dict = current_config.dict()
    
    tab1, tab2, tab3 = st.tabs(["Configuration générale", "Éditeur YAML", "Presets"])
    
    with tab1:
        general_config_editor(config_dict)
    
    with tab2:
        yaml_config_editor(config_dict)
    
    with tab3:
        preset_manager()


def general_config_editor(config_dict):
    """Éditeur de configuration avec interface graphique."""
    st.markdown("**Configuration par interface graphique**")
    
    # Application
    with st.expander("🎯 Configuration de l'application", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            config_dict['app']['title'] = st.text_input(
                "Titre de l'application",
                value=config_dict['app']['title']
            )
            config_dict['app']['language'] = st.selectbox(
                "Langue",
                options=['fr', 'en'],
                index=0 if config_dict['app']['language'] == 'fr' else 1
            )
        
        with col2:
            config_dict['app']['seed'] = st.number_input(
                "Seed aléatoire",
                value=config_dict['app']['seed'],
                min_value=0
            )
            config_dict['app']['debug'] = st.checkbox(
                "Mode debug",
                value=config_dict['app']['debug']
            )
    
    # Données
    with st.expander("📊 Configuration des données"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Taille d'image
            height = st.number_input("Hauteur image", value=config_dict['data']['image_size'][0], min_value=32, max_value=1024)
            width = st.number_input("Largeur image", value=config_dict['data']['image_size'][1], min_value=32, max_value=1024)
            config_dict['data']['image_size'] = [height, width]
            
            config_dict['data']['channels'] = st.selectbox(
                "Canaux d'image",
                options=[1, 3, 4],
                index=[1, 3, 4].index(config_dict['data']['channels'])
            )
            
            config_dict['data']['batch_size'] = st.number_input(
                "Taille du batch",
                value=config_dict['data']['batch_size'],
                min_value=1, max_value=512
            )
        
        with col2:
            config_dict['data']['validation_split'] = st.slider(
                "Split validation",
                0.0, 0.5, config_dict['data']['validation_split']
            )
            
            config_dict['data']['test_split'] = st.slider(
                "Split test",
                0.0, 0.3, config_dict['data']['test_split']
            )
            
            config_dict['data']['normalization'] = st.selectbox(
                "Normalisation",
                options=['imagenet', '0-1'],
                index=0 if config_dict['data']['normalization'] == 'imagenet' else 1
            )
    
    # Entraînement
    with st.expander("🏋️ Configuration d'entraînement"):
        col1, col2 = st.columns(2)
        
        with col1:
            config_dict['training']['epochs'] = st.number_input(
                "Epochs",
                value=config_dict['training']['epochs'],
                min_value=1, max_value=1000
            )
            
            config_dict['training']['learning_rate'] = st.number_input(
                "Learning rate",
                value=config_dict['training']['learning_rate'],
                min_value=0.0001, max_value=1.0,
                format="%.4f"
            )
            
            config_dict['training']['optimizer'] = st.selectbox(
                "Optimiseur",
                options=['adam', 'sgd', 'rmsprop'],
                index=['adam', 'sgd', 'rmsprop'].index(config_dict['training']['optimizer'])
            )
        
        with col2:
            # Early stopping
            config_dict['training']['early_stopping']['patience'] = st.number_input(
                "Early stopping - Patience",
                value=config_dict['training']['early_stopping']['patience'],
                min_value=1, max_value=100
            )
            
            config_dict['training']['early_stopping']['monitor'] = st.selectbox(
                "Métrique à surveiller",
                options=['val_loss', 'val_accuracy', 'loss', 'accuracy'],
                index=['val_loss', 'val_accuracy', 'loss', 'accuracy'].index(
                    config_dict['training']['early_stopping']['monitor']
                )
            )
    
    # Modèle
    with st.expander("🤖 Configuration du modèle"):
        col1, col2 = st.columns(2)
        
        with col1:
            config_dict['model']['backbone'] = st.selectbox(
                "Architecture",
                options=['MobileNetV3Small', 'MobileNetV3Large', 'EfficientNetB0', 'ResNet50'],
                index=['MobileNetV3Small', 'MobileNetV3Large', 'EfficientNetB0', 'ResNet50'].index(
                    config_dict['model']['backbone']
                )
            )
            
            config_dict['model']['pretrained'] = st.checkbox(
                "Poids pré-entraînés",
                value=config_dict['model']['pretrained']
            )
        
        with col2:
            config_dict['model']['trainable_layers'] = st.number_input(
                "Couches entraînables",
                value=config_dict['model']['trainable_layers'],
                min_value=-1, max_value=200
            )
            
            config_dict['model']['dropout'] = st.slider(
                "Dropout",
                0.0, 0.8, config_dict['model']['dropout']
            )
    
    # Boutons de sauvegarde
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Sauvegarder", type="primary"):
            # Valider la configuration
            is_valid, errors = validate_config(config_dict)
            
            if is_valid:
                try:
                    from src.schemas.dataclasses import Config
                    new_config = Config(**config_dict)
                    config_manager.save_config(new_config)
                    config_manager.save_to_streamlit(new_config)
                    st.success("Configuration sauvegardée !")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur lors de la sauvegarde : {e}")
            else:
                st.error("Configuration invalide :")
                for error in errors:
                    st.error(f"• {error}")
    
    with col2:
        if st.button("🔄 Réinitialiser"):
            # Recharger la configuration par défaut
            config_manager._config = None
            st.rerun()
    
    with col3:
        if st.button("✅ Valider"):
            is_valid, errors = validate_config(config_dict)
            if is_valid:
                st.success("✅ Configuration valide")
            else:
                st.error("❌ Configuration invalide")
                for error in errors:
                    st.error(f"• {error}")


def yaml_config_editor(config_dict):
    """Éditeur YAML brut."""
    st.markdown("**Éditeur YAML avancé**")
    
    # Convertir en YAML pour l'édition
    yaml_str = yaml.dump(config_dict, default_flow_style=False, allow_unicode=True)
    
    # Éditeur de texte
    edited_yaml = st.text_area(
        "Configuration YAML",
        value=yaml_str,
        height=400,
        help="Éditez directement la configuration en YAML"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Valider YAML"):
            try:
                # Parser le YAML
                parsed_config = yaml.safe_load(edited_yaml)
                
                # Valider
                is_valid, errors = validate_config(parsed_config)
                
                if is_valid:
                    st.success("✅ YAML valide")
                else:
                    st.error("❌ Configuration invalide")
                    for error in errors:
                        st.error(f"• {error}")
                        
            except yaml.YAMLError as e:
                st.error(f"❌ YAML invalide : {e}")
    
    with col2:
        if st.button("💾 Sauvegarder YAML"):
            try:
                # Parser et valider
                parsed_config = yaml.safe_load(edited_yaml)
                is_valid, errors = validate_config(parsed_config)
                
                if is_valid:
                    from src.schemas.dataclasses import Config
                    new_config = Config(**parsed_config)
                    config_manager.save_config(new_config)
                    config_manager.save_to_streamlit(new_config)
                    st.success("Configuration YAML sauvegardée !")
                    st.rerun()
                else:
                    st.error("Configuration invalide, impossible de sauvegarder")
                    
            except Exception as e:
                st.error(f"Erreur : {e}")


def preset_manager():
    """Gestionnaire de presets."""
    st.markdown("**Gestionnaire de presets**")
    
    # Presets par défaut
    st.subheader("📋 Presets par défaut")
    
    presets = ExperimentPreset.get_default_presets()
    
    for preset in presets:
        with st.expander(f"🎯 {preset.name}"):
            st.markdown(f"**Description :** {preset.description}")
            st.markdown("**Modifications :**")
            
            for key, value in preset.config_overrides.items():
                st.write(f"• `{key}`: {value}")
            
            if st.button(f"Appliquer {preset.name}", key=f"apply_{preset.name}"):
                try:
                    updated_config = config_manager.apply_preset(preset.name)
                    config_manager.save_to_streamlit(updated_config)
                    st.success(f"Preset '{preset.name}' appliqué !")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur lors de l'application : {e}")
    
    # Presets personnalisés (placeholder)
    st.subheader("🛠️ Presets personnalisés")
    st.info("🚧 Fonctionnalité de presets personnalisés à implémenter")


def cache_management():
    """Gestion du cache."""
    st.subheader("💾 Gestion du cache")
    
    # Informations sur le cache
    cache_files, cache_size = default_cache_manager.get_cache_size()
    cache_size_mb = cache_size / (1024 * 1024)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Fichiers en cache", cache_files)
    with col2:
        st.metric("Taille totale", f"{cache_size_mb:.1f} MB")
    with col3:
        if st.button("🧹 Nettoyer le cache expiré"):
            deleted = default_cache_manager.cleanup_expired()
            st.success(f"{deleted} fichiers expirés supprimés")
            st.rerun()
    with col4:
        if st.button("🗑️ Vider tout le cache"):
            deleted = default_cache_manager.clear()
            st.success(f"Cache vidé : {deleted} fichiers supprimés")
            st.rerun()
    
    # Configuration du cache
    with st.expander("⚙️ Configuration du cache"):
        st.info("Configuration du cache à implémenter")


def logs_viewer():
    """Visualiseur de logs."""
    st.subheader("📋 Visualiseur de logs")
    
    logs_dir = Path("logs")
    
    if logs_dir.exists():
        # Lister les fichiers de logs
        log_files = list(logs_dir.glob("*.jsonl"))
        
        if log_files:
            selected_log = st.selectbox(
                "Sélectionnez un fichier de log",
                options=[f.name for f in log_files]
            )
            
            if selected_log:
                log_path = logs_dir / selected_log
                
                # Options d'affichage
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    show_lines = st.number_input("Nombre de lignes", value=50, min_value=10, max_value=1000)
                with col2:
                    log_level = st.selectbox("Niveau", options=["ALL", "ERROR", "WARNING", "INFO", "DEBUG"])
                with col3:
                    if st.button("🔄 Actualiser"):
                        st.rerun()
                
                # Lire et afficher les logs
                try:
                    with open(log_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Filtrer par niveau si nécessaire
                    if log_level != "ALL":
                        lines = [line for line in lines if log_level in line]
                    
                    # Prendre les dernières lignes
                    lines = lines[-show_lines:]
                    
                    # Afficher dans un text_area
                    log_content = ''.join(lines)
                    st.text_area("Contenu des logs", value=log_content, height=400)
                    
                except Exception as e:
                    st.error(f"Erreur lors de la lecture des logs : {e}")
        else:
            st.info("Aucun fichier de log trouvé")
    else:
        st.info("Répertoire de logs non trouvé")


def system_info():
    """Informations système."""
    st.subheader("💻 Informations système")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dépendances Python**")
        
        dependencies = [
            ("streamlit", "streamlit"),
            ("tensorflow", "tensorflow"),
            ("pandas", "pandas"),
            ("numpy", "numpy"),
            ("scikit-learn", "sklearn"),
            ("pillow", "PIL"),
            ("pyyaml", "yaml"),
            ("pydantic", "pydantic")
        ]
        
        for name, module in dependencies:
            try:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'N/A')
                st.success(f"✅ {name} {version}")
            except ImportError:
                st.error(f"❌ {name} non installé")
    
    with col2:
        st.markdown("**Ressources système**")
        
        # Informations sur TensorFlow et GPU
        try:
            import tensorflow as tf
            
            # Version TensorFlow
            st.info(f"TensorFlow: {tf.__version__}")
            
            # GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                st.success(f"✅ {len(gpus)} GPU(s) disponible(s)")
                for i, gpu in enumerate(gpus):
                    st.write(f"  • GPU {i}: {gpu.name}")
            else:
                st.info("ℹ️ Aucun GPU détecté (CPU uniquement)")
                
        except ImportError:
            st.error("❌ TensorFlow non disponible")
        
        # Informations sur les artifacts et modèles sauvegardés
        st.markdown("---")
        st.markdown("**🗂️ Modèles sauvegardés**")
        
        artifacts_dir = Path("artifacts")
        if artifacts_dir.exists():
            runs = sorted(list(artifacts_dir.glob("run_*")), reverse=True)  # Plus récents en premier
            
            if runs:
                st.metric("Modèles sauvegardés", len(runs))
                
                total_size = sum(
                    sum(f.stat().st_size for f in run_dir.rglob('*') if f.is_file())
                    for run_dir in runs
                )
                total_size_mb = total_size / (1024 * 1024)
                st.metric("Taille totale", f"{total_size_mb:.1f} MB")
                
                # Affichage des modèles récents
                with st.expander("📋 Détails des modèles sauvegardés", expanded=True):
                    for i, run_dir in enumerate(runs[:5]):  # 5 plus récents
                        with st.container():
                            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                            
                            # Lire les métadonnées si disponibles
                            metadata_file = run_dir / "metadata.json"
                            if metadata_file.exists():
                                try:
                                    import json
                                    with open(metadata_file, 'r') as f:
                                        metadata = json.load(f)
                                    
                                    with col1:
                                        st.write(f"**{run_dir.name}**")
                                        st.caption(f"Dataset: {metadata.get('dataset_name', 'N/A')}")
                                    
                                    with col2:
                                        accuracy = metadata.get('final_metrics', {}).get('accuracy', 0)
                                        st.metric("Accuracy", f"{accuracy:.1%}")
                                    
                                    with col3:
                                        epochs = metadata.get('final_metrics', {}).get('epochs_completed', 0)
                                        st.metric("Epochs", epochs)
                                    
                                    with col4:
                                        if st.button("🗑️", key=f"delete_{run_dir.name}", help="Supprimer"):
                                            try:
                                                import shutil
                                                shutil.rmtree(run_dir)
                                                st.success(f"Modèle {run_dir.name} supprimé")
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Erreur: {e}")
                                    
                                except Exception as e:
                                    with col1:
                                        st.write(f"**{run_dir.name}**")
                                        st.caption("Métadonnées non lisibles")
                            else:
                                with col1:
                                    st.write(f"**{run_dir.name}**")
                                    st.caption("Ancien format")
                                with col4:
                                    if st.button("🗑️", key=f"delete_{run_dir.name}", help="Supprimer"):
                                        try:
                                            import shutil
                                            shutil.rmtree(run_dir)
                                            st.success(f"Modèle {run_dir.name} supprimé")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Erreur: {e}")
                            
                            st.markdown("---")
                
                # Actions de nettoyage
                if st.button("🧹 Nettoyer tous les anciens modèles", type="secondary"):
                    if st.confirm("Êtes-vous sûr de vouloir supprimer tous les modèles sauvegardés ?"):
                        try:
                            import shutil
                            for run_dir in runs:
                                shutil.rmtree(run_dir)
                            st.success(f"{len(runs)} modèles supprimés")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur lors du nettoyage: {e}")
            else:
                st.info("Aucun modèle sauvegardé pour le moment")
        else:
            st.info("Dossier artifacts non trouvé")


def export_import_config():
    """Export/Import de configuration."""
    st.subheader("📤 Export/Import de configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Export**")
        
        config = config_manager.load_config()
        config_yaml = yaml.dump(config.dict(), default_flow_style=False, allow_unicode=True)
        
        st.download_button(
            label="💾 Télécharger config.yaml",
            data=config_yaml,
            file_name="teachable_machine_config.yaml",
            mime="text/yaml"
        )
    
    with col2:
        st.markdown("**Import**")
        
        uploaded_config = st.file_uploader(
            "Sélectionnez un fichier de configuration",
            type=['yaml', 'yml']
        )
        
        if uploaded_config is not None:
            if st.button("📥 Importer la configuration"):
                try:
                    config_content = uploaded_config.read().decode('utf-8')
                    config_dict = yaml.safe_load(config_content)
                    
                    # Valider
                    is_valid, errors = validate_config(config_dict)
                    
                    if is_valid:
                        from src.schemas.dataclasses import Config
                        new_config = Config(**config_dict)
                        config_manager.save_config(new_config)
                        config_manager.save_to_streamlit(new_config)
                        st.success("Configuration importée avec succès !")
                        st.rerun()
                    else:
                        st.error("Configuration invalide :")
                        for error in errors:
                            st.error(f"• {error}")
                            
                except Exception as e:
                    st.error(f"Erreur lors de l'import : {e}")


def main():
    """Fonction principale de la page."""
    setup_page()
    
    st.title("⚙️ Configuration et logs")
    st.markdown("Configurez l'application et visualisez les logs")
    
    # Configuration
    configuration_editor()
    
    st.markdown("---")
    
    # Gestion du cache
    cache_management()
    
    st.markdown("---")
    
    # Visualiseur de logs
    logs_viewer()
    
    st.markdown("---")
    
    # Informations système
    system_info()
    
    st.markdown("---")
    
    # Export/Import
    export_import_config()


if __name__ == "__main__":
    main()
