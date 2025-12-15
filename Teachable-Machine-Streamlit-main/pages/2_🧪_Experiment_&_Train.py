"""
Page 2 - Expérimentation et entraînement.
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import config_manager
from src.schemas.dataclasses import ExperimentPreset


def setup_page():
    """Configuration de la page."""
    st.set_page_config(
        page_title="Entraînement - Teachable Machine",
        page_icon="🧪",
        layout="wide"
    )


def check_data_ready():
    """Vérifie si les données sont prêtes."""
    if 'dataset_info' not in st.session_state:
        st.error("❌ Aucune donnée chargée")
        st.markdown("Veuillez d'abord charger vos données dans la page précédente.")
        if st.button("📁 Aller aux données"):
            st.switch_page("pages/1_📁_Data_&_Labelling.py")
        return False
    
    if not st.session_state.get('splits_configured'):
        st.warning("⚠️ Splits non configurés")
        st.markdown("Veuillez configurer les splits train/val/test.")
        if st.button("📁 Configurer les splits"):
            st.switch_page("pages/1_📁_Data_&_Labelling.py")
        return False
    
    return True


def task_configuration():
    """Configuration du type de tâche."""
    st.subheader("🎯 Configuration de la tâche")
    
    dataset_info = st.session_state.dataset_info
    
    # Afficher le type détecté
    st.info(f"Type détecté automatiquement : **{dataset_info.task_type.title()}**")
    
    if dataset_info.task_type == "classification":
        st.success(f"✅ Classification avec {dataset_info.num_classes} classes")
        
        # Afficher les classes
        st.markdown("**Classes :**")
        for i, class_name in enumerate(dataset_info.class_names):
            count = dataset_info.class_distribution.get(class_name, 0)
            st.write(f"{i+1}. {class_name} ({count} échantillons)")
    
    elif dataset_info.task_type == "regression":
        st.success("✅ Régression configurée")
        st.write(f"Plage des valeurs : {dataset_info.target_range[0]:.2f} - {dataset_info.target_range[1]:.2f}")
    
    return dataset_info.task_type


def model_selection():
    """Sélection du modèle."""
    st.subheader("🤖 Sélection du modèle")
    
    # Modèles disponibles
    models = {
        "MobileNetV3Small": {
            "description": "Léger et efficace (2.5M paramètres)",
            "speed": "⚡⚡⚡",
            "accuracy": "⭐⭐⭐"
        },
        "MobileNetV3Large": {
            "description": "Équilibré (5.4M paramètres)", 
            "speed": "⚡⚡",
            "accuracy": "⭐⭐⭐⭐"
        },
        "EfficientNetB0": {
            "description": "Excellent rapport qualité/taille (5.3M paramètres)",
            "speed": "⚡⚡",
            "accuracy": "⭐⭐⭐⭐⭐"
        },
        "ResNet50": {
            "description": "Classique et robuste (25.6M paramètres)",
            "speed": "⚡",
            "accuracy": "⭐⭐⭐⭐"
        }
    }
    
    selected_model = st.selectbox(
        "Choisissez une architecture",
        options=list(models.keys()),
        help="Sélectionnez le modèle selon vos besoins en vitesse/précision"
    )
    
    # Afficher les détails du modèle sélectionné
    model_info = models[selected_model]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vitesse", model_info["speed"])
    with col2:
        st.metric("Précision", model_info["accuracy"])
    with col3:
        st.info(model_info["description"])
    
    # Options du modèle
    with st.expander("⚙️ Options avancées du modèle"):
        pretrained = st.checkbox("Utiliser des poids pré-entraînés", value=True,
                                help="Recommandé pour de meilleures performances")
        
        trainable_layers = st.slider(
            "Couches entraînables", 
            min_value=-1, max_value=100, value=20,
            help="-1 = toutes, 0 = aucune (feature extraction), >0 = N dernières couches"
        )
        
        dropout = st.slider("Dropout", 0.0, 0.8, 0.2, 0.1,
                           help="Régularisation pour éviter le surapprentissage")
    
    return {
        'backbone': selected_model,
        'pretrained': pretrained,
        'trainable_layers': trainable_layers,
        'dropout': dropout
    }


def training_configuration():
    """Configuration de l'entraînement."""
    st.subheader("🏋️ Configuration de l'entraînement")
    
    # Presets rapides
    st.markdown("**Presets rapides :**")
    presets = ExperimentPreset.get_default_presets()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⚡ Rapide", help=presets[0].description):
            st.session_state.selected_preset = presets[0]
            st.rerun()
    
    with col2:
        if st.button("⚖️ Équilibré", help=presets[1].description):
            st.session_state.selected_preset = presets[1]
            st.rerun()
    
    with col3:
        if st.button("🎯 Précis", help=presets[2].description):
            st.session_state.selected_preset = presets[2]
            st.rerun()
    
    # Configuration manuelle
    st.markdown("---")
    st.markdown("**Configuration manuelle :**")
    
    # Utiliser le preset sélectionné comme base
    if 'selected_preset' in st.session_state:
        preset = st.session_state.selected_preset
        st.info(f"Preset '{preset.name}' sélectionné : {preset.description}")
        
        # Valeurs par défaut du preset
        default_epochs = preset.config_overrides.get('training.epochs', 30)
        default_lr = preset.config_overrides.get('training.learning_rate', 0.001)
        default_trainable = preset.config_overrides.get('model.trainable_layers', 20)
    else:
        default_epochs = 30
        default_lr = 0.001
        default_trainable = 20
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.number_input("Nombre d'epochs", 1, 200, default_epochs)
        learning_rate = st.number_input("Taux d'apprentissage", 0.0001, 0.1, default_lr, format="%.4f")
        batch_size = st.selectbox("Taille du batch", [8, 16, 32, 64], index=2)
    
    with col2:
        optimizer = st.selectbox("Optimiseur", ["adam", "sgd", "rmsprop"])
        
        # Early stopping
        early_stopping = st.checkbox("Early stopping", value=True)
        if early_stopping:
            patience = st.number_input("Patience", 1, 20, 10)
        else:
            patience = None
    
    # Augmentation de données
    with st.expander("🎨 Augmentation de données"):
        augmentation_enabled = st.checkbox("Activer l'augmentation", value=True)
        
        if augmentation_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                horizontal_flip = st.checkbox("Flip horizontal", value=True)
                rotation_range = st.slider("Rotation (degrés)", 0, 45, 15)
                
            with col2:
                brightness_range = st.slider("Variation luminosité", 0.0, 0.5, 0.1)
                zoom_range = st.slider("Zoom", 0.0, 0.3, 0.1)
    
    return {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'early_stopping': {
            'enabled': early_stopping,
            'patience': patience
        },
        'augmentation': {
            'enabled': augmentation_enabled,
            'horizontal_flip': horizontal_flip if augmentation_enabled else False,
            'rotation_range': rotation_range if augmentation_enabled else 0,
            'brightness_range': [1-brightness_range, 1+brightness_range] if augmentation_enabled else [1.0, 1.0],
            'zoom_range': zoom_range if augmentation_enabled else 0
        }
    }


def training_section():
    """Section d'entraînement."""
    st.subheader("🚀 Entraînement")
    
    if 'training_config' not in st.session_state or 'model_config' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord configurer le modèle et l'entraînement ci-dessus")
        return
    
    training_config = st.session_state.training_config
    model_config = st.session_state.model_config
    
    # Résumé de la configuration
    with st.expander("📋 Résumé de la configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Modèle :**")
            st.write(f"• Architecture : {model_config['backbone']}")
            st.write(f"• Pré-entraîné : {'Oui' if model_config['pretrained'] else 'Non'}")
            st.write(f"• Couches entraînables : {model_config['trainable_layers']}")
            st.write(f"• Dropout : {model_config['dropout']}")
        
        with col2:
            st.markdown("**Entraînement :**")
            st.write(f"• Epochs : {training_config['epochs']}")
            st.write(f"• Learning rate : {training_config['learning_rate']}")
            st.write(f"• Batch size : {training_config['batch_size']}")
            st.write(f"• Optimiseur : {training_config['optimizer']}")
    
    # Estimation du temps
    dataset_info = st.session_state.dataset_info
    estimated_time = estimate_training_time(
        dataset_info.num_samples, 
        training_config['epochs'], 
        training_config['batch_size']
    )
    
    st.info(f"⏱️ Temps d'entraînement estimé : {estimated_time}")
    
    # Bouton d'entraînement
    if st.button("🎯 Lancer l'entraînement", type="primary", use_container_width=True):
        
        # Placeholder pour la simulation d'entraînement
        st.info("🚧 Simulation d'entraînement (fonctionnalité complète à implémenter)")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        import time
        import random
        import numpy as np
        
        # Utiliser le nombre d'epochs configuré par l'utilisateur
        total_epochs = training_config['epochs']
        learning_rate = training_config['learning_rate']
        batch_size = training_config['batch_size']
        optimizer = training_config['optimizer']
        
        # Variables pour early stopping
        early_stopping_enabled = training_config['early_stopping']['enabled']
        patience = training_config['early_stopping']['patience']
        best_val_loss = float('inf')
        patience_counter = 0
        stopped_early = False
        
        # Paramètres de simulation réalistes basés sur la configuration
        # Learning rate influence la vitesse de convergence
        lr_factor = learning_rate / 0.001  # Normaliser autour de 0.001
        if lr_factor > 5:  # LR trop élevé
            convergence_speed = 0.3
            final_accuracy = 0.7  # Performance dégradée
            instability = 0.1  # Plus d'instabilité
        elif lr_factor < 0.1:  # LR trop faible
            convergence_speed = 0.1
            final_accuracy = 0.85
            instability = 0.02
        else:  # LR optimal
            convergence_speed = 0.6
            final_accuracy = 0.92
            instability = 0.03
        
        # Optimizer influence
        optimizer_bonus = {
            'adam': 0.05,      # Meilleure convergence
            'sgd': 0.0,        # Baseline
            'rmsprop': 0.02    # Légèrement mieux que SGD
        }.get(optimizer, 0.0)
        
        final_accuracy += optimizer_bonus
        
        # Batch size influence
        if batch_size < 16:
            instability += 0.05  # Plus de bruit avec petit batch
        elif batch_size > 64:
            convergence_speed *= 0.8  # Convergence plus lente
        
        # Génerer un seed basé sur les paramètres pour la reproductibilité
        import hashlib
        param_str = f"{learning_rate}_{batch_size}_{optimizer}_{total_epochs}"
        seed = int(hashlib.md5(param_str.encode()).hexdigest()[:8], 16) % 10000
        random.seed(seed)
        
        st.info(f"🎯 Simulation avec LR={learning_rate}, Batch={batch_size}, Optimizer={optimizer}")
        
        for epoch in range(1, total_epochs + 1):
            progress = epoch / total_epochs
            progress_bar.progress(progress)
            
            # Calculer les métriques basées sur les paramètres réels
            convergence_factor = 1 - np.exp(-convergence_speed * epoch / total_epochs)
            
            # Training loss - diminue selon la vitesse de convergence
            base_train_loss = 1.5 * (1 - convergence_factor) + 0.05
            train_loss = max(0.01, base_train_loss + random.uniform(-instability, instability))
            
            # Validation loss - légèrement plus élevée que training
            overfitting_factor = max(0, (epoch - total_epochs * 0.7) / (total_epochs * 0.3))
            val_loss_base = train_loss + 0.05 + overfitting_factor * 0.1
            val_loss = max(0.01, val_loss_base + random.uniform(-instability*0.5, instability))
            
            # Accuracy - basée sur la loss
            accuracy = min(final_accuracy, 
                          0.2 + (final_accuracy - 0.2) * convergence_factor + 
                          random.uniform(-instability*0.5, instability*0.5))
            accuracy = max(0.1, accuracy)
            
            # Logic early stopping
            if early_stopping_enabled:
                if val_loss < best_val_loss - 0.001:  # Amélioration significative
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    stopped_early = True
                    status_text.text(f"Early stopping à l'epoch {epoch}/{total_epochs} - "
                                   f"Pas d'amélioration depuis {patience} epochs")
                    time.sleep(1)
                    break
            
            status_text.text(f"Epoch {epoch}/{total_epochs} - "
                           f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                           f"Accuracy: {accuracy:.3f}")
            
            # Temps d'attente adaptatif (plus court pour de nombreux epochs)
            sleep_time = max(0.3, min(1.5, 8.0 / total_epochs))
            time.sleep(sleep_time)
        
        # Sauvegarder l'historique d'entraînement
        training_history = {
            'epochs_completed': epoch if stopped_early else total_epochs,
            'final_train_loss': train_loss,
            'final_val_loss': val_loss,
            'final_accuracy': accuracy,
            'stopped_early': stopped_early,
            'best_val_loss': best_val_loss,
            'config_used': training_config,
            'convergence_speed': convergence_speed,
            'final_accuracy_target': final_accuracy
        }
        
        if stopped_early:
            st.info(f"🛑 Entraînement arrêté par early stopping après {epoch} epochs (patience: {patience})")
            st.metric("Accuracy finale", f"{accuracy:.3f}")
            st.metric("Validation Loss finale", f"{val_loss:.4f}")
        else:
            st.success(f"✅ Entraînement terminé après {total_epochs} epochs (simulation) !")
            st.metric("Accuracy finale", f"{accuracy:.3f}")
            st.metric("Training Loss finale", f"{train_loss:.4f}")
            st.metric("Validation Loss finale", f"{val_loss:.4f}")
            
        st.session_state.model_trained = True
        st.session_state.training_history = training_history
        
        # Créer vraiment les artefacts de sauvegarde
        from datetime import datetime
        from pathlib import Path
        import json
        
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        st.session_state.last_run_id = run_id
        
        # Créer le dossier d'artefacts
        artifacts_dir = Path("artifacts") / run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder la configuration d'entraînement
        config_file = artifacts_dir / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump(training_config, f, indent=2, default=str)
        
        # Sauvegarder l'historique d'entraînement
        history_file = artifacts_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2, default=str)
        
        # Créer un modèle factice (placeholder)
        model_file = artifacts_dir / "model.keras"
        with open(model_file, 'w') as f:
            f.write(f"# Modèle simulé\n# Accuracy: {accuracy:.3f}\n# Config: {training_config}")
        
        # Créer des métadonnées
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "dataset_name": getattr(st.session_state.get('dataset_info'), 'name', 'Unknown'),
            "task_type": getattr(st.session_state.get('dataset_info'), 'task_type', 'classification'),
            "final_metrics": {
                "accuracy": accuracy,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epochs_completed": epoch if stopped_early else total_epochs,
                "stopped_early": stopped_early
            },
            "config_summary": {
                "learning_rate": training_config['learning_rate'],
                "optimizer": training_config['optimizer'],
                "batch_size": training_config['batch_size'],
                "epochs_target": training_config['epochs']
            }
        }
        
        metadata_file = artifacts_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Créer un README pour l'utilisateur
        readme_file = artifacts_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(f"""# Entraînement {run_id}

## Résultats
- **Accuracy finale**: {accuracy:.3f}
- **Training Loss**: {train_loss:.4f}
- **Validation Loss**: {val_loss:.4f}
- **Epochs terminés**: {epoch if stopped_early else total_epochs}/{total_epochs}
- **Early Stopping**: {'Oui' if stopped_early else 'Non'}

## Configuration
- **Learning Rate**: {training_config['learning_rate']}
- **Optimizer**: {training_config['optimizer']}
- **Batch Size**: {training_config['batch_size']}
- **Dataset**: {getattr(st.session_state.get('dataset_info'), 'name', 'Unknown')}

## Fichiers
- `model.keras`: Modèle entraîné (simulé)
- `training_config.json`: Configuration complète d'entraînement
- `training_history.json`: Historique détaillé de l'entraînement
- `metadata.json`: Métadonnées du run
""")
        
        st.success(f"✅ Modèle et artefacts sauvegardés dans : `artifacts/{run_id}/`")
        
        # Afficher la liste des fichiers créés
        with st.expander("📁 Fichiers créés"):
            for file in artifacts_dir.iterdir():
                file_size = file.stat().st_size
                st.write(f"• `{file.name}` ({file_size} bytes)")
        
        if st.button("📊 Voir les résultats"):
            st.switch_page("pages/3_📊_Evaluate_&_Explain.py")


def display_training_comparison():
    """Affiche une comparaison des entraînements précédents."""
    if 'training_history' not in st.session_state:
        return
    
    st.subheader("📈 Résumé de l'entraînement")
    
    history = st.session_state.training_history
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Epochs terminés", 
            history['epochs_completed'],
            delta=f"Target: {history['config_used']['epochs']}"
        )
    
    with col2:
        st.metric(
            "Accuracy finale", 
            f"{history['final_accuracy']:.3f}",
            delta=f"Target: {history['final_accuracy_target']:.3f}"
        )
    
    with col3:
        st.metric(
            "Val Loss finale", 
            f"{history['final_val_loss']:.4f}",
            delta=f"Best: {history['best_val_loss']:.4f}"
        )
    
    with col4:
        early_stop_text = "Oui" if history['stopped_early'] else "Non"
        st.metric("Early Stop", early_stop_text)
    
    # Détails de la configuration
    with st.expander("🔧 Configuration utilisée"):
        config = history['config_used']
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Learning Rate:** {config['learning_rate']}")
            st.write(f"**Batch Size:** {config['batch_size']}")
            st.write(f"**Optimizer:** {config['optimizer']}")
        
        with col2:
            st.write(f"**Epochs configurés:** {config['epochs']}")
            st.write(f"**Early Stopping:** {'Activé' if config['early_stopping']['enabled'] else 'Désactivé'}")
            if config['early_stopping']['enabled']:
                st.write(f"**Patience:** {config['early_stopping']['patience']}")
    
    # Conseils d'amélioration
    if history['final_accuracy'] < 0.7:
        st.warning("🔧 **Suggestions d'amélioration:**")
        if history['config_used']['learning_rate'] > 0.01:
            st.write("• Essayez un learning rate plus faible (0.001 - 0.01)")
        if history['stopped_early']:
            st.write("• Augmentez la patience pour l'early stopping")
        st.write("• Augmentez le nombre d'epochs")
    elif history['final_accuracy'] > 0.9:
        st.success("🎉 Excellent! Votre modèle converge bien.")
    

def estimate_training_time(num_samples, epochs, batch_size):
    """Estime le temps d'entraînement."""
    # Estimation très approximative
    steps_per_epoch = num_samples // batch_size
    seconds_per_step = 0.5  # Estimation
    
    total_seconds = steps_per_epoch * epochs * seconds_per_step
    
    if total_seconds < 60:
        return f"{total_seconds:.0f} secondes"
    elif total_seconds < 3600:
        return f"{total_seconds/60:.1f} minutes"
    else:
        return f"{total_seconds/3600:.1f} heures"


def main():
    """Fonction principale de la page."""
    setup_page()
    
    st.title("🧪 Expérimentation et entraînement")
    st.markdown("Configurez et entraînez votre modèle d'IA")
    
    # Vérifier que les données sont prêtes
    if not check_data_ready():
        return
    
    # Configuration de la tâche
    task_type = task_configuration()
    
    st.markdown("---")
    
    # Sélection du modèle
    model_config = model_selection()
    st.session_state.model_config = model_config
    
    st.markdown("---")
    
    # Configuration de l'entraînement
    training_config = training_configuration()
    st.session_state.training_config = training_config
    
    st.markdown("---")
    
    # Section d'entraînement
    training_section()
    
    st.markdown("---")
    
    # Comparaison des entraînements
    display_training_comparison()


if __name__ == "__main__":
    main()
