"""
Gestion de la configuration centralisée.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st
from ..schemas.dataclasses import Config


class ConfigManager:
    """Gestionnaire de configuration centralisé."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config = None
        self._config_cache = {}
    
    def load_config(self) -> Config:
        """Charge la configuration depuis le fichier YAML."""
        if self._config is None:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_dict = yaml.safe_load(f)
                self._config = Config(**config_dict)
            else:
                # Configuration par défaut si le fichier n'existe pas
                self._config = Config()
        return self._config
    
    def save_config(self, config: Config) -> None:
        """Sauvegarde la configuration dans le fichier YAML."""
        config_dict = config.dict()
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        self._config = config
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Retourne la configuration sous forme de dictionnaire."""
        config = self.load_config()
        return config.dict()
    
    def update_config(self, updates: Dict[str, Any]) -> Config:
        """Met à jour la configuration avec un dictionnaire de modifications."""
        config = self.load_config()
        config_dict = config.dict()
        
        # Applique les mises à jour récursivement
        self._deep_update(config_dict, updates)
        
        # Recrée l'objet Config avec validation
        updated_config = Config(**config_dict)
        self._config = updated_config
        return updated_config
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """Met à jour récursivement un dictionnaire."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def apply_preset(self, preset_name: str) -> Config:
        """Applique un preset de configuration."""
        from ..schemas.dataclasses import ExperimentPreset
        
        presets = ExperimentPreset.get_default_presets()
        preset = next((p for p in presets if p.name == preset_name), None)
        
        if preset is None:
            raise ValueError(f"Preset '{preset_name}' non trouvé")
        
        return self.update_config(preset.config_overrides)
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Retourne la configuration adaptée pour Streamlit session_state."""
        config = self.load_config()
        return config.dict()
    
    def load_from_streamlit(self) -> Config:
        """Charge la configuration depuis Streamlit session_state."""
        if 'config' in st.session_state:
            config_dict = st.session_state.config
            return Config(**config_dict)
        return self.load_config()
    
    def save_to_streamlit(self, config: Config) -> None:
        """Sauvegarde la configuration dans Streamlit session_state."""
        st.session_state.config = config.dict()


def setup_tensorflow_config(config: Config) -> None:
    """Configure TensorFlow selon la configuration."""
    try:
        import tensorflow as tf
        
        # Configuration GPU
        if config.device.use_gpu:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    # Croissance de mémoire progressive
                    if config.device.memory_growth:
                        for gpu in gpus:
                            tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"GPU configuré : {len(gpus)} device(s) trouvé(s)")
                except RuntimeError as e:
                    print(f"Erreur configuration GPU : {e}")
            else:
                print("Aucun GPU trouvé, utilisation du CPU")
        else:
            # Force l'utilisation du CPU
            tf.config.set_visible_devices([], 'GPU')
            print("Configuration forcée sur CPU")
        
        # Seed pour reproductibilité
        tf.random.set_seed(config.app.seed)
        
    except ImportError:
        print("TensorFlow non disponible")


def setup_global_seed(seed: int) -> None:
    """Configure les seeds globaux pour la reproductibilité."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def validate_config(config_dict: Dict[str, Any]) -> tuple[bool, list[str]]:
    """Valide une configuration et retourne les erreurs."""
    errors = []
    
    try:
        Config(**config_dict)
        return True, []
    except Exception as e:
        errors.append(str(e))
        return False, errors


def get_config_schema() -> Dict[str, Any]:
    """Retourne le schéma de configuration pour l'éditeur UI."""
    return {
        "app": {
            "title": {"type": "string", "description": "Titre de l'application"},
            "debug": {"type": "boolean", "description": "Mode debug"},
            "seed": {"type": "integer", "description": "Seed pour reproductibilité"},
            "language": {"type": "string", "enum": ["fr", "en"], "description": "Langue"}
        },
        "data": {
            "image_size": {"type": "array", "items": {"type": "integer"}, "description": "Taille des images [H, W]"},
            "channels": {"type": "integer", "description": "Nombre de canaux"},
            "batch_size": {"type": "integer", "description": "Taille du batch"},
            "validation_split": {"type": "number", "minimum": 0, "maximum": 1, "description": "Split validation"},
            "test_split": {"type": "number", "minimum": 0, "maximum": 1, "description": "Split test"},
            "normalization": {"type": "string", "enum": ["imagenet", "0-1"], "description": "Normalisation"}
        },
        "training": {
            "epochs": {"type": "integer", "minimum": 1, "description": "Nombre d'epochs"},
            "learning_rate": {"type": "number", "minimum": 0, "description": "Taux d'apprentissage"},
            "optimizer": {"type": "string", "enum": ["adam", "sgd", "rmsprop"], "description": "Optimiseur"}
        },
        "model": {
            "backbone": {
                "type": "string", 
                "enum": ["MobileNetV3Small", "MobileNetV3Large", "EfficientNetB0", "ResNet50"],
                "description": "Architecture"
            },
            "pretrained": {"type": "boolean", "description": "Poids pré-entraînés"},
            "trainable_layers": {"type": "integer", "description": "Couches entraînables"},
            "dropout": {"type": "number", "minimum": 0, "maximum": 1, "description": "Dropout"}
        }
    }


# Instance globale du gestionnaire de configuration
config_manager = ConfigManager()
