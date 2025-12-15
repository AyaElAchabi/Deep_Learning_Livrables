"""
Registre des modèles et factory pour la création de modèles.
"""

from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod


class ModelRegistry:
    """Registre des architectures de modèles disponibles."""
    
    def __init__(self):
        self._models = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Enregistre les modèles par défaut."""
        self._models.update({
            'MobileNetV3Small': {
                'class': MobileNetV3Builder,
                'variant': 'small',
                'input_shape': (224, 224, 3),
                'description': 'MobileNetV3 Small - Léger et efficace pour mobile',
                'params_approx': '2.5M'
            },
            'MobileNetV3Large': {
                'class': MobileNetV3Builder,
                'variant': 'large', 
                'input_shape': (224, 224, 3),
                'description': 'MobileNetV3 Large - Plus précis que Small',
                'params_approx': '5.4M'
            },
            'EfficientNetB0': {
                'class': EfficientNetBuilder,
                'variant': 'B0',
                'input_shape': (224, 224, 3),
                'description': 'EfficientNet B0 - Excellent rapport précision/efficacité',
                'params_approx': '5.3M'
            },
            'ResNet50': {
                'class': ResNetBuilder,
                'variant': '50',
                'input_shape': (224, 224, 3),
                'description': 'ResNet50 - Architecture classique et robuste',
                'params_approx': '25.6M'
            }
        })
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Retourne la liste des modèles disponibles."""
        return self._models.copy()
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Retourne les informations d'un modèle."""
        if model_name not in self._models:
            raise ValueError(f"Modèle '{model_name}' non trouvé")
        return self._models[model_name].copy()
    
    def create_model(self, model_name: str, task_type: str, num_classes: int = None,
                    pretrained: bool = True, trainable_layers: int = 20,
                    dropout: float = 0.2, input_shape: Tuple[int, int, int] = None) -> 'tf.keras.Model':
        """Crée une instance de modèle."""
        if model_name not in self._models:
            raise ValueError(f"Modèle '{model_name}' non trouvé")
        
        model_info = self._models[model_name]
        builder_class = model_info['class']
        
        if input_shape is None:
            input_shape = model_info['input_shape']
        
        builder = builder_class(
            variant=model_info['variant'],
            input_shape=input_shape,
            pretrained=pretrained
        )
        
        return builder.build(
            task_type=task_type,
            num_classes=num_classes,
            trainable_layers=trainable_layers,
            dropout=dropout
        )


class ModelBuilder(ABC):
    """Classe de base pour les constructeurs de modèles."""
    
    def __init__(self, variant: str, input_shape: Tuple[int, int, int], 
                 pretrained: bool = True):
        self.variant = variant
        self.input_shape = input_shape
        self.pretrained = pretrained
    
    @abstractmethod
    def _create_backbone(self) -> 'tf.keras.Model':
        """Crée le backbone du modèle."""
        pass
    
    @abstractmethod
    def _get_feature_layer_name(self) -> str:
        """Retourne le nom de la couche de features."""
        pass
    
    def build(self, task_type: str, num_classes: int = None,
              trainable_layers: int = 20, dropout: float = 0.2) -> 'tf.keras.Model':
        """Construit le modèle complet."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, Model
            
            # Créer le backbone
            backbone = self._create_backbone()
            
            # Configurer les couches entraînables
            self._set_trainable_layers(backbone, trainable_layers)
            
            # Ajouter la tête de classification/régression
            x = backbone.output
            
            # Global Average Pooling si nécessaire
            if len(x.shape) > 2:
                x = layers.GlobalAveragePooling2D()(x)
            
            # Dropout
            if dropout > 0:
                x = layers.Dropout(dropout)(x)
            
            # Couche de sortie
            if task_type == "classification":
                if num_classes is None:
                    raise ValueError("num_classes requis pour la classification")
                
                if num_classes == 2:
                    # Classification binaire
                    predictions = layers.Dense(1, activation='sigmoid', name='predictions')(x)
                else:
                    # Classification multi-classe
                    predictions = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
            
            elif task_type == "regression":
                predictions = layers.Dense(1, activation='linear', name='predictions')(x)
            
            else:
                raise ValueError(f"Type de tâche non supporté : {task_type}")
            
            # Créer le modèle final
            model = Model(inputs=backbone.input, outputs=predictions)
            
            return model
            
        except ImportError:
            raise ImportError("TensorFlow requis pour créer les modèles")
    
    def _set_trainable_layers(self, model: 'tf.keras.Model', trainable_layers: int):
        """Configure les couches entraînables."""
        if trainable_layers == -1:
            # Toutes les couches entraînables
            model.trainable = True
        elif trainable_layers == 0:
            # Aucune couche entraînable (feature extraction)
            model.trainable = False
        else:
            # N dernières couches entraînables
            model.trainable = True
            total_layers = len(model.layers)
            freeze_until = max(0, total_layers - trainable_layers)
            
            for i, layer in enumerate(model.layers):
                layer.trainable = i >= freeze_until


class MobileNetV3Builder(ModelBuilder):
    """Constructeur pour MobileNetV3."""
    
    def _create_backbone(self) -> 'tf.keras.Model':
        try:
            import tensorflow as tf
            
            if self.variant.lower() == 'small':
                backbone = tf.keras.applications.MobileNetV3Small(
                    input_shape=self.input_shape,
                    include_top=False,
                    weights='imagenet' if self.pretrained else None
                )
            else:  # large
                backbone = tf.keras.applications.MobileNetV3Large(
                    input_shape=self.input_shape,
                    include_top=False,
                    weights='imagenet' if self.pretrained else None
                )
            
            return backbone
            
        except ImportError:
            raise ImportError("TensorFlow requis pour MobileNetV3")
    
    def _get_feature_layer_name(self) -> str:
        return "global_average_pooling2d"


class EfficientNetBuilder(ModelBuilder):
    """Constructeur pour EfficientNet."""
    
    def _create_backbone(self) -> 'tf.keras.Model':
        try:
            import tensorflow as tf
            
            if self.variant == 'B0':
                backbone = tf.keras.applications.EfficientNetB0(
                    input_shape=self.input_shape,
                    include_top=False,
                    weights='imagenet' if self.pretrained else None
                )
            elif self.variant == 'B1':
                backbone = tf.keras.applications.EfficientNetB1(
                    input_shape=self.input_shape,
                    include_top=False,
                    weights='imagenet' if self.pretrained else None
                )
            else:
                raise ValueError(f"Variante EfficientNet non supportée : {self.variant}")
            
            return backbone
            
        except ImportError:
            raise ImportError("TensorFlow requis pour EfficientNet")
    
    def _get_feature_layer_name(self) -> str:
        return "global_average_pooling2d"


class ResNetBuilder(ModelBuilder):
    """Constructeur pour ResNet."""
    
    def _create_backbone(self) -> 'tf.keras.Model':
        try:
            import tensorflow as tf
            
            if self.variant == '50':
                backbone = tf.keras.applications.ResNet50(
                    input_shape=self.input_shape,
                    include_top=False,
                    weights='imagenet' if self.pretrained else None
                )
            elif self.variant == '101':
                backbone = tf.keras.applications.ResNet101(
                    input_shape=self.input_shape,
                    include_top=False,
                    weights='imagenet' if self.pretrained else None
                )
            else:
                raise ValueError(f"Variante ResNet non supportée : {self.variant}")
            
            return backbone
            
        except ImportError:
            raise ImportError("TensorFlow requis pour ResNet")
    
    def _get_feature_layer_name(self) -> str:
        return "global_average_pooling2d"


def get_model_summary(model: 'tf.keras.Model') -> Dict[str, Any]:
    """Retourne un résumé du modèle."""
    try:
        import io
        import sys
        
        # Capturer le résumé
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        model.summary()
        
        sys.stdout = old_stdout
        summary_str = buffer.getvalue()
        
        # Calculer les statistiques
        total_params = model.count_params()
        trainable_params = sum([layer.count_params() for layer in model.layers if layer.trainable])
        non_trainable_params = total_params - trainable_params
        
        return {
            'summary_text': summary_str,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Approximation 4 bytes par param
            'layers_count': len(model.layers)
        }
        
    except Exception as e:
        return {'error': str(e)}


def transfer_learning_recommendations(task_type: str, dataset_size: int, 
                                    time_budget: str = "medium") -> Dict[str, Any]:
    """Recommandations pour le transfer learning."""
    recommendations = {
        'backbone': 'MobileNetV3Small',
        'trainable_layers': 20,
        'learning_rate': 0.001,
        'reasoning': []
    }
    
    # Choix du backbone selon la taille du dataset et le budget temps
    if dataset_size < 1000:
        if time_budget == "fast":
            recommendations['backbone'] = 'MobileNetV3Small'
            recommendations['trainable_layers'] = 5
        else:
            recommendations['backbone'] = 'MobileNetV3Large'
            recommendations['trainable_layers'] = 10
        recommendations['reasoning'].append("Dataset petit : modèle léger pour éviter le surapprentissage")
    
    elif dataset_size < 10000:
        if time_budget == "fast":
            recommendations['backbone'] = 'MobileNetV3Large'
            recommendations['trainable_layers'] = 20
        else:
            recommendations['backbone'] = 'EfficientNetB0'
            recommendations['trainable_layers'] = 30
        recommendations['reasoning'].append("Dataset moyen : équilibre entre performance et efficacité")
    
    else:
        if time_budget == "fast":
            recommendations['backbone'] = 'EfficientNetB0'
            recommendations['trainable_layers'] = 50
        else:
            recommendations['backbone'] = 'ResNet50'
            recommendations['trainable_layers'] = -1  # Tout entraînable
        recommendations['reasoning'].append("Grand dataset : modèle plus complexe possible")
    
    # Ajustement du learning rate
    if recommendations['trainable_layers'] == -1 or recommendations['trainable_layers'] > 50:
        recommendations['learning_rate'] = 0.0001
        recommendations['reasoning'].append("Fine-tuning complet : LR plus faible")
    elif recommendations['trainable_layers'] < 10:
        recommendations['learning_rate'] = 0.01
        recommendations['reasoning'].append("Feature extraction : LR plus élevé")
    
    return recommendations


# Instance globale du registre
model_registry = ModelRegistry()
