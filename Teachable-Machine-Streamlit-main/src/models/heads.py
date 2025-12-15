"""
Têtes de modèles et configurations de pertes/métriques.
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod


class ModelHead(ABC):
    """Classe de base pour les têtes de modèles."""
    
    def __init__(self, task_type: str):
        self.task_type = task_type
    
    @abstractmethod
    def get_loss_function(self) -> str:
        """Retourne la fonction de perte appropriée."""
        pass
    
    @abstractmethod 
    def get_metrics(self) -> List[str]:
        """Retourne les métriques d'évaluation."""
        pass
    
    @abstractmethod
    def get_output_activation(self) -> str:
        """Retourne la fonction d'activation de sortie."""
        pass


class ClassificationHead(ModelHead):
    """Tête pour tâches de classification."""
    
    def __init__(self, num_classes: int, label_smoothing: float = 0.0):
        super().__init__("classification")
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
    
    def get_loss_function(self) -> str:
        """Retourne la fonction de perte pour classification."""
        if self.num_classes == 2:
            return "binary_crossentropy"
        else:
            return "sparse_categorical_crossentropy"
    
    def get_metrics(self) -> List[str]:
        """Retourne les métriques pour classification."""
        if self.num_classes == 2:
            return ["accuracy", "precision", "recall"]
        else:
            return ["accuracy", "sparse_categorical_accuracy", "sparse_top_k_categorical_accuracy"]
    
    def get_output_activation(self) -> str:
        """Retourne l'activation de sortie."""
        if self.num_classes == 2:
            return "sigmoid"
        else:
            return "softmax"
    
    def create_loss_with_smoothing(self):
        """Crée une fonction de perte avec label smoothing."""
        if self.label_smoothing <= 0:
            return self.get_loss_function()
        
        try:
            import tensorflow as tf
            
            if self.num_classes == 2:
                return tf.keras.losses.BinaryCrossentropy(label_smoothing=self.label_smoothing)
            else:
                return tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=self.label_smoothing)
        
        except ImportError:
            return self.get_loss_function()


class RegressionHead(ModelHead):
    """Tête pour tâches de régression."""
    
    def __init__(self, loss_type: str = "mse"):
        super().__init__("regression")
        self.loss_type = loss_type
    
    def get_loss_function(self) -> str:
        """Retourne la fonction de perte pour régression."""
        return self.loss_type
    
    def get_metrics(self) -> List[str]:
        """Retourne les métriques pour régression."""
        return ["mae", "mse", "root_mean_squared_error"]
    
    def get_output_activation(self) -> str:
        """Retourne l'activation de sortie."""
        return "linear"


class CustomMetrics:
    """Métriques personnalisées pour différentes tâches."""
    
    @staticmethod
    def f1_score_metric():
        """Métrique F1-Score pour classification."""
        try:
            import tensorflow as tf
            from tensorflow.keras import backend as K
            
            def f1_score(y_true, y_pred):
                def recall(y_true, y_pred):
                    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
                    recall_val = true_positives / (possible_positives + K.epsilon())
                    return recall_val
                
                def precision(y_true, y_pred):
                    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
                    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
                    precision_val = true_positives / (predicted_positives + K.epsilon())
                    return precision_val
                
                precision_val = precision(y_true, y_pred)
                recall_val = recall(y_true, y_pred)
                return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))
            
            return f1_score
            
        except ImportError:
            return None
    
    @staticmethod
    def r2_score_metric():
        """Métrique R² pour régression."""
        try:
            import tensorflow as tf
            from tensorflow.keras import backend as K
            
            def r2_score(y_true, y_pred):
                SS_res = K.sum(K.square(y_true - y_pred))
                SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
                return 1 - SS_res / (SS_tot + K.epsilon())
            
            return r2_score
            
        except ImportError:
            return None
    
    @staticmethod
    def mean_absolute_percentage_error():
        """MAPE pour régression."""
        try:
            import tensorflow as tf
            from tensorflow.keras import backend as K
            
            def mape(y_true, y_pred):
                diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
                return 100. * K.mean(diff)
            
            return mape
            
        except ImportError:
            return None


class LossScheduler:
    """Planificateur de fonctions de perte."""
    
    @staticmethod
    def focal_loss(alpha: float = 0.25, gamma: float = 2.0):
        """Focal Loss pour classification déséquilibrée."""
        try:
            import tensorflow as tf
            from tensorflow.keras import backend as K
            
            def focal_loss_fixed(y_true, y_pred):
                # Clip pour éviter log(0)
                y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
                
                # Calcul de la focal loss
                alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
                p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
                focal_loss_val = -alpha_t * K.pow(1 - p_t, gamma) * K.log(p_t)
                
                return K.mean(focal_loss_val)
            
            return focal_loss_fixed
            
        except ImportError:
            return None
    
    @staticmethod
    def huber_loss(delta: float = 1.0):
        """Huber Loss pour régression robuste."""
        try:
            import tensorflow as tf
            from tensorflow.keras import backend as K
            
            def huber_loss_fixed(y_true, y_pred):
                error = y_true - y_pred
                condition = K.abs(error) <= delta
                squared_loss = 0.5 * K.square(error)
                linear_loss = delta * K.abs(error) - 0.5 * K.square(delta)
                return K.mean(tf.where(condition, squared_loss, linear_loss))
            
            return huber_loss_fixed
            
        except ImportError:
            return None


def create_model_head(task_type: str, **kwargs) -> ModelHead:
    """Factory pour créer une tête de modèle."""
    if task_type == "classification":
        num_classes = kwargs.get('num_classes', 2)
        label_smoothing = kwargs.get('label_smoothing', 0.0)
        return ClassificationHead(num_classes, label_smoothing)
    
    elif task_type == "regression":
        loss_type = kwargs.get('loss_type', 'mse')
        return RegressionHead(loss_type)
    
    else:
        raise ValueError(f"Type de tâche non supporté : {task_type}")


def get_optimizer_config(optimizer_name: str, learning_rate: float, 
                        **kwargs) -> Dict[str, Any]:
    """Configuration des optimiseurs."""
    configs = {
        'adam': {
            'class_name': 'Adam',
            'config': {
                'learning_rate': learning_rate,
                'beta_1': kwargs.get('beta_1', 0.9),
                'beta_2': kwargs.get('beta_2', 0.999),
                'epsilon': kwargs.get('epsilon', 1e-7)
            }
        },
        'sgd': {
            'class_name': 'SGD',
            'config': {
                'learning_rate': learning_rate,
                'momentum': kwargs.get('momentum', 0.9),
                'nesterov': kwargs.get('nesterov', True)
            }
        },
        'rmsprop': {
            'class_name': 'RMSprop',
            'config': {
                'learning_rate': learning_rate,
                'rho': kwargs.get('rho', 0.9),
                'momentum': kwargs.get('momentum', 0.0),
                'epsilon': kwargs.get('epsilon', 1e-7)
            }
        },
        'adamw': {
            'class_name': 'AdamW',
            'config': {
                'learning_rate': learning_rate,
                'weight_decay': kwargs.get('weight_decay', 0.01),
                'beta_1': kwargs.get('beta_1', 0.9),
                'beta_2': kwargs.get('beta_2', 0.999),
                'epsilon': kwargs.get('epsilon', 1e-7)
            }
        }
    }
    
    if optimizer_name.lower() not in configs:
        raise ValueError(f"Optimiseur non supporté : {optimizer_name}")
    
    return configs[optimizer_name.lower()]


def compile_model(model: 'tf.keras.Model', task_type: str, num_classes: int = None,
                 optimizer: str = "adam", learning_rate: float = 0.001,
                 loss_config: Dict[str, Any] = None, 
                 custom_metrics: List[str] = None) -> 'tf.keras.Model':
    """Compile un modèle avec la configuration appropriée."""
    try:
        import tensorflow as tf
        
        # Créer la tête appropriée
        head = create_model_head(task_type, num_classes=num_classes)
        
        # Configuration de l'optimiseur
        opt_config = get_optimizer_config(optimizer, learning_rate)
        optimizer_obj = tf.keras.optimizers.get(opt_config)
        
        # Configuration de la perte
        if loss_config:
            loss = loss_config.get('loss', head.get_loss_function())
        else:
            loss = head.get_loss_function()
        
        # Métriques
        metrics = head.get_metrics()
        if custom_metrics:
            metrics.extend(custom_metrics)
        
        # Ajouter métriques personnalisées si disponibles
        if task_type == "classification":
            f1_metric = CustomMetrics.f1_score_metric()
            if f1_metric:
                metrics.append(f1_metric)
        
        elif task_type == "regression":
            r2_metric = CustomMetrics.r2_score_metric()
            if r2_metric:
                metrics.append(r2_metric)
        
        # Compiler le modèle
        model.compile(
            optimizer=optimizer_obj,
            loss=loss,
            metrics=metrics
        )
        
        return model
        
    except ImportError:
        raise ImportError("TensorFlow requis pour compiler le modèle")


def get_class_weights(class_distribution: Dict[str, int]) -> Dict[int, float]:
    """Calcule les poids de classes pour gérer le déséquilibre."""
    total_samples = sum(class_distribution.values())
    num_classes = len(class_distribution)
    
    class_weights = {}
    for i, (class_name, count) in enumerate(class_distribution.items()):
        weight = total_samples / (num_classes * count)
        class_weights[i] = weight
    
    return class_weights
