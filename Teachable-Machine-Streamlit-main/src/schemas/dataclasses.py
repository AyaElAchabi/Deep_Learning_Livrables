"""
Schémas de données et classes de configuration avec Pydantic.
"""

from typing import List, Optional, Union, Dict, Any, Literal
from pathlib import Path
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
import datetime


class DataConfig(BaseModel):
    """Configuration des données."""
    image_size: List[int] = Field([224, 224], description="Taille des images [height, width]")
    channels: int = Field(3, description="Nombre de canaux")
    batch_size: int = Field(32, description="Taille du batch")
    validation_split: float = Field(0.2, ge=0.0, le=1.0, description="Proportion de validation")
    test_split: float = Field(0.1, ge=0.0, le=1.0, description="Proportion de test")
    shuffle: bool = Field(True, description="Mélanger les données")
    normalization: Literal["imagenet", "0-1"] = Field("imagenet", description="Type de normalisation")
    
    @validator('validation_split', 'test_split')
    def validate_splits(cls, v, values):
        if 'validation_split' in values:
            total = values['validation_split'] + v
            if total >= 1.0:
                raise ValueError("La somme validation_split + test_split doit être < 1.0")
        return v


class AugmentationConfig(BaseModel):
    """Configuration de l'augmentation de données."""
    enabled: bool = Field(True, description="Activer l'augmentation")
    horizontal_flip: bool = Field(True, description="Flip horizontal")
    vertical_flip: bool = Field(False, description="Flip vertical")
    rotation_range: float = Field(15.0, ge=0.0, le=180.0, description="Rotation en degrés")
    width_shift_range: float = Field(0.1, ge=0.0, le=1.0, description="Décalage largeur")
    height_shift_range: float = Field(0.1, ge=0.0, le=1.0, description="Décalage hauteur")
    zoom_range: float = Field(0.1, ge=0.0, le=1.0, description="Zoom")
    brightness_range: List[float] = Field([0.9, 1.1], description="Plage de luminosité")
    cutout_prob: float = Field(0.0, ge=0.0, le=1.0, description="Probabilité cutout")
    mixup_alpha: float = Field(0.0, ge=0.0, description="Alpha pour mixup")
    cutmix_alpha: float = Field(0.0, ge=0.0, description="Alpha pour cutmix")


class TrainingConfig(BaseModel):
    """Configuration de l'entraînement."""
    epochs: int = Field(50, gt=0, description="Nombre d'epochs")
    learning_rate: float = Field(0.001, gt=0.0, description="Taux d'apprentissage")
    optimizer: Literal["adam", "sgd", "rmsprop"] = Field("adam", description="Optimiseur")
    
    class EarlyStoppingConfig(BaseModel):
        patience: int = Field(10, gt=0)
        monitor: str = Field("val_loss")
        restore_best_weights: bool = Field(True)
    
    class ReduceLRConfig(BaseModel):
        factor: float = Field(0.5, gt=0.0, lt=1.0)
        patience: int = Field(5, gt=0)
        min_lr: float = Field(1e-7, gt=0.0)
    
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    reduce_lr: ReduceLRConfig = Field(default_factory=ReduceLRConfig)


class ModelConfig(BaseModel):
    """Configuration du modèle."""
    backbone: Literal[
        "MobileNetV3Small", 
        "MobileNetV3Large", 
        "EfficientNetB0", 
        "ResNet50"
    ] = Field("MobileNetV3Small", description="Architecture backbone")
    pretrained: bool = Field(True, description="Utiliser des poids pré-entraînés")
    trainable_layers: int = Field(20, ge=-1, description="Nombre de couches entraînables (-1 pour toutes)")
    dropout: float = Field(0.2, ge=0.0, le=1.0, description="Taux de dropout")


class TaskConfig(BaseModel):
    """Configuration de la tâche."""
    task_type: Literal["classification", "regression"] = Field("classification")
    num_classes: Optional[int] = Field(None, gt=0, description="Nombre de classes (classification)")
    class_names: Optional[List[str]] = Field(None, description="Noms des classes")
    
    # Configuration classification
    classification_loss: str = Field("sparse_categorical_crossentropy")
    classification_metrics: List[str] = Field(["accuracy"])
    
    # Configuration régression
    regression_loss: str = Field("mse")
    regression_metrics: List[str] = Field(["mae", "mse"])


class AppConfig(BaseModel):
    """Configuration générale de l'application."""
    title: str = Field("Teachable Machine Streamlit")
    debug: bool = Field(False)
    seed: int = Field(42, ge=0)
    language: Literal["fr", "en"] = Field("fr")


class PathsConfig(BaseModel):
    """Configuration des chemins."""
    artifacts: str = Field("artifacts")
    logs: str = Field("logs")
    cache: str = Field(".cache")


class DeviceConfig(BaseModel):
    """Configuration du device."""
    use_gpu: bool = Field(True)
    memory_growth: bool = Field(True)


class Config(BaseModel):
    """Configuration complète de l'application."""
    app: AppConfig = Field(default_factory=AppConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    device: DeviceConfig = Field(default_factory=DeviceConfig)


@dataclass
class DatasetInfo:
    """Informations sur un dataset."""
    name: str
    path: Path
    task_type: str
    num_samples: int
    num_classes: Optional[int] = None
    class_names: Optional[List[str]] = None
    class_distribution: Optional[Dict[str, int]] = None
    target_range: Optional[tuple] = None  # (min, max) pour régression
    splits: Optional[Dict[str, int]] = None  # train/val/test counts
    image_shape: Optional[tuple] = None
    created_at: Optional[datetime.datetime] = None


@dataclass
class TrainingState:
    """État de l'entraînement."""
    is_training: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    current_metrics: Dict[str, float] = None
    best_metrics: Dict[str, float] = None
    training_history: Dict[str, List[float]] = None
    start_time: Optional[datetime.datetime] = None
    estimated_time_remaining: Optional[float] = None


@dataclass
class ModelInfo:
    """Informations sur un modèle."""
    name: str
    path: Path
    task_type: str
    backbone: str
    num_classes: Optional[int]
    class_names: Optional[List[str]]
    input_shape: tuple
    total_params: int
    trainable_params: int
    config: Dict[str, Any]
    metrics: Dict[str, float]
    created_at: datetime.datetime
    training_time: float  # en secondes


@dataclass
class PredictionResult:
    """Résultat de prédiction."""
    image_path: Optional[str]
    predictions: Union[List[float], float]  # probas pour classification, valeur pour régression
    predicted_class: Optional[str] = None  # pour classification
    confidence: Optional[float] = None  # pour classification
    processing_time: float = 0.0


class ExperimentPreset(BaseModel):
    """Preset d'expérience."""
    name: str
    description: str
    config_overrides: Dict[str, Any]
    
    @classmethod
    def get_default_presets(cls) -> List['ExperimentPreset']:
        """Retourne les presets par défaut."""
        return [
            cls(
                name="Rapide",
                description="Entraînement rapide pour tests (10 epochs)",
                config_overrides={
                    "training.epochs": 10,
                    "training.learning_rate": 0.01,
                    "model.trainable_layers": 5
                }
            ),
            cls(
                name="Équilibré",
                description="Configuration équilibrée (30 epochs)",
                config_overrides={
                    "training.epochs": 30,
                    "training.learning_rate": 0.001,
                    "model.trainable_layers": 20
                }
            ),
            cls(
                name="Précis",
                description="Entraînement précis et long (100 epochs)",
                config_overrides={
                    "training.epochs": 100,
                    "training.learning_rate": 0.0005,
                    "model.trainable_layers": -1,
                    "training.early_stopping.patience": 20
                }
            )
        ]


class SessionState(BaseModel):
    """État de la session Streamlit."""
    dataset_info: Optional[Dict[str, Any]] = None
    training_state: Optional[Dict[str, Any]] = None
    selected_model_path: Optional[str] = None
    current_config: Optional[Dict[str, Any]] = None
    last_run_dir: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
