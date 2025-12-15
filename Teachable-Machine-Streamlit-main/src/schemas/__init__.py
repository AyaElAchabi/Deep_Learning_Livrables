"""
Module schemas - Types et validation des données.
"""

from .dataclasses import (
    Config,
    DataConfig,
    AugmentationConfig,
    TrainingConfig,
    ModelConfig,
    TaskConfig,
    DatasetInfo,
    TrainingState,
    ModelInfo,
    PredictionResult,
    ExperimentPreset,
    SessionState
)

__all__ = [
    'Config',
    'DataConfig', 
    'AugmentationConfig',
    'TrainingConfig',
    'ModelConfig',
    'TaskConfig',
    'DatasetInfo',
    'TrainingState',
    'ModelInfo',
    'PredictionResult',
    'ExperimentPreset',
    'SessionState'
]
