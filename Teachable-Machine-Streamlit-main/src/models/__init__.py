"""
Module models - architectures et configuration des modèles.
"""

from .registry import (
    ModelRegistry,
    ModelBuilder,
    MobileNetV3Builder,
    EfficientNetBuilder,
    ResNetBuilder,
    get_model_summary,
    transfer_learning_recommendations,
    model_registry
)

from .heads import (
    ModelHead,
    ClassificationHead,
    RegressionHead,
    CustomMetrics,
    LossScheduler,
    create_model_head,
    get_optimizer_config,
    compile_model,
    get_class_weights
)

__all__ = [
    # Registry
    'ModelRegistry',
    'ModelBuilder',
    'MobileNetV3Builder',
    'EfficientNetBuilder', 
    'ResNetBuilder',
    'get_model_summary',
    'transfer_learning_recommendations',
    'model_registry',
    
    # Heads
    'ModelHead',
    'ClassificationHead',
    'RegressionHead',
    'CustomMetrics',
    'LossScheduler',
    'create_model_head',
    'get_optimizer_config',
    'compile_model',
    'get_class_weights'
]
