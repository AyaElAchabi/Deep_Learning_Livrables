"""
Module data - chargement et transformation des données.
"""

from .loaders import (
    DataLoader, 
    ClassificationDataLoader, 
    RegressionDataLoader, 
    ZipDataLoader,
    create_data_loader,
    detect_dataset_type
)

from .transforms import (
    ImageTransforms,
    DataAugmentation, 
    MixupCutmix,
    create_tf_data_pipeline
)

from .utils import (
    create_directory_structure,
    calculate_class_weights,
    generate_data_hash,
    validate_image_files,
    save_dataset_metadata,
    load_dataset_metadata,
    get_dataset_statistics,
    create_sample_dataset,
    create_sample_regression_dataset
)

__all__ = [
    # Loaders
    'DataLoader',
    'ClassificationDataLoader',
    'RegressionDataLoader', 
    'ZipDataLoader',
    'create_data_loader',
    'detect_dataset_type',
    
    # Transforms
    'ImageTransforms',
    'DataAugmentation',
    'MixupCutmix', 
    'create_tf_data_pipeline',
    
    # Utils
    'create_directory_structure',
    'calculate_class_weights',
    'generate_data_hash',
    'validate_image_files',
    'save_dataset_metadata',
    'load_dataset_metadata',
    'get_dataset_statistics',
    'create_sample_dataset',
    'create_sample_regression_dataset'
]
