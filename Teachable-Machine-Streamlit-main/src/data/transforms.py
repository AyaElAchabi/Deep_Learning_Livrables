"""
Transformations et augmentations pour les images.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import random


class ImageTransforms:
    """Classe de base pour les transformations d'images."""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224), 
                 normalization: str = "imagenet"):
        self.image_size = image_size
        self.normalization = normalization
        
        # Paramètres de normalisation
        if normalization == "imagenet":
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:  # "0-1"
            self.mean = [0.0, 0.0, 0.0]
            self.std = [1.0, 1.0, 1.0]
    
    def load_and_preprocess_image(self, image_path: Union[str, Path]) -> 'tf.Tensor':
        """Charge et préprocesse une image."""
        try:
            import tensorflow as tf
            
            # Charger l'image
            image = tf.io.read_file(str(image_path))
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.cast(image, tf.float32)
            
            # Redimensionner
            image = tf.image.resize(image, self.image_size)
            
            # Normaliser
            if self.normalization == "imagenet":
                image = tf.cast(image, tf.float32) / 255.0
                image = (image - self.mean) / self.std
            else:  # "0-1"
                image = tf.cast(image, tf.float32) / 255.0
            
            return image
            
        except ImportError:
            raise ImportError("TensorFlow requis pour le préprocessing d'images")
    
    def preprocess_batch(self, image_paths: List[Union[str, Path]]) -> 'tf.Tensor':
        """Préprocesse un batch d'images."""
        try:
            import tensorflow as tf
            
            images = []
            for path in image_paths:
                img = self.load_and_preprocess_image(path)
                images.append(img)
            
            return tf.stack(images)
            
        except ImportError:
            raise ImportError("TensorFlow requis pour le préprocessing")


class DataAugmentation:
    """Classe pour l'augmentation de données."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('enabled', True)
    
    def get_augmentation_layer(self) -> Optional['tf.keras.Sequential']:
        """Retourne une couche d'augmentation TensorFlow/Keras."""
        if not self.enabled:
            return None
        
        try:
            import tensorflow as tf
            from tensorflow.keras import layers
            
            augmentation_layers = []
            
            # Flips
            if self.config.get('horizontal_flip', False):
                augmentation_layers.append(
                    layers.RandomFlip(mode="horizontal")
                )
            
            if self.config.get('vertical_flip', False):
                augmentation_layers.append(
                    layers.RandomFlip(mode="vertical")
                )
            
            # Rotation
            rotation_range = self.config.get('rotation_range', 0.0)
            if rotation_range > 0:
                augmentation_layers.append(
                    layers.RandomRotation(factor=rotation_range / 180.0)
                )
            
            # Translations
            width_shift = self.config.get('width_shift_range', 0.0)
            height_shift = self.config.get('height_shift_range', 0.0)
            if width_shift > 0 or height_shift > 0:
                augmentation_layers.append(
                    layers.RandomTranslation(
                        height_factor=height_shift,
                        width_factor=width_shift
                    )
                )
            
            # Zoom
            zoom_range = self.config.get('zoom_range', 0.0)
            if zoom_range > 0:
                augmentation_layers.append(
                    layers.RandomZoom(height_factor=zoom_range)
                )
            
            # Brightness
            brightness_range = self.config.get('brightness_range', [1.0, 1.0])
            if brightness_range[0] != 1.0 or brightness_range[1] != 1.0:
                delta = max(abs(1.0 - brightness_range[0]), abs(brightness_range[1] - 1.0))
                augmentation_layers.append(
                    layers.RandomBrightness(factor=delta)
                )
            
            if augmentation_layers:
                return tf.keras.Sequential(augmentation_layers, name="data_augmentation")
            else:
                return None
                
        except ImportError:
            raise ImportError("TensorFlow requis pour l'augmentation")
    
    def get_advanced_augmentation(self) -> Optional[callable]:
        """Retourne une fonction d'augmentation avancée avec Albumentations."""
        if not self.enabled:
            return None
        
        try:
            import albumentations as A
            import cv2
            
            transforms = []
            
            # Transformations géométriques
            if self.config.get('horizontal_flip', False):
                transforms.append(A.HorizontalFlip(p=0.5))
            
            if self.config.get('vertical_flip', False):
                transforms.append(A.VerticalFlip(p=0.5))
            
            rotation_range = self.config.get('rotation_range', 0.0)
            if rotation_range > 0:
                transforms.append(A.Rotate(limit=rotation_range, p=0.5))
            
            # Transformations de couleur
            brightness_range = self.config.get('brightness_range', [1.0, 1.0])
            if brightness_range[0] != 1.0 or brightness_range[1] != 1.0:
                transforms.append(A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ))
            
            # Cutout
            cutout_prob = self.config.get('cutout_prob', 0.0)
            if cutout_prob > 0:
                transforms.append(A.CoarseDropout(
                    max_holes=8, max_height=8, max_width=8, 
                    min_holes=1, fill_value=0, p=cutout_prob
                ))
            
            if transforms:
                augmentation = A.Compose(transforms)
                
                def augment_function(image):
                    """Fonction d'augmentation compatible TensorFlow."""
                    if isinstance(image, str) or isinstance(image, Path):
                        # Charger l'image avec OpenCV
                        img = cv2.imread(str(image))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        # Convertir depuis tensor TensorFlow
                        img = image.numpy()
                        img = (img * 255).astype(np.uint8)
                    
                    # Appliquer l'augmentation
                    augmented = augmentation(image=img)
                    img_aug = augmented['image']
                    
                    # Reconvertir en float32 normalisé
                    return img_aug.astype(np.float32) / 255.0
                
                return augment_function
            
            return None
            
        except ImportError:
            print("Albumentations non disponible, utilisation des augmentations TensorFlow")
            return None


class MixupCutmix:
    """Implémentation de Mixup et CutMix."""
    
    def __init__(self, mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
    
    def mixup(self, x: 'tf.Tensor', y: 'tf.Tensor') -> Tuple['tf.Tensor', 'tf.Tensor']:
        """Applique Mixup sur un batch."""
        try:
            import tensorflow as tf
            
            if self.mixup_alpha <= 0:
                return x, y
            
            batch_size = tf.shape(x)[0]
            
            # Générer lambda depuis distribution Beta
            lam = tf.random.gamma([batch_size], alpha=self.mixup_alpha, beta=self.mixup_alpha)
            lam = tf.reshape(lam, [batch_size, 1, 1, 1])
            
            # Mélanger l'ordre des échantillons
            indices = tf.random.shuffle(tf.range(batch_size))
            x_shuffled = tf.gather(x, indices)
            y_shuffled = tf.gather(y, indices)
            
            # Mixup
            x_mixed = lam * x + (1 - lam) * x_shuffled
            
            # Pour les labels, on garde les deux labels avec leurs poids
            lam_labels = tf.reshape(lam[:, 0, 0, 0], [batch_size, 1])
            y_mixed = lam_labels * y + (1 - lam_labels) * y_shuffled
            
            return x_mixed, y_mixed
            
        except ImportError:
            raise ImportError("TensorFlow requis pour Mixup")
    
    def cutmix(self, x: 'tf.Tensor', y: 'tf.Tensor') -> Tuple['tf.Tensor', 'tf.Tensor']:
        """Applique CutMix sur un batch."""
        try:
            import tensorflow as tf
            
            if self.cutmix_alpha <= 0:
                return x, y
            
            batch_size = tf.shape(x)[0]
            image_size = tf.shape(x)[1]
            
            # Générer lambda
            lam = tf.random.gamma([batch_size], alpha=self.cutmix_alpha, beta=self.cutmix_alpha)
            
            # Calculer la taille des patches
            cut_ratio = tf.sqrt(1.0 - lam)
            cut_w = tf.cast(cut_ratio * tf.cast(image_size, tf.float32), tf.int32)
            cut_h = cut_w
            
            # Positions aléatoires
            cx = tf.random.uniform([batch_size], 0, image_size, dtype=tf.int32)
            cy = tf.random.uniform([batch_size], 0, image_size, dtype=tf.int32)
            
            # Calculer les boîtes
            x1 = tf.clip_by_value(cx - cut_w // 2, 0, image_size)
            y1 = tf.clip_by_value(cy - cut_h // 2, 0, image_size)
            x2 = tf.clip_by_value(cx + cut_w // 2, 0, image_size)
            y2 = tf.clip_by_value(cy + cut_h // 2, 0, image_size)
            
            # Mélanger l'ordre
            indices = tf.random.shuffle(tf.range(batch_size))
            x_shuffled = tf.gather(x, indices)
            y_shuffled = tf.gather(y, indices)
            
            # Appliquer CutMix (version simplifiée)
            # Note : implémentation complète nécessiterait des opérations plus complexes
            
            # Pour simplifier, on utilise une approximation de lambda
            actual_lam = 1.0 - tf.cast((x2 - x1) * (y2 - y1), tf.float32) / tf.cast(image_size * image_size, tf.float32)
            actual_lam = tf.reshape(actual_lam, [batch_size, 1])
            
            y_mixed = actual_lam * y + (1 - actual_lam) * y_shuffled
            
            # Pour les images, implémentation simplifiée
            x_mixed = x  # TODO: implémenter le masquage réel
            
            return x_mixed, y_mixed
            
        except ImportError:
            raise ImportError("TensorFlow requis pour CutMix")


def create_tf_data_pipeline(image_paths: List[str], labels: Optional[List] = None,
                           batch_size: int = 32, shuffle: bool = True,
                           augmentation: Optional[DataAugmentation] = None,
                           transforms: Optional[ImageTransforms] = None,
                           task_type: str = "classification") -> 'tf.data.Dataset':
    """Crée un pipeline tf.data optimisé."""
    try:
        import tensorflow as tf
        
        if transforms is None:
            transforms = ImageTransforms()
        
        # Créer le dataset de base
        if labels is not None:
            dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        
        # Fonction de chargement et préprocessing
        def load_and_preprocess(path, label=None):
            image = transforms.load_and_preprocess_image(path)
            if label is not None:
                if task_type == "classification":
                    return image, label
                else:  # regression
                    return image, tf.cast(label, tf.float32)
            return image
        
        # Mapper la fonction de chargement
        if labels is not None:
            dataset = dataset.map(
                lambda path, label: load_and_preprocess(path, label),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        else:
            dataset = dataset.map(
                lambda path: load_and_preprocess(path),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Cache si le dataset n'est pas trop grand
        if len(image_paths) < 10000:  # Threshold arbitraire
            dataset = dataset.cache()
        
        # Shuffle si demandé
        if shuffle:
            dataset = dataset.shuffle(buffer_size=min(1000, len(image_paths)))
        
        # Batch
        dataset = dataset.batch(batch_size)
        
        # Augmentation de données (seulement en entraînement)
        if augmentation is not None and shuffle:  # shuffle indique training
            aug_layer = augmentation.get_augmentation_layer()
            if aug_layer is not None:
                dataset = dataset.map(
                    lambda x, y: (aug_layer(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
        
        # Prefetch pour les performances
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
        
    except ImportError:
        raise ImportError("TensorFlow requis pour créer le pipeline de données")
