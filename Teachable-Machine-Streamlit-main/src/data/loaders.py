"""
Chargeurs de données pour classification et régression.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import zipfile
import tempfile
import shutil
from sklearn.model_selection import train_test_split
from ..schemas.dataclasses import DatasetInfo
from ..utils.logging import logger
import datetime


class DataLoader:
    """Classe de base pour le chargement de données."""
    
    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def _is_image_file(self, filepath: Path) -> bool:
        """Vérifie si un fichier est une image supportée."""
        return filepath.suffix.lower() in self.supported_formats
    
    def _scan_images(self, directory: Path) -> List[Path]:
        """Scanner récursivement un dossier pour trouver des images."""
        images = []
        for ext in self.supported_formats:
            images.extend(directory.rglob(f"*{ext}"))
            images.extend(directory.rglob(f"*{ext.upper()}"))
        return sorted(images)


class ClassificationDataLoader(DataLoader):
    """Chargeur de données pour classification."""
    
    def __init__(self):
        super().__init__("classification")
    
    def load_from_directory(self, data_path: str) -> DatasetInfo:
        """Charge des données depuis un dossier organisé par classes."""
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Le dossier {data_path} n'existe pas")
        
        # Détecter si c'est déjà organisé en train/val/test
        if self._is_split_structure(data_path):
            return self._load_split_structure(data_path)
        else:
            return self._load_flat_structure(data_path)
    
    def _is_split_structure(self, data_path: Path) -> bool:
        """Détermine si le dossier a une structure train/val/test."""
        subdirs = [d.name.lower() for d in data_path.iterdir() if d.is_dir()]
        return any(split in subdirs for split in ['train', 'training', 'val', 'validation', 'test'])
    
    def _load_split_structure(self, data_path: Path) -> DatasetInfo:
        """Charge des données déjà organisées en splits."""
        splits = {}
        all_classes = set()
        total_samples = 0
        
        # Mapper les noms de dossiers aux splits standards
        split_mapping = {
            'train': 'train', 'training': 'train',
            'val': 'val', 'validation': 'val', 'valid': 'val',
            'test': 'test'
        }
        
        for split_dir in data_path.iterdir():
            if split_dir.is_dir() and split_dir.name.lower() in split_mapping:
                split_name = split_mapping[split_dir.name.lower()]
                
                split_data = self._scan_class_directory(split_dir)
                splits[split_name] = split_data
                
                # Collecter les classes et compter les échantillons
                for class_name, images in split_data.items():
                    all_classes.add(class_name)
                    total_samples += len(images)
        
        class_names = sorted(list(all_classes))
        
        # Calculer la distribution des classes (sur le split train si disponible)
        class_distribution = {}
        train_data = splits.get('train', splits.get(list(splits.keys())[0], {}))
        for class_name in class_names:
            class_distribution[class_name] = len(train_data.get(class_name, []))
        
        return DatasetInfo(
            name=data_path.name,
            path=data_path,
            task_type="classification",
            num_samples=total_samples,
            num_classes=len(class_names),
            class_names=class_names,
            class_distribution=class_distribution,
            splits={split: sum(len(images) for images in data.values()) 
                   for split, data in splits.items()},
            created_at=datetime.datetime.now()
        )
    
    def _load_flat_structure(self, data_path: Path) -> DatasetInfo:
        """Charge des données depuis un dossier plat organisé par classes."""
        class_data = self._scan_class_directory(data_path)
        
        if not class_data:
            raise ValueError(f"Aucune classe trouvée dans {data_path}")
        
        class_names = sorted(list(class_data.keys()))
        total_samples = sum(len(images) for images in class_data.values())
        
        class_distribution = {
            class_name: len(images) 
            for class_name, images in class_data.items()
        }
        
        return DatasetInfo(
            name=data_path.name,
            path=data_path,
            task_type="classification",
            num_samples=total_samples,
            num_classes=len(class_names),
            class_names=class_names,
            class_distribution=class_distribution,
            created_at=datetime.datetime.now()
        )
    
    def _scan_class_directory(self, directory: Path) -> Dict[str, List[Path]]:
        """Scanner un dossier pour extraire les classes et leurs images."""
        class_data = {}
        
        for class_dir in directory.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                images = self._scan_images(class_dir)
                if images:
                    class_data[class_name] = images
        
        return class_data
    
    def create_train_val_test_split(self, dataset_info: DatasetInfo, 
                                   val_split: float = 0.2, 
                                   test_split: float = 0.1,
                                   random_state: int = 42) -> Dict[str, Dict[str, List[Path]]]:
        """Créer les splits train/val/test de manière stratifiée."""
        if dataset_info.task_type != "classification":
            raise ValueError("Cette méthode est uniquement pour la classification")
        
        # Collecter toutes les images avec leurs labels
        all_images = []
        all_labels = []
        
        # Si le dataset est déjà splitté, utiliser seulement le train
        if hasattr(dataset_info, 'splits') and dataset_info.splits:
            # Dataset déjà splitté, on ne peut pas re-splitter
            logger.warning("Dataset déjà splitté, impossible de créer de nouveaux splits")
            return {}
        
        # Scanner le dossier pour collecter les images
        for class_dir in dataset_info.path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                images = self._scan_images(class_dir)
                all_images.extend(images)
                all_labels.extend([class_name] * len(images))
        
        if not all_images:
            raise ValueError("Aucune image trouvée")
        
        # Premier split : séparer train+val et test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            all_images, all_labels, 
            test_size=test_split, 
            random_state=random_state,
            stratify=all_labels
        )
        
        # Deuxième split : séparer train et val
        val_size_adjusted = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_train_val
        )
        
        # Organiser par split et par classe
        splits = {'train': {}, 'val': {}, 'test': {}}
        
        for split_name, (images, labels) in [
            ('train', (X_train, y_train)),
            ('val', (X_val, y_val)),
            ('test', (X_test, y_test))
        ]:
            for image, label in zip(images, labels):
                if label not in splits[split_name]:
                    splits[split_name][label] = []
                splits[split_name][label].append(image)
        
        logger.info(
            f"Splits créés",
            train_samples=len(X_train),
            val_samples=len(X_val),
            test_samples=len(X_test)
        )
        
        return splits


class RegressionDataLoader(DataLoader):
    """Chargeur de données pour régression."""
    
    def __init__(self):
        super().__init__("regression")
    
    def load_from_csv(self, csv_path: str, image_col: str = "image_path", 
                     target_col: str = "target", base_path: str = None) -> DatasetInfo:
        """Charge des données de régression depuis un CSV."""
        csv_path = Path(csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Le fichier CSV {csv_path} n'existe pas")
        
        # Charger le CSV
        df = pd.read_csv(csv_path)
        
        # Vérifier les colonnes requises
        if image_col not in df.columns:
            raise ValueError(f"Colonne '{image_col}' non trouvée dans le CSV")
        if target_col not in df.columns:
            raise ValueError(f"Colonne '{target_col}' non trouvée dans le CSV")
        
        # Résoudre les chemins d'images
        if base_path:
            base_path = Path(base_path)
            df[image_col] = df[image_col].apply(lambda x: base_path / x)
        else:
            df[image_col] = df[image_col].apply(Path)
        
        # Vérifier que les images existent
        missing_images = []
        for img_path in df[image_col]:
            if not img_path.exists() or not self._is_image_file(img_path):
                missing_images.append(str(img_path))
        
        if missing_images:
            logger.warning(f"{len(missing_images)} images manquantes ou invalides")
            # Filtrer les images manquantes
            df = df[df[image_col].apply(lambda x: x.exists() and self._is_image_file(x))]
        
        if df.empty:
            raise ValueError("Aucune image valide trouvée")
        
        # Statistiques sur les targets
        targets = df[target_col].astype(float)
        target_range = (targets.min(), targets.max())
        
        return DatasetInfo(
            name=csv_path.stem,
            path=csv_path.parent,
            task_type="regression",
            num_samples=len(df),
            target_range=target_range,
            created_at=datetime.datetime.now()
        )
    
    def create_train_val_test_split(self, csv_path: str, image_col: str = "image_path",
                                   target_col: str = "target", val_split: float = 0.2,
                                   test_split: float = 0.1, random_state: int = 42,
                                   base_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Créer les splits train/val/test pour régression."""
        # Charger les données
        df = pd.read_csv(csv_path)
        
        # Résoudre les chemins si nécessaire
        if base_path:
            base_path = Path(base_path)
            df[image_col] = df[image_col].apply(lambda x: base_path / x)
        
        # Filtrer les images valides
        df = df[df[image_col].apply(lambda x: Path(x).exists())]
        
        if df.empty:
            raise ValueError("Aucune image valide trouvée")
        
        # Splits séquentiels
        X = df[[image_col]]
        y = df[target_col]
        
        # Premier split : train+val et test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_split, random_state=random_state
        )
        
        # Deuxième split : train et val
        val_size_adjusted = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state
        )
        
        # Reconstruire les DataFrames
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        logger.info(
            f"Splits de régression créés",
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(test_df)
        )
        
        return train_df, val_df, test_df


class ZipDataLoader:
    """Chargeur pour fichiers ZIP."""
    
    def __init__(self):
        self.temp_dirs = []
    
    def extract_zip(self, zip_path: str, extract_to: str = None) -> Path:
        """Extrait un fichier ZIP et retourne le dossier d'extraction."""
        zip_path = Path(zip_path)
        
        if not zip_path.exists():
            raise FileNotFoundError(f"Le fichier ZIP {zip_path} n'existe pas")
        
        if extract_to is None:
            extract_dir = Path(tempfile.mkdtemp(prefix="teachable_"))
            self.temp_dirs.append(extract_dir)
        else:
            extract_dir = Path(extract_to)
            extract_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            logger.info(f"ZIP extrait vers {extract_dir}")
            return extract_dir
            
        except zipfile.BadZipFile:
            raise ValueError(f"Le fichier {zip_path} n'est pas un ZIP valide")
    
    def cleanup(self):
        """Nettoie les dossiers temporaires."""
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()
    
    def __del__(self):
        """Cleanup automatique."""
        self.cleanup()


def create_data_loader(task_type: str) -> DataLoader:
    """Factory pour créer le bon type de data loader."""
    if task_type == "classification":
        return ClassificationDataLoader()
    elif task_type == "regression":
        return RegressionDataLoader()
    else:
        raise ValueError(f"Type de tâche non supporté : {task_type}")


def detect_dataset_type(data_path: str) -> str:
    """Détecte automatiquement le type de dataset."""
    data_path = Path(data_path)
    
    if data_path.is_file():
        if data_path.suffix.lower() == '.csv':
            return "csv_regression"
        elif data_path.suffix.lower() == '.zip':
            return "zip"
    elif data_path.is_dir():
        # Vérifier s'il y a des sous-dossiers qui ressemblent à des classes
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        if subdirs:
            return "classification_directory"
    
    return "unknown"
