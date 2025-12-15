"""
Tests pour les chargeurs de données.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.loaders import (
    ClassificationDataLoader,
    RegressionDataLoader, 
    detect_dataset_type,
    create_data_loader
)
from src.data.utils import create_sample_dataset, create_sample_regression_dataset


class TestClassificationDataLoader:
    """Tests pour le chargeur de classification."""
    
    def setup_method(self):
        """Setup avant chaque test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.loader = ClassificationDataLoader()
    
    def teardown_method(self):
        """Cleanup après chaque test."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_create_sample_dataset(self):
        """Test de création d'un dataset d'exemple."""
        dataset_path = create_sample_dataset(
            str(self.temp_dir / "sample"),
            num_classes=3,
            samples_per_class=5
        )
        
        assert dataset_path.exists()
        
        # Vérifier la structure
        class_dirs = list(dataset_path.iterdir())
        assert len(class_dirs) == 3
        
        for class_dir in class_dirs:
            assert class_dir.is_dir()
            images = list(class_dir.glob("*.png"))
            assert len(images) == 5
    
    def test_load_from_directory(self):
        """Test de chargement depuis un dossier."""
        # Créer un dataset d'exemple
        dataset_path = create_sample_dataset(
            str(self.temp_dir / "test_dataset"),
            num_classes=2,
            samples_per_class=3
        )
        
        # Charger avec le loader
        dataset_info = self.loader.load_from_directory(str(dataset_path))
        
        # Vérifications
        assert dataset_info.task_type == "classification"
        assert dataset_info.num_classes == 2
        assert dataset_info.num_samples == 6  # 2 classes × 3 samples
        assert len(dataset_info.class_names) == 2
        assert all(name.startswith("classe_") for name in dataset_info.class_names)
    
    def test_detect_dataset_type(self):
        """Test de détection automatique du type."""
        # Créer un dataset de classification
        dataset_path = create_sample_dataset(
            str(self.temp_dir / "classification"),
            num_classes=2,
            samples_per_class=2
        )
        
        detected_type = detect_dataset_type(str(dataset_path))
        assert detected_type == "classification_directory"
    
    def test_create_splits(self):
        """Test de création de splits."""
        # Créer un dataset suffisamment grand
        dataset_path = create_sample_dataset(
            str(self.temp_dir / "split_test"),
            num_classes=2,
            samples_per_class=10
        )
        
        dataset_info = self.loader.load_from_directory(str(dataset_path))
        
        # Créer les splits
        splits = self.loader.create_train_val_test_split(
            dataset_info,
            val_split=0.2,
            test_split=0.1,
            random_state=42
        )
        
        # Vérifications
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        # Compter les échantillons
        total_samples = 0
        for split_data in splits.values():
            for images in split_data.values():
                total_samples += len(images)
        
        assert total_samples == dataset_info.num_samples


class TestRegressionDataLoader:
    """Tests pour le chargeur de régression."""
    
    def setup_method(self):
        """Setup avant chaque test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.loader = RegressionDataLoader()
    
    def teardown_method(self):
        """Cleanup après chaque test."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_create_sample_regression_dataset(self):
        """Test de création d'un dataset de régression."""
        try:
            images_dir, csv_path = create_sample_regression_dataset(
                str(self.temp_dir / "regression"),
                num_samples=10
            )
            
            assert images_dir.exists()
            assert csv_path.exists()
            
            # Vérifier les images
            images = list(images_dir.glob("*.png"))
            assert len(images) == 10
            
            # Vérifier le CSV
            import pandas as pd
            df = pd.read_csv(csv_path)
            assert len(df) == 10
            assert 'image_path' in df.columns
            assert 'target' in df.columns
            
        except ImportError:
            pytest.skip("Pandas ou PIL non disponible")
    
    def test_load_from_csv(self):
        """Test de chargement depuis CSV."""
        try:
            # Créer un dataset de régression
            images_dir, csv_path = create_sample_regression_dataset(
                str(self.temp_dir / "csv_test"),
                num_samples=5
            )
            
            # Charger avec le loader
            dataset_info = self.loader.load_from_csv(
                str(csv_path),
                base_path=str(self.temp_dir / "csv_test")
            )
            
            # Vérifications
            assert dataset_info.task_type == "regression"
            assert dataset_info.num_samples == 5
            assert dataset_info.target_range is not None
            assert len(dataset_info.target_range) == 2
            
        except ImportError:
            pytest.skip("Pandas ou PIL non disponible")


class TestDataLoaderFactory:
    """Tests pour la factory de data loaders."""
    
    def test_create_classification_loader(self):
        """Test de création d'un loader de classification."""
        loader = create_data_loader("classification")
        assert isinstance(loader, ClassificationDataLoader)
        assert loader.task_type == "classification"
    
    def test_create_regression_loader(self):
        """Test de création d'un loader de régression."""
        loader = create_data_loader("regression")
        assert isinstance(loader, RegressionDataLoader)
        assert loader.task_type == "regression"
    
    def test_invalid_task_type(self):
        """Test avec un type de tâche invalide."""
        with pytest.raises(ValueError):
            create_data_loader("invalid_task")


def test_supported_formats():
    """Test des formats d'images supportés."""
    loader = ClassificationDataLoader()
    
    # Formats supportés
    assert loader._is_image_file(Path("test.jpg"))
    assert loader._is_image_file(Path("test.PNG"))
    assert loader._is_image_file(Path("test.jpeg"))
    assert loader._is_image_file(Path("test.bmp"))
    
    # Formats non supportés
    assert not loader._is_image_file(Path("test.txt"))
    assert not loader._is_image_file(Path("test.pdf"))
    assert not loader._is_image_file(Path("test.mp4"))


if __name__ == "__main__":
    pytest.main([__file__])
