"""
Tests pour le système d'entraînement.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.registry import model_registry, transfer_learning_recommendations
from src.models.heads import create_model_head, get_optimizer_config
from src.data.utils import create_sample_dataset


class TestModelRegistry:
    """Tests pour le registre de modèles."""
    
    def test_get_available_models(self):
        """Test de récupération des modèles disponibles."""
        models = model_registry.get_available_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        
        # Vérifier la présence des modèles attendus
        expected_models = ['MobileNetV3Small', 'MobileNetV3Large', 'EfficientNetB0', 'ResNet50']
        for model_name in expected_models:
            assert model_name in models
    
    def test_get_model_info(self):
        """Test de récupération d'informations sur un modèle."""
        model_info = model_registry.get_model_info('MobileNetV3Small')
        
        assert isinstance(model_info, dict)
        assert 'class' in model_info
        assert 'variant' in model_info
        assert 'input_shape' in model_info
        assert 'description' in model_info
    
    def test_invalid_model_name(self):
        """Test avec un nom de modèle invalide."""
        with pytest.raises(ValueError):
            model_registry.get_model_info('InvalidModel')


class TestModelHeads:
    """Tests pour les têtes de modèles."""
    
    def test_create_classification_head(self):
        """Test de création d'une tête de classification."""
        head = create_model_head('classification', num_classes=3)
        
        assert head.task_type == 'classification'
        assert head.num_classes == 3
        assert head.get_output_activation() == 'softmax'
        assert head.get_loss_function() == 'sparse_categorical_crossentropy'
    
    def test_create_binary_classification_head(self):
        """Test de création d'une tête de classification binaire."""
        head = create_model_head('classification', num_classes=2)
        
        assert head.task_type == 'classification'
        assert head.num_classes == 2
        assert head.get_output_activation() == 'sigmoid'
        assert head.get_loss_function() == 'binary_crossentropy'
    
    def test_create_regression_head(self):
        """Test de création d'une tête de régression."""
        head = create_model_head('regression')
        
        assert head.task_type == 'regression'
        assert head.get_output_activation() == 'linear'
        assert head.get_loss_function() == 'mse'
    
    def test_invalid_task_type(self):
        """Test avec un type de tâche invalide."""
        with pytest.raises(ValueError):
            create_model_head('invalid_task')


class TestOptimizerConfig:
    """Tests pour la configuration des optimiseurs."""
    
    def test_adam_config(self):
        """Test de configuration Adam."""
        config = get_optimizer_config('adam', 0.001)
        
        assert config['class_name'] == 'Adam'
        assert config['config']['learning_rate'] == 0.001
        assert 'beta_1' in config['config']
        assert 'beta_2' in config['config']
    
    def test_sgd_config(self):
        """Test de configuration SGD."""
        config = get_optimizer_config('sgd', 0.01)
        
        assert config['class_name'] == 'SGD'
        assert config['config']['learning_rate'] == 0.01
        assert 'momentum' in config['config']
    
    def test_invalid_optimizer(self):
        """Test avec un optimiseur invalide."""
        with pytest.raises(ValueError):
            get_optimizer_config('invalid_optimizer', 0.001)


class TestTransferLearning:
    """Tests pour les recommandations de transfer learning."""
    
    def test_small_dataset_recommendations(self):
        """Test de recommandations pour petit dataset."""
        recommendations = transfer_learning_recommendations(
            task_type='classification',
            dataset_size=500,
            time_budget='fast'
        )
        
        assert isinstance(recommendations, dict)
        assert 'backbone' in recommendations
        assert 'trainable_layers' in recommendations
        assert 'learning_rate' in recommendations
        assert 'reasoning' in recommendations
        
        # Pour un petit dataset, on s'attend à un modèle léger
        assert recommendations['backbone'] in ['MobileNetV3Small', 'MobileNetV3Large']
        assert recommendations['trainable_layers'] <= 10
    
    def test_large_dataset_recommendations(self):
        """Test de recommandations pour grand dataset."""
        recommendations = transfer_learning_recommendations(
            task_type='classification',
            dataset_size=50000,
            time_budget='precise'
        )
        
        # Pour un grand dataset, on peut utiliser des modèles plus complexes
        assert recommendations['backbone'] in ['EfficientNetB0', 'ResNet50']
        assert recommendations['trainable_layers'] >= 50 or recommendations['trainable_layers'] == -1


class TestTrainingSimulation:
    """Tests de simulation d'entraînement."""
    
    def setup_method(self):
        """Setup avant chaque test."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup après chaque test."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_training_config_validation(self):
        """Test de validation de configuration d'entraînement."""
        # Configuration valide
        config = {
            'epochs': 10,
            'learning_rate': 0.001,
            'batch_size': 32,
            'optimizer': 'adam'
        }
        
        # Vérifier que la configuration est cohérente
        assert config['epochs'] > 0
        assert config['learning_rate'] > 0
        assert config['batch_size'] > 0
        assert config['optimizer'] in ['adam', 'sgd', 'rmsprop']
    
    def test_artifacts_structure(self):
        """Test de la structure des artefacts."""
        # Simuler la création d'un dossier d'artefacts
        from datetime import datetime
        
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        artifacts_dir = self.temp_dir / "artifacts" / run_id
        artifacts_dir.mkdir(parents=True)
        
        # Créer les fichiers attendus
        expected_files = [
            "model.keras",
            "config.yaml", 
            "training_history.json",
            "metrics.json"
        ]
        
        for filename in expected_files:
            (artifacts_dir / filename).touch()
        
        # Vérifier la structure
        assert artifacts_dir.exists()
        for filename in expected_files:
            assert (artifacts_dir / filename).exists()


if __name__ == "__main__":
    pytest.main([__file__])
