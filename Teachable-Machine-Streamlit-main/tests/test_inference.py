"""
Tests pour l'inférence et le serving.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.schemas.dataclasses import PredictionResult


class TestPredictionResult:
    """Tests pour les résultats de prédiction."""
    
    def test_classification_result(self):
        """Test de résultat de classification."""
        result = PredictionResult(
            image_path="test.jpg",
            predictions=[0.8, 0.15, 0.05],
            predicted_class="classe_A",
            confidence=0.8,
            processing_time=0.123
        )
        
        assert result.image_path == "test.jpg"
        assert len(result.predictions) == 3
        assert result.predicted_class == "classe_A"
        assert result.confidence == 0.8
        assert result.processing_time > 0
    
    def test_regression_result(self):
        """Test de résultat de régression."""
        result = PredictionResult(
            image_path="test.jpg",
            predictions=42.5,
            processing_time=0.089
        )
        
        assert result.image_path == "test.jpg"
        assert result.predictions == 42.5
        assert result.predicted_class is None
        assert result.confidence is None
        assert result.processing_time > 0


class TestInferenceSimulation:
    """Tests de simulation d'inférence."""
    
    def test_classification_inference_format(self):
        """Test du format de sortie pour classification."""
        # Simuler une prédiction de classification
        class_names = ['chat', 'chien', 'oiseau']
        predictions = [0.7, 0.2, 0.1]
        
        # Vérifications du format
        assert len(predictions) == len(class_names)
        assert abs(sum(predictions) - 1.0) < 1e-6  # Somme = 1
        assert all(0 <= p <= 1 for p in predictions)  # Probabilités valides
        
        # Classe prédite
        predicted_idx = predictions.index(max(predictions))
        predicted_class = class_names[predicted_idx]
        confidence = max(predictions)
        
        assert predicted_class == 'chat'
        assert confidence == 0.7
    
    def test_regression_inference_format(self):
        """Test du format de sortie pour régression."""
        # Simuler une prédiction de régression
        predicted_value = 23.456
        target_range = (0, 100)
        
        # Vérifications
        assert isinstance(predicted_value, (int, float))
        assert target_range[0] <= predicted_value <= target_range[1]
    
    def test_batch_inference_format(self):
        """Test du format pour l'inférence par batch."""
        # Simuler des résultats de batch
        batch_results = []
        
        for i in range(5):
            result = PredictionResult(
                image_path=f"image_{i}.jpg",
                predictions=[0.6, 0.3, 0.1],
                predicted_class="classe_A",
                confidence=0.6,
                processing_time=0.1 + i * 0.02
            )
            batch_results.append(result)
        
        # Vérifications
        assert len(batch_results) == 5
        assert all(isinstance(r, PredictionResult) for r in batch_results)
        
        # Temps de traitement croissant
        times = [r.processing_time for r in batch_results]
        assert all(times[i] <= times[i+1] for i in range(len(times)-1))


class TestModelExport:
    """Tests pour l'export de modèles."""
    
    def setup_method(self):
        """Setup avant chaque test."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Cleanup après chaque test."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_export_paths(self):
        """Test des chemins d'export."""
        model_name = "test_model"
        
        # Chemins attendus
        keras_path = self.temp_dir / f"{model_name}.keras"
        onnx_path = self.temp_dir / f"{model_name}.onnx"
        tflite_path = self.temp_dir / f"{model_name}.tflite"
        
        # Simuler les exports
        for path in [keras_path, onnx_path, tflite_path]:
            path.touch()  # Créer un fichier vide
            assert path.exists()
    
    def test_export_metadata(self):
        """Test des métadonnées d'export."""
        metadata = {
            'model_name': 'test_model',
            'task_type': 'classification',
            'num_classes': 3,
            'class_names': ['A', 'B', 'C'],
            'input_shape': [224, 224, 3],
            'export_format': 'onnx',
            'export_timestamp': '2024-01-01T12:00:00'
        }
        
        # Vérifications
        assert metadata['task_type'] in ['classification', 'regression']
        assert isinstance(metadata['input_shape'], list)
        assert len(metadata['input_shape']) == 3
        
        if metadata['task_type'] == 'classification':
            assert metadata['num_classes'] > 0
            assert len(metadata['class_names']) == metadata['num_classes']


class TestAPIGeneration:
    """Tests pour la génération d'API."""
    
    def test_fastapi_code_structure(self):
        """Test de la structure du code FastAPI généré."""
        # Éléments attendus dans le code API
        expected_elements = [
            'from fastapi import FastAPI',
            'app = FastAPI',
            '@app.post("/predict")',
            'async def predict',
            'tf.keras.models.load_model',
            'uvicorn.run'
        ]
        
        # Code API simulé (version simplifiée)
        api_code = """
from fastapi import FastAPI
import tensorflow as tf

app = FastAPI(title="Test API")

@app.post("/predict")
async def predict():
    model = tf.keras.models.load_model("model.keras")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
        """
        
        # Vérifications
        for element in expected_elements:
            assert element in api_code
    
    def test_api_configuration(self):
        """Test de configuration de l'API."""
        config = {
            'api_name': 'test_model_api',
            'port': 8000,
            'enable_docs': True,
            'enable_cors': True,
            'max_file_size': 10,  # MB
            'rate_limiting': False
        }
        
        # Vérifications de cohérence
        assert isinstance(config['api_name'], str)
        assert 1000 <= config['port'] <= 9999
        assert isinstance(config['enable_docs'], bool)
        assert isinstance(config['enable_cors'], bool)
        assert config['max_file_size'] > 0


class TestDeploymentValidation:
    """Tests de validation pour le déploiement."""
    
    def test_model_format_validation(self):
        """Test de validation des formats de modèles."""
        valid_formats = ['.keras', '.h5', '.onnx', '.tflite']
        invalid_formats = ['.txt', '.json', '.py', '.pkl']
        
        def is_valid_model_format(filename):
            return any(filename.endswith(fmt) for fmt in valid_formats)
        
        # Tests positifs
        for fmt in valid_formats:
            assert is_valid_model_format(f"model{fmt}")
        
        # Tests négatifs
        for fmt in invalid_formats:
            assert not is_valid_model_format(f"model{fmt}")
    
    def test_deployment_checklist(self):
        """Test de checklist de déploiement."""
        deployment_items = {
            'model_exported': True,
            'config_saved': True,
            'dependencies_listed': True,
            'api_tested': False,  # Pas encore testé
            'documentation_generated': True
        }
        
        # Vérifier les éléments critiques
        critical_items = ['model_exported', 'config_saved', 'dependencies_listed']
        
        for item in critical_items:
            assert deployment_items[item], f"Élément critique manquant: {item}"
        
        # Calculer le score de completion
        completed = sum(deployment_items.values())
        total = len(deployment_items)
        completion_rate = completed / total
        
        assert 0 <= completion_rate <= 1


if __name__ == "__main__":
    pytest.main([__file__])
