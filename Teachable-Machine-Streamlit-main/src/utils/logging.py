"""
Système de logging structuré.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import datetime
import json


class StructuredFormatter(logging.Formatter):
    """Formateur de logs structuré en JSON."""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Ajouter des données supplémentaires si disponibles
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data, ensure_ascii=False)


class TeachableLogger:
    """Logger personnalisé pour l'application."""
    
    def __init__(self, name: str = "teachable_machine", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Éviter la duplication des handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Configure les handlers de logging."""
        # Handler pour fichier (JSON structuré)
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.name}_{datetime.date.today().isoformat()}.jsonl"
        )
        file_handler.setFormatter(StructuredFormatter())
        file_handler.setLevel(logging.INFO)
        
        # Handler pour console (format lisible)
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, **kwargs):
        """Log info avec données supplémentaires."""
        self.logger.info(message, extra={'extra_data': kwargs})
    
    def warning(self, message: str, **kwargs):
        """Log warning avec données supplémentaires."""
        self.logger.warning(message, extra={'extra_data': kwargs})
    
    def error(self, message: str, **kwargs):
        """Log error avec données supplémentaires."""
        self.logger.error(message, extra={'extra_data': kwargs})
    
    def debug(self, message: str, **kwargs):
        """Log debug avec données supplémentaires."""
        self.logger.debug(message, extra={'extra_data': kwargs})
    
    def log_training_start(self, config: dict, dataset_info: dict):
        """Log le début d'un entraînement."""
        self.info(
            "Début d'entraînement",
            event_type="training_start",
            config=config,
            dataset_info=dataset_info
        )
    
    def log_training_epoch(self, epoch: int, metrics: dict, time_elapsed: float):
        """Log une epoch d'entraînement."""
        self.info(
            f"Epoch {epoch} terminée",
            event_type="training_epoch",
            epoch=epoch,
            metrics=metrics,
            time_elapsed=time_elapsed
        )
    
    def log_training_end(self, final_metrics: dict, total_time: float, artifacts_path: str):
        """Log la fin d'un entraînement."""
        self.info(
            "Entraînement terminé",
            event_type="training_end",
            final_metrics=final_metrics,
            total_time=total_time,
            artifacts_path=artifacts_path
        )
    
    def log_prediction(self, image_path: str, prediction: dict, processing_time: float):
        """Log une prédiction."""
        self.info(
            "Prédiction effectuée",
            event_type="prediction",
            image_path=image_path,
            prediction=prediction,
            processing_time=processing_time
        )
    
    def log_error(self, error_type: str, error_message: str, context: dict = None):
        """Log une erreur avec contexte."""
        self.error(
            error_message,
            event_type="error",
            error_type=error_type,
            context=context or {}
        )
    
    def log_data_loading(self, dataset_path: str, num_samples: int, splits: dict):
        """Log le chargement de données."""
        self.info(
            "Données chargées",
            event_type="data_loading",
            dataset_path=dataset_path,
            num_samples=num_samples,
            splits=splits
        )


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> TeachableLogger:
    """Configure le système de logging."""
    # Configurer le niveau de logging global
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)
    
    # Créer le logger principal
    logger = TeachableLogger("teachable_machine", log_dir)
    
    return logger


# Instance globale du logger
logger = setup_logging()
