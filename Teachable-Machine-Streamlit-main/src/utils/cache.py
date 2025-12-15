"""
Système de cache pour optimiser les performances.
"""

import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional, Union, Callable
import time
import functools


class CacheManager:
    """Gestionnaire de cache basé sur les fichiers."""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, key: str) -> str:
        """Génère une clé de cache hashée."""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Retourne le chemin du fichier de cache."""
        cache_key = self._get_cache_key(key)
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur du cache."""
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Vérifier l'expiration
                if 'expires_at' in cached_data:
                    if time.time() > cached_data['expires_at']:
                        cache_path.unlink()  # Supprimer le cache expiré
                        return default
                
                return cached_data['value']
            except Exception:
                # En cas d'erreur, supprimer le cache corrompu
                cache_path.unlink()
                return default
        
        return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Stocke une valeur dans le cache."""
        cache_path = self._get_cache_path(key)
        
        cached_data = {'value': value}
        
        if ttl is not None:
            cached_data['expires_at'] = time.time() + ttl
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde en cache : {e}")
    
    def delete(self, key: str) -> bool:
        """Supprime une entrée du cache."""
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False
    
    def clear(self) -> int:
        """Vide tout le cache et retourne le nombre de fichiers supprimés."""
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1
        return count
    
    def get_cache_size(self) -> tuple[int, int]:
        """Retourne la taille du cache (nombre de fichiers, taille en bytes)."""
        files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in files)
        return len(files), total_size
    
    def cleanup_expired(self) -> int:
        """Supprime les entrées expirées et retourne le nombre supprimé."""
        count = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                if 'expires_at' in cached_data and current_time > cached_data['expires_at']:
                    cache_file.unlink()
                    count += 1
            except Exception:
                # Supprimer les fichiers corrompus
                cache_file.unlink()
                count += 1
        
        return count


def cached(ttl: Optional[int] = None, cache_manager: Optional[CacheManager] = None):
    """Décorateur pour mettre en cache les résultats de fonction."""
    
    def decorator(func: Callable) -> Callable:
        cache = cache_manager or default_cache_manager
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Créer une clé de cache basée sur la fonction et les arguments
            cache_key = f"{func.__name__}_{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Essayer de récupérer depuis le cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Calculer le résultat et le mettre en cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def memory_cache(maxsize: int = 128):
    """Cache mémoire simple avec LRU."""
    from functools import lru_cache
    return lru_cache(maxsize=maxsize)


class DatasetCache:
    """Cache spécialisé pour les datasets."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    def get_dataset_info(self, dataset_path: str) -> Optional[dict]:
        """Récupère les informations mises en cache d'un dataset."""
        key = f"dataset_info_{dataset_path}"
        return self.cache.get(key)
    
    def set_dataset_info(self, dataset_path: str, info: dict, ttl: int = 3600) -> None:
        """Met en cache les informations d'un dataset."""
        key = f"dataset_info_{dataset_path}"
        self.cache.set(key, info, ttl)
    
    def get_processed_images(self, image_path: str, processing_params: dict) -> Optional[Any]:
        """Récupère une image traitée du cache."""
        params_hash = hashlib.md5(str(processing_params).encode()).hexdigest()
        key = f"processed_image_{image_path}_{params_hash}"
        return self.cache.get(key)
    
    def set_processed_images(self, image_path: str, processing_params: dict, 
                           processed_data: Any, ttl: int = 1800) -> None:
        """Met en cache une image traitée."""
        params_hash = hashlib.md5(str(processing_params).encode()).hexdigest()
        key = f"processed_image_{image_path}_{params_hash}"
        self.cache.set(key, processed_data, ttl)


class ModelCache:
    """Cache spécialisé pour les modèles."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
    
    def get_model_predictions(self, model_path: str, image_hash: str) -> Optional[dict]:
        """Récupère des prédictions mises en cache."""
        key = f"predictions_{model_path}_{image_hash}"
        return self.cache.get(key)
    
    def set_model_predictions(self, model_path: str, image_hash: str, 
                            predictions: dict, ttl: int = 1800) -> None:
        """Met en cache des prédictions."""
        key = f"predictions_{model_path}_{image_hash}"
        self.cache.set(key, predictions, ttl)


# Instance globale du gestionnaire de cache
default_cache_manager = CacheManager()

# Caches spécialisés
dataset_cache = DatasetCache(default_cache_manager)
model_cache = ModelCache(default_cache_manager)
