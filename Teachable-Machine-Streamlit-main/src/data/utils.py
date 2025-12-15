"""
Utilitaires pour la gestion des données.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import hashlib


def create_directory_structure(base_path: str, splits: Dict[str, Dict[str, List]], 
                              copy_files: bool = True) -> Path:
    """Crée une structure de dossiers organisée pour les splits."""
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, classes_data in splits.items():
        split_dir = base_path / split_name
        split_dir.mkdir(exist_ok=True)
        
        for class_name, image_paths in classes_data.items():
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            if copy_files:
                for i, src_path in enumerate(image_paths):
                    src_path = Path(src_path)
                    dst_path = class_dir / f"{i:04d}_{src_path.name}"
                    
                    if not dst_path.exists():
                        shutil.copy2(src_path, dst_path)
    
    return base_path


def calculate_class_weights(class_distribution: Dict[str, int]) -> Dict[str, float]:
    """Calcule les poids des classes pour gérer le déséquilibre."""
    total_samples = sum(class_distribution.values())
    num_classes = len(class_distribution)
    
    # Poids inversement proportionnels à la fréquence
    class_weights = {}
    for class_name, count in class_distribution.items():
        weight = total_samples / (num_classes * count)
        class_weights[class_name] = weight
    
    return class_weights


def generate_data_hash(data_paths: List[str]) -> str:
    """Génère un hash unique pour un ensemble de données."""
    # Créer une signature basée sur les chemins et les tailles de fichiers
    signature_data = []
    
    for path in sorted(data_paths):
        path_obj = Path(path)
        if path_obj.exists():
            stat = path_obj.stat()
            signature_data.append(f"{path}:{stat.st_size}:{stat.st_mtime}")
    
    signature_str = "|".join(signature_data)
    return hashlib.md5(signature_str.encode()).hexdigest()


def validate_image_files(image_paths: List[str]) -> Tuple[List[str], List[str]]:
    """Valide une liste de fichiers image et retourne les valides et invalides."""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    
    valid_files = []
    invalid_files = []
    
    for path in image_paths:
        path_obj = Path(path)
        
        if not path_obj.exists():
            invalid_files.append(f"{path} (fichier inexistant)")
            continue
        
        if path_obj.suffix.lower() not in valid_extensions:
            invalid_files.append(f"{path} (extension non supportée)")
            continue
        
        # Vérifier si le fichier peut être ouvert (test basic)
        try:
            with open(path_obj, 'rb') as f:
                # Lire les premiers bytes pour vérifier la signature
                header = f.read(16)
                if len(header) < 4:
                    invalid_files.append(f"{path} (fichier trop petit)")
                    continue
            
            valid_files.append(path)
            
        except Exception as e:
            invalid_files.append(f"{path} (erreur lecture: {e})")
    
    return valid_files, invalid_files


def save_dataset_metadata(dataset_info: 'DatasetInfo', output_path: str) -> None:
    """Sauvegarde les métadonnées d'un dataset."""
    metadata = {
        'name': dataset_info.name,
        'path': str(dataset_info.path),
        'task_type': dataset_info.task_type,
        'num_samples': dataset_info.num_samples,
        'num_classes': dataset_info.num_classes,
        'class_names': dataset_info.class_names,
        'class_distribution': dataset_info.class_distribution,
        'target_range': dataset_info.target_range,
        'splits': dataset_info.splits,
        'image_shape': dataset_info.image_shape,
        'created_at': dataset_info.created_at.isoformat() if dataset_info.created_at else None
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_dataset_metadata(metadata_path: str) -> Dict[str, Any]:
    """Charge les métadonnées d'un dataset."""
    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_dataset_statistics(dataset_info: 'DatasetInfo') -> Dict[str, Any]:
    """Calcule des statistiques détaillées sur un dataset."""
    stats = {
        'basic_info': {
            'name': dataset_info.name,
            'task_type': dataset_info.task_type,
            'total_samples': dataset_info.num_samples
        }
    }
    
    if dataset_info.task_type == "classification":
        stats['classification'] = {
            'num_classes': dataset_info.num_classes,
            'class_names': dataset_info.class_names,
            'class_distribution': dataset_info.class_distribution,
        }
        
        # Calculer le déséquilibre des classes
        if dataset_info.class_distribution:
            counts = list(dataset_info.class_distribution.values())
            min_count = min(counts)
            max_count = max(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            stats['classification']['balance'] = {
                'min_samples_per_class': min_count,
                'max_samples_per_class': max_count,
                'imbalance_ratio': imbalance_ratio,
                'is_balanced': imbalance_ratio <= 2.0  # Seuil arbitraire
            }
    
    elif dataset_info.task_type == "regression":
        stats['regression'] = {
            'target_range': dataset_info.target_range,
        }
        
        if dataset_info.target_range:
            min_val, max_val = dataset_info.target_range
            stats['regression']['target_statistics'] = {
                'min_value': min_val,
                'max_value': max_val,
                'range': max_val - min_val
            }
    
    # Informations sur les splits
    if dataset_info.splits:
        stats['splits'] = dataset_info.splits
        total_split_samples = sum(dataset_info.splits.values())
        stats['split_proportions'] = {
            split: count / total_split_samples 
            for split, count in dataset_info.splits.items()
        }
    
    return stats


def create_sample_dataset(output_dir: str, num_classes: int = 3, 
                         samples_per_class: int = 10) -> Path:
    """Crée un dataset d'exemple avec des images synthétiques."""
    try:
        import numpy as np
        from PIL import Image, ImageDraw
        import random
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        class_names = [f"classe_{i}" for i in range(num_classes)]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                 (255, 0, 255), (0, 255, 255)]
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = output_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            base_color = colors[class_idx % len(colors)]
            
            for sample_idx in range(samples_per_class):
                # Créer une image synthétique simple
                img = Image.new('RGB', (224, 224), base_color)
                draw = ImageDraw.Draw(img)
                
                # Ajouter quelques formes aléatoires
                for _ in range(random.randint(3, 8)):
                    shape_type = random.choice(['rectangle', 'circle'])
                    x1, y1 = random.randint(0, 150), random.randint(0, 150)
                    x2, y2 = x1 + random.randint(20, 74), y1 + random.randint(20, 74)
                    
                    # Couleur aléatoire avec variation
                    color = tuple(max(0, min(255, c + random.randint(-50, 50))) 
                                for c in base_color)
                    
                    if shape_type == 'rectangle':
                        draw.rectangle([x1, y1, x2, y2], fill=color)
                    else:
                        draw.ellipse([x1, y1, x2, y2], fill=color)
                
                # Sauvegarder
                img_path = class_dir / f"sample_{sample_idx:03d}.png"
                img.save(img_path)
        
        return output_dir
        
    except ImportError:
        raise ImportError("PIL/Pillow requis pour créer des images synthétiques")


def create_sample_regression_dataset(output_dir: str, num_samples: int = 100) -> Tuple[Path, Path]:
    """Crée un dataset d'exemple pour la régression."""
    try:
        import numpy as np
        import pandas as pd
        from PIL import Image, ImageDraw
        import random
        
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Créer des images avec des valeurs cibles corrélées
        data = []
        
        for i in range(num_samples):
            # Générer une valeur cible
            target_value = random.uniform(0, 100)
            
            # Créer une image dont la luminosité est corrélée à la valeur cible
            brightness = int(target_value * 2.55)  # 0-100 -> 0-255
            img = Image.new('RGB', (224, 224), (brightness, brightness, brightness))
            
            # Ajouter du bruit
            draw = ImageDraw.Draw(img)
            for _ in range(random.randint(0, 5)):
                x, y = random.randint(0, 224), random.randint(0, 224)
                radius = random.randint(5, 20)
                noise_brightness = max(0, min(255, brightness + random.randint(-30, 30)))
                color = (noise_brightness, noise_brightness, noise_brightness)
                draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
            
            # Sauvegarder l'image
            img_path = images_dir / f"img_{i:04d}.png"
            img.save(img_path)
            
            # Ajouter aux données
            data.append({
                'image_path': f"images/img_{i:04d}.png",
                'target': target_value
            })
        
        # Créer le CSV
        df = pd.DataFrame(data)
        csv_path = output_dir / "regression_data.csv"
        df.to_csv(csv_path, index=False)
        
        return images_dir, csv_path
        
    except ImportError:
        raise ImportError("pandas et PIL/Pillow requis pour créer le dataset de régression")
