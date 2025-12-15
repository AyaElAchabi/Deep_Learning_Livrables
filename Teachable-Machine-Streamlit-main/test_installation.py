#!/usr/bin/env python3
"""
Script de test d'installation et de configuration.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Vérifie la version de Python."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 10:
        print("❌ Python 3.10+ requis")
        return False
    
    print("✅ Version Python OK")
    return True

def check_dependencies():
    """Vérifie les dépendances critiques."""
    critical_deps = [
        'streamlit',
        'tensorflow',
        'pandas',
        'numpy',
        'scikit-learn',
        'pyyaml',
        'pydantic'
    ]
    
    print("\nVérification des dépendances:")
    all_good = True
    
    for dep in critical_deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - À installer")
            all_good = False
    
    return all_good

def check_structure():
    """Vérifie la structure du projet."""
    print("\nVérification de la structure:")
    
    required_files = [
        'app.py',
        'config.yaml',
        'requirements.txt',
        'Makefile',
        'README.md'
    ]
    
    required_dirs = [
        'src',
        'pages',
        'tests',
        'samples',
        'artifacts'
    ]
    
    all_good = True
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} manquant")
            all_good = False
    
    for dir in required_dirs:
        if Path(dir).exists():
            print(f"✅ {dir}/")
        else:
            print(f"❌ {dir}/ manquant")
            all_good = False
    
    return all_good

def test_imports():
    """Test les imports des modules principaux."""
    print("\nTest des imports:")
    
    modules = [
        'src.utils.config',
        'src.utils.logging',
        'src.data.loaders',
        'src.models.registry',
        'src.schemas.dataclasses'
    ]
    
    all_good = True
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module} - {e}")
            all_good = False
    
    return all_good

def test_config():
    """Test la configuration."""
    print("\nTest de la configuration:")
    
    try:
        sys.path.append('src')
        from src.utils.config import config_manager
        
        config = config_manager.load_config()
        print(f"✅ Configuration chargée: {config.app.title}")
        
        # Test de validation
        config_dict = config.dict()
        from src.utils.config import validate_config
        is_valid, errors = validate_config(config_dict)
        
        if is_valid:
            print("✅ Configuration valide")
        else:
            print(f"❌ Configuration invalide: {errors}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur de configuration: {e}")
        return False

def run_basic_tests():
    """Exécute les tests de base."""
    print("\nExécution des tests de base:")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Tests passés")
            return True
        else:
            print(f"❌ Tests échoués:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Tests timeout (>60s)")
        return False
    except FileNotFoundError:
        print("❌ pytest non trouvé")
        return False

def main():
    """Fonction principale."""
    print("🤖 Test d'installation Teachable Machine Streamlit")
    print("=" * 50)
    
    checks = [
        ("Version Python", check_python_version),
        ("Dépendances", check_dependencies),
        ("Structure", check_structure),
        ("Imports", test_imports),
        ("Configuration", test_config)
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Erreur lors du test {name}: {e}")
            results.append((name, False))
    
    # Résumé
    print("\n" + "=" * 50)
    print("RÉSUMÉ:")
    
    all_passed = True
    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 Installation complète et fonctionnelle !")
        print("\nPour démarrer l'application:")
        print("  streamlit run app.py")
    else:
        print("\n⚠️ Problèmes détectés. Consultez les erreurs ci-dessus.")
        return 1
    
    # Tests optionnels
    print("\n" + "-" * 30)
    print("Tests optionnels (peuvent prendre du temps):")
    
    if input("Exécuter les tests unitaires ? (y/N): ").lower().startswith('y'):
        run_basic_tests()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
