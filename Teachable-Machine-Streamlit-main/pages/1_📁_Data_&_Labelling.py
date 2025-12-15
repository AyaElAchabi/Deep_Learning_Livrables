"""
Page 1 - Gestion des données et labelling.
"""

import streamlit as st
import sys
from pathlib import Path
import tempfile
import zipfile
import shutil

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data import (
    detect_dataset_type, 
    create_data_loader,
    create_sample_dataset,
    create_sample_regression_dataset,
    ZipDataLoader
)
from src.utils.logging import logger
from src.schemas.dataclasses import DatasetInfo, SessionState


def setup_page():
    """Configuration de la page."""
    st.set_page_config(
        page_title="Données & Labelling - Teachable Machine",
        page_icon="📁",
        layout="wide"
    )


def create_sample_data_section():
    """Section pour créer des données d'exemple."""
    st.subheader("🎮 Données d'exemple")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Classification - Images synthétiques**")
        num_classes = st.number_input("Nombre de classes", min_value=2, max_value=10, value=3)
        samples_per_class = st.number_input("Échantillons par classe", min_value=5, max_value=50, value=10)
        
        if st.button("🎨 Créer dataset de classification"):
            with st.spinner("Création des images synthétiques..."):
                try:
                    output_dir = Path("samples") / "synthetic_classification"
                    dataset_path = create_sample_dataset(
                        str(output_dir), 
                        num_classes=num_classes,
                        samples_per_class=samples_per_class
                    )
                    
                    st.success(f"Dataset créé dans : {dataset_path}")
                    st.session_state.sample_classification_path = str(dataset_path)
                    
                except Exception as e:
                    st.error(f"Erreur lors de la création : {e}")
    
    with col2:
        st.markdown("**Régression - Images corrélées**")
        num_samples = st.number_input("Nombre d'échantillons", min_value=20, max_value=200, value=50)
        
        if st.button("📈 Créer dataset de régression"):
            with st.spinner("Création du dataset de régression..."):
                try:
                    output_dir = Path("samples") / "synthetic_regression"
                    images_dir, csv_path = create_sample_regression_dataset(
                        str(output_dir),
                        num_samples=num_samples
                    )
                    
                    st.success(f"Dataset créé : {csv_path}")
                    st.session_state.sample_regression_path = str(csv_path)
                    
                except Exception as e:
                    st.error(f"Erreur lors de la création : {e}")


def upload_data_section():
    """Section pour uploader des données."""
    st.subheader("📤 Import de données")
    
    tab1, tab2, tab3 = st.tabs(["Dossier d'images", "Fichier ZIP", "CSV de régression"])
    
    with tab1:
        st.markdown("**Classification : Dossier organisé par classes**")
        st.markdown("""
        Structure attendue :
        ```
        mon_dataset/
        ├── classe_A/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── classe_B/
            ├── img3.jpg
            └── img4.jpg
        ```
        """)
        
        # Sélection de dossier (simulation avec path input)
        data_path = st.text_input(
            "Chemin du dossier de données",
            placeholder="/path/to/your/dataset",
            help="Chemin absolu vers votre dossier de données"
        )
        
        if data_path and st.button("📁 Charger le dossier"):
            if Path(data_path).exists():
                with st.spinner("Analyse du dataset..."):
                    try:
                        dataset_type = detect_dataset_type(data_path)
                        
                        if dataset_type == "classification_directory":
                            loader = create_data_loader("classification")
                            dataset_info = loader.load_from_directory(data_path)
                            
                            # Sauvegarder dans la session
                            st.session_state.dataset_info = dataset_info
                            st.session_state.data_source = "directory"
                            st.session_state.data_path = data_path
                            
                            st.success("Dataset chargé avec succès !")
                            st.rerun()
                            
                        else:
                            st.error("Le dossier ne semble pas contenir de classes d'images")
                            
                    except Exception as e:
                        st.error(f"Erreur lors du chargement : {e}")
            else:
                st.error("Le chemin spécifié n'existe pas")
    
    with tab2:
        st.markdown("**Archive ZIP contenant des images**")
        
        uploaded_zip = st.file_uploader(
            "Sélectionnez un fichier ZIP",
            type=['zip'],
            help="ZIP contenant des images organisées par dossiers/classes"
        )
        
        if uploaded_zip is not None:
            if st.button("📦 Extraire et analyser"):
                with st.spinner("Extraction du ZIP..."):
                    try:
                        # Sauvegarder temporairement le ZIP
                        temp_zip = Path(tempfile.mkdtemp()) / "upload.zip"
                        with open(temp_zip, "wb") as f:
                            f.write(uploaded_zip.getbuffer())
                        
                        # Extraire
                        zip_loader = ZipDataLoader()
                        extract_dir = zip_loader.extract_zip(str(temp_zip))
                        
                        # Analyser le contenu extrait
                        dataset_type = detect_dataset_type(str(extract_dir))
                        
                        if dataset_type == "classification_directory":
                            loader = create_data_loader("classification")
                            dataset_info = loader.load_from_directory(str(extract_dir))
                            
                            # Sauvegarder dans la session
                            st.session_state.dataset_info = dataset_info
                            st.session_state.data_source = "zip"
                            st.session_state.data_path = str(extract_dir)
                            
                            st.success("ZIP extrait et dataset chargé !")
                            st.rerun()
                        else:
                            st.error("Le ZIP ne contient pas une structure de classification valide")
                            
                    except Exception as e:
                        st.error(f"Erreur lors de l'extraction : {e}")
    
    with tab3:
        st.markdown("**Régression : CSV avec chemins et valeurs cibles**")
        st.markdown("""
        Format CSV attendu :
        ```csv
        image_path,target
        images/img1.jpg,23.5
        images/img2.jpg,45.2
        ```
        """)
        
        uploaded_csv = st.file_uploader(
            "Sélectionnez un fichier CSV",
            type=['csv'],
            help="CSV avec colonnes 'image_path' et 'target'"
        )
        
        if uploaded_csv is not None:
            # Options pour le CSV
            col1, col2 = st.columns(2)
            with col1:
                image_col = st.text_input("Nom colonne images", value="image_path")
            with col2:
                target_col = st.text_input("Nom colonne target", value="target")
            
            base_path = st.text_input(
                "Chemin de base pour les images (optionnel)",
                placeholder="/path/to/images/folder",
                help="Si les chemins dans le CSV sont relatifs"
            )
            
            if st.button("📊 Charger les données de régression"):
                with st.spinner("Analyse du CSV..."):
                    try:
                        # Sauvegarder temporairement le CSV
                        temp_csv = Path(tempfile.mkdtemp()) / "regression.csv"
                        with open(temp_csv, "wb") as f:
                            f.write(uploaded_csv.getbuffer())
                        
                        # Charger avec le loader de régression
                        loader = create_data_loader("regression")
                        dataset_info = loader.load_from_csv(
                            str(temp_csv),
                            image_col=image_col,
                            target_col=target_col,
                            base_path=base_path if base_path else None
                        )
                        
                        # Sauvegarder dans la session
                        st.session_state.dataset_info = dataset_info
                        st.session_state.data_source = "csv"
                        st.session_state.data_path = str(temp_csv)
                        
                        st.success("Données de régression chargées !")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Erreur lors du chargement : {e}")


def display_dataset_info():
    """Affiche les informations du dataset chargé."""
    if 'dataset_info' not in st.session_state:
        return
    
    dataset_info = st.session_state.dataset_info
    
    st.subheader("📊 Informations du dataset")
    
    # Informations générales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nom", dataset_info.name)
    with col2:
        st.metric("Type", dataset_info.task_type.title())
    with col3:
        st.metric("Échantillons", dataset_info.num_samples)
    with col4:
        if dataset_info.task_type == "classification":
            st.metric("Classes", dataset_info.num_classes)
        else:
            st.metric("Plage", f"{dataset_info.target_range[0]:.2f} - {dataset_info.target_range[1]:.2f}")
    
    # Détails selon le type
    if dataset_info.task_type == "classification":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Classes détectées :**")
            for class_name in dataset_info.class_names:
                count = dataset_info.class_distribution.get(class_name, 0)
                st.write(f"• {class_name}: {count} images")
        
        with col2:
            st.markdown("**Distribution des classes :**")
            # Graphique simple avec Streamlit
            import pandas as pd
            df = pd.DataFrame({
                'Classe': list(dataset_info.class_distribution.keys()),
                'Nombre': list(dataset_info.class_distribution.values())
            })
            st.bar_chart(df.set_index('Classe'))
    
    elif dataset_info.task_type == "regression":
        st.markdown("**Statistiques des targets :**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Valeur minimale", f"{dataset_info.target_range[0]:.3f}")
        with col2:
            st.metric("Valeur maximale", f"{dataset_info.target_range[1]:.3f}")


def data_splitting_section():
    """Section pour configurer les splits de données."""
    if 'dataset_info' not in st.session_state:
        return
    
    dataset_info = st.session_state.dataset_info
    
    st.subheader("📂 Configuration des splits")
    
    # Vérifier si déjà splitté
    if hasattr(dataset_info, 'splits') and dataset_info.splits:
        st.info("Dataset déjà organisé en splits train/val/test")
        
        # Afficher les splits existants
        for split_name, count in dataset_info.splits.items():
            st.write(f"• **{split_name.title()}**: {count} échantillons")
        
        if st.button("✅ Confirmer l'utilisation de ces splits"):
            st.session_state.splits_configured = True
            st.success("Splits confirmés ! Vous pouvez passer à l'entraînement.")
    
    else:
        st.info("Dataset non splitté - Configuration des proportions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            train_split = st.slider("Training (%)", 60, 90, 70)
        with col2:
            val_split = st.slider("Validation (%)", 5, 25, 20)
        with col3:
            test_split = 100 - train_split - val_split
            st.metric("Test (%)", test_split)
        
        # Vérifier que la somme fait 100%
        if train_split + val_split + test_split != 100:
            st.error("La somme des splits doit faire 100%")
            return
        
        # Options avancées
        with st.expander("⚙️ Options avancées"):
            random_seed = st.number_input("Seed aléatoire", value=42, help="Pour la reproductibilité")
            stratify = st.checkbox("Split stratifié", value=True, 
                                 help="Maintenir les proportions de classes (classification uniquement)")
        
        if st.button("🔀 Créer les splits"):
            with st.spinner("Création des splits..."):
                try:
                    if dataset_info.task_type == "classification":
                        loader = create_data_loader("classification")
                        splits = loader.create_train_val_test_split(
                            dataset_info,
                            val_split=val_split / 100,
                            test_split=test_split / 100,
                            random_state=random_seed
                        )
                        
                        # Sauvegarder les splits
                        st.session_state.data_splits = splits
                        st.session_state.splits_configured = True
                        
                        st.success("Splits créés avec succès !")
                        
                        # Afficher les statistiques
                        for split_name, split_data in splits.items():
                            total_samples = sum(len(images) for images in split_data.values())
                            st.write(f"• **{split_name.title()}**: {total_samples} échantillons")
                    
                    elif dataset_info.task_type == "regression":
                        # Pour la régression, on utilise directement pandas
                        import pandas as pd
                        
                        loader = create_data_loader("regression")
                        train_df, val_df, test_df = loader.create_train_val_test_split(
                            st.session_state.data_path,
                            val_split=val_split / 100,
                            test_split=test_split / 100,
                            random_state=random_seed
                        )
                        
                        st.session_state.data_splits = {
                            'train': train_df,
                            'val': val_df,
                            'test': test_df
                        }
                        st.session_state.splits_configured = True
                        
                        st.success("Splits créés avec succès !")
                        st.write(f"• **Train**: {len(train_df)} échantillons")
                        st.write(f"• **Validation**: {len(val_df)} échantillons")
                        st.write(f"• **Test**: {len(test_df)} échantillons")
                
                except Exception as e:
                    st.error(f"Erreur lors de la création des splits : {e}")


def quick_actions():
    """Actions rapides pour utiliser les données d'exemple."""
    if st.session_state.get('demo_mode'):
        st.info("🎮 Mode démo activé - Chargement des données d'exemple")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📁 Charger démo classification"):
                # Créer et charger les données d'exemple
                try:
                    output_dir = Path("samples") / "demo_classification"
                    dataset_path = create_sample_dataset(str(output_dir), num_classes=3, samples_per_class=15)
                    
                    loader = create_data_loader("classification")
                    dataset_info = loader.load_from_directory(str(dataset_path))
                    
                    st.session_state.dataset_info = dataset_info
                    st.session_state.data_source = "demo"
                    st.session_state.data_path = str(dataset_path)
                    
                    st.success("Démo classification chargée !")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Erreur : {e}")
        
        with col2:
            if st.button("📈 Charger démo régression"):
                try:
                    output_dir = Path("samples") / "demo_regression"
                    images_dir, csv_path = create_sample_regression_dataset(str(output_dir), num_samples=60)
                    
                    loader = create_data_loader("regression")
                    dataset_info = loader.load_from_csv(str(csv_path))
                    
                    st.session_state.dataset_info = dataset_info
                    st.session_state.data_source = "demo"
                    st.session_state.data_path = str(csv_path)
                    
                    st.success("Démo régression chargée !")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Erreur : {e}")


def display_image_samples():
    """Affiche un échantillon d'images du dataset."""
    if 'dataset_info' not in st.session_state:
        return
    
    dataset_info = st.session_state.dataset_info
    
    # Vérifier si c'est un dataset d'images
    if dataset_info.task_type not in ["classification", "regression"]:
        return
    
    st.subheader("🖼️ Échantillon d'images")
    
    # Configuration pour l'affichage
    cols_per_row = st.slider("Images par ligne", min_value=2, max_value=6, value=4)
    max_images_per_class = st.slider("Max images par classe", min_value=2, max_value=10, value=4)
    
    try:
        import os
        from PIL import Image
        import random
        
        if dataset_info.task_type == "classification":
            # Pour la classification, afficher des images par classe
            for class_name in dataset_info.class_names[:5]:  # Max 5 classes pour éviter surcharge
                st.markdown(f"**Classe : {class_name}**")
                
                # Trouver le chemin de la classe
                class_path = dataset_info.path / class_name
                
                if class_path.exists():
                    # Lister les images de cette classe
                    image_files = []
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                        image_files.extend(list(class_path.glob(f"*{ext}")))
                        image_files.extend(list(class_path.glob(f"*{ext.upper()}")))
                    
                    # Sélectionner un échantillon aléatoire
                    if image_files:
                        sample_size = min(max_images_per_class, len(image_files))
                        sample_images = random.sample(image_files, sample_size)
                        
                        # Afficher les images en colonnes
                        cols = st.columns(cols_per_row)
                        for i, img_path in enumerate(sample_images):
                            with cols[i % cols_per_row]:
                                try:
                                    image = Image.open(img_path)
                                    # Redimensionner pour l'affichage
                                    image.thumbnail((200, 200))
                                    st.image(image, caption=img_path.name, use_column_width=True)
                                except Exception as e:
                                    st.error(f"Erreur image {img_path.name}: {e}")
                    else:
                        st.warning(f"Aucune image trouvée dans {class_name}")
                
                st.markdown("---")
        
        elif dataset_info.task_type == "regression":
            # Pour la régression, afficher un échantillon d'images avec leurs valeurs
            st.markdown("**Échantillon d'images avec valeurs de régression**")
            
            # Charger le CSV s'il existe
            csv_path = None
            for file in dataset_info.path.parent.glob("*.csv"):
                csv_path = file
                break
            
            if csv_path and csv_path.exists():
                import pandas as pd
                df = pd.read_csv(csv_path)
                
                # Prendre un échantillon
                sample_size = min(max_images_per_class * 2, len(df))
                sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
                
                # Afficher en colonnes
                cols = st.columns(cols_per_row)
                for i, (_, row) in enumerate(sample_df.iterrows()):
                    with cols[i % cols_per_row]:
                        img_path = Path(row['image_path'])
                        if not img_path.is_absolute():
                            img_path = dataset_info.path.parent / img_path
                        
                        try:
                            if img_path.exists():
                                image = Image.open(img_path)
                                image.thumbnail((200, 200))
                                st.image(image, caption=f"{img_path.name}\nValeur: {row['target']:.2f}", use_column_width=True)
                            else:
                                st.error(f"Image non trouvée: {img_path.name}")
                        except Exception as e:
                            st.error(f"Erreur: {e}")
            else:
                st.warning("Fichier CSV non trouvé pour la régression")
    
    except Exception as e:
        st.error(f"Erreur lors de l'affichage des images: {e}")
        st.info("Assurez-vous que Pillow est installé: `pip install Pillow`")


def main():
    """Fonction principale de la page."""
    setup_page()
    
    st.title("📁 Gestion des données et labelling")
    st.markdown("Importez, analysez et préparez vos données pour l'entraînement")
    
    # Mode démo
    quick_actions()
    
    # Section principale d'import
    upload_data_section()
    
    st.markdown("---")
    
    # Section données d'exemple
    create_sample_data_section()
    
    st.markdown("---")
    
    # Afficher les infos du dataset si chargé
    display_dataset_info()
    
    # Échantillons d'images
    display_image_samples()
    
    # Configuration des splits
    if 'dataset_info' in st.session_state:
        st.markdown("---")
        data_splitting_section()
    
    # Navigation
    if st.session_state.get('splits_configured'):
        st.markdown("---")
        st.success("✅ Données prêtes pour l'entraînement !")
        
        if st.button("🧪 Passer à l'entraînement", type="primary"):
            st.switch_page("pages/2_🧪_Experiment_&_Train.py")


if __name__ == "__main__":
    main()
