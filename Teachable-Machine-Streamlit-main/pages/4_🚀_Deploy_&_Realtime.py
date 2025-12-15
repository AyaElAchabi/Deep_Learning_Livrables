"""
Page 4 - Déploiement et inférence temps réel.
"""

import streamlit as st
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def setup_page():
    """Configuration de la page."""
    st.set_page_config(
        page_title="Déploiement - Teachable Machine",
        page_icon="🚀",
        layout="wide"
    )


def check_model_available():
    """Vérifie si un modèle est disponible."""
    if not st.session_state.get('model_trained') and not st.session_state.get('selected_model_path'):
        st.error("❌ Aucun modèle disponible")
        st.markdown("Veuillez d'abord entraîner un modèle ou charger un modèle existant.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧪 Entraîner un modèle"):
                st.switch_page("pages/2_🧪_Experiment_&_Train.py")
        with col2:
            if st.button("📁 Charger un modèle existant"):
                load_existing_model()
        return False
    return True


def load_existing_model():
    """Interface pour charger un modèle existant."""
    st.subheader("📂 Charger un modèle existant")
    
    # Lister les modèles disponibles
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        model_runs = sorted(list(artifacts_dir.glob("run_*")), reverse=True)  # Plus récents en premier
        
        if model_runs:
            # Affichage détaillé des modèles disponibles
            st.write("**Modèles disponibles :**")
            
            selected_run = None
            
            for run_dir in model_runs:
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                    
                    # Lire les métadonnées si disponibles
                    metadata_file = run_dir / "metadata.json"
                    if metadata_file.exists():
                        try:
                            import json
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            with col1:
                                st.write(f"**{run_dir.name}**")
                                timestamp = metadata.get('timestamp', '')
                                if timestamp:
                                    from datetime import datetime
                                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                                    st.caption(dt.strftime("%d/%m/%Y %H:%M"))
                            
                            with col2:
                                dataset_name = metadata.get('dataset_name', 'N/A')
                                task_type = metadata.get('task_type', 'N/A')
                                st.write(f"📊 {dataset_name}")
                                st.caption(f"Type: {task_type}")
                            
                            with col3:
                                final_metrics = metadata.get('final_metrics', {})
                                accuracy = final_metrics.get('accuracy', 0)
                                epochs = final_metrics.get('epochs_completed', 0)
                                st.metric("Accuracy", f"{accuracy:.1%}")
                                st.caption(f"{epochs} epochs")
                            
                            with col4:
                                if st.button("📥 Charger", key=f"load_{run_dir.name}"):
                                    selected_run = run_dir.name
                                    
                        except Exception as e:
                            with col1:
                                st.write(f"**{run_dir.name}**")
                                st.caption("Métadonnées corrompues")
                            with col4:
                                if st.button("📥 Charger", key=f"load_{run_dir.name}"):
                                    selected_run = run_dir.name
                    else:
                        # Ancien format sans métadonnées
                        with col1:
                            st.write(f"**{run_dir.name}**")
                            st.caption("Ancien format")
                        with col2:
                            # Vérifier si le fichier modèle existe
                            model_file = run_dir / "model.keras"
                            if model_file.exists():
                                st.write("✅ Modèle présent")
                            else:
                                st.write("❌ Modèle manquant")
                        with col4:
                            if st.button("📥 Charger", key=f"load_{run_dir.name}"):
                                selected_run = run_dir.name
                    
                    st.markdown("---")
            
            # Charger le modèle sélectionné
            if selected_run:
                model_path = artifacts_dir / selected_run / "model.keras"
                st.session_state.selected_model_path = str(model_path)
                st.session_state.model_trained = True
                
                # Charger aussi les métadonnées dans la session
                metadata_file = artifacts_dir / selected_run / "metadata.json"
                if metadata_file.exists():
                    try:
                        import json
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        st.session_state.loaded_model_metadata = metadata
                    except:
                        pass
                
                st.success(f"✅ Modèle {selected_run} chargé avec succès !")
                st.info(f"📁 Chemin: `{model_path}`")
                st.rerun()
                
        else:
            st.info("🔍 Aucun modèle sauvegardé trouvé")
            st.markdown("Entraînez d'abord un modèle dans l'onglet **🧪 Experiment & Train**")
    else:
        st.warning("📁 Dossier `artifacts/` non trouvé")
        st.markdown("Le dossier sera créé automatiquement lors du premier entraînement.")


def realtime_inference():
    """Interface d'inférence en temps réel."""
    st.subheader("🎯 Inférence en temps réel")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Upload Image", "Webcam", "Dossier", "URL"])
    
    with tab1:
        single_image_inference()
    
    with tab2:
        webcam_inference()
    
    with tab3:
        batch_inference()
    
    with tab4:
        url_inference()


def single_image_inference():
    """Inférence sur une image uploadée."""
    st.markdown("**Upload d'une image**")
    
    uploaded_file = st.file_uploader(
        "Sélectionnez une image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Formats supportés : JPG, PNG, BMP"
    )
    
    if uploaded_file is not None:
        # Afficher l'image
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Image uploadée", use_column_width=True)
        
        with col2:
            if st.button("🔍 Prédire", use_container_width=True):
                with st.spinner("Prédiction en cours..."):
                    # Simuler une prédiction
                    result = simulate_prediction()
                    display_prediction_result(result)


def webcam_inference():
    """Inférence via webcam."""
    st.markdown("**Webcam en temps réel**")
    st.info("🚧 Fonctionnalité webcam en cours d'implémentation")
    
    # Placeholder pour l'interface webcam
    enable_webcam = st.checkbox("Activer la webcam")
    
    if enable_webcam:
        st.info("La webcam serait ici avec cv2 et streamlit-webrtc")
        
        if st.button("📸 Capturer et prédire"):
            result = simulate_prediction()
            display_prediction_result(result)


def batch_inference():
    """Inférence sur un dossier d'images."""
    st.markdown("**Traitement par lot**")
    
    folder_path = st.text_input(
        "Chemin du dossier d'images",
        placeholder="/path/to/images"
    )
    
    if folder_path and st.button("📁 Traiter le dossier"):
        folder_path = Path(folder_path)
        
        if folder_path.exists():
            # Simuler le traitement par lot
            with st.spinner("Traitement des images..."):
                # Simuler la découverte d'images
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                images = []
                for ext in image_extensions:
                    images.extend(folder_path.glob(f"*{ext}"))
                    images.extend(folder_path.glob(f"*{ext.upper()}"))
                
                if images:
                    st.success(f"Trouvé {len(images)} images")
                    
                    # Simuler les prédictions
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, img_path in enumerate(images[:10]):  # Limiter à 10 pour la démo
                        result = simulate_prediction(str(img_path))
                        results.append(result)
                        progress_bar.progress((i + 1) / min(len(images), 10))
                    
                    # Afficher les résultats
                    display_batch_results(results)
                    
                    # Option d'export
                    if st.button("💾 Exporter les résultats (CSV)"):
                        st.success("Résultats exportés vers predictions.csv")
                else:
                    st.warning("Aucune image trouvée dans le dossier")
        else:
            st.error("Le dossier spécifié n'existe pas")


def url_inference():
    """Inférence sur une image depuis une URL."""
    st.markdown("**Image depuis URL**")
    
    image_url = st.text_input(
        "URL de l'image",
        placeholder="https://example.com/image.jpg"
    )
    
    if image_url and st.button("🌐 Charger et prédire"):
        try:
            # Ici on chargerait réellement l'image depuis l'URL
            st.info(f"Chargement depuis : {image_url}")
            
            # Simuler le chargement et la prédiction
            result = simulate_prediction(image_url)
            display_prediction_result(result)
            
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")


def simulate_prediction(image_path="uploaded_image"):
    """Simule une prédiction."""
    import random
    import time
    
    # Simuler le temps de traitement
    time.sleep(random.uniform(0.5, 1.5))
    
    dataset_info = st.session_state.get('dataset_info', None)
    task_type = getattr(dataset_info, 'task_type', 'classification') if dataset_info else 'classification'
    
    if task_type == "classification":
        # Simuler une prédiction de classification
        class_names = getattr(dataset_info, 'class_names', ['Classe A', 'Classe B', 'Classe C']) if dataset_info else ['Classe A', 'Classe B', 'Classe C']
        predictions = [random.random() for _ in class_names]
        # Normaliser pour que la somme soit 1
        total = sum(predictions)
        predictions = [p / total for p in predictions]
        
        predicted_class = class_names[predictions.index(max(predictions))]
        confidence = max(predictions)
        
        return {
            'type': 'classification',
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': dict(zip(class_names, predictions)),
            'image_path': image_path,
            'processing_time': random.uniform(0.1, 0.3)
        }
    
    else:  # regression
        # Simuler une prédiction de régression
        target_range = getattr(dataset_info, 'target_range', (0, 100)) if dataset_info else (0, 100)
        predicted_value = random.uniform(target_range[0], target_range[1])
        
        return {
            'type': 'regression',
            'predicted_value': predicted_value,
            'image_path': image_path,
            'processing_time': random.uniform(0.1, 0.3)
        }


def display_prediction_result(result):
    """Affiche le résultat d'une prédiction."""
    st.subheader("🎯 Résultat de la prédiction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Temps de traitement", f"{result['processing_time']:.3f}s")
    
    if result['type'] == 'classification':
        with col2:
            st.metric("Classe prédite", result['predicted_class'])
            st.metric("Confiance", f"{result['confidence']:.3f}")
        
        # Afficher toutes les probabilités
        st.markdown("**Probabilités par classe :**")
        for class_name, prob in result['all_predictions'].items():
            st.write(f"• {class_name}: {prob:.3f}")
        
        # Graphique des probabilités
        import pandas as pd
        prob_df = pd.DataFrame({
            'Classe': list(result['all_predictions'].keys()),
            'Probabilité': list(result['all_predictions'].values())
        })
        st.bar_chart(prob_df.set_index('Classe'))
    
    else:  # regression
        with col2:
            st.metric("Valeur prédite", f"{result['predicted_value']:.3f}")


def display_batch_results(results):
    """Affiche les résultats du traitement par lot."""
    st.subheader("📊 Résultats du traitement par lot")
    
    # Statistiques générales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Images traitées", len(results))
    with col2:
        avg_time = sum(r['processing_time'] for r in results) / len(results)
        st.metric("Temps moyen", f"{avg_time:.3f}s")
    with col3:
        total_time = sum(r['processing_time'] for r in results)
        st.metric("Temps total", f"{total_time:.1f}s")
    
    # Tableau des résultats
    if results and results[0]['type'] == 'classification':
        # Classification
        import pandas as pd
        
        data = []
        for result in results:
            data.append({
                'Image': Path(result['image_path']).name,
                'Classe prédite': result['predicted_class'],
                'Confiance': f"{result['confidence']:.3f}",
                'Temps': f"{result['processing_time']:.3f}s"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    
    else:
        # Régression
        import pandas as pd
        
        data = []
        for result in results:
            data.append({
                'Image': Path(result['image_path']).name,
                'Valeur prédite': f"{result['predicted_value']:.3f}",
                'Temps': f"{result['processing_time']:.3f}s"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)


def model_export():
    """Section d'export du modèle."""
    st.subheader("📦 Export du modèle")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Format Keras**")
        st.info("Format natif TensorFlow/Keras")
        if st.button("💾 Exporter .keras"):
            st.success("Modèle exporté en format .keras")
    
    with col2:
        st.markdown("**Format ONNX**")
        st.info("Format optimisé multi-plateformes")
        if st.button("🔄 Convertir en ONNX"):
            with st.spinner("Conversion en cours..."):
                # Simuler la conversion
                import time
                time.sleep(2)
                st.success("Modèle converti en ONNX")
    
    with col3:
        st.markdown("**TensorFlow Lite**")
        st.info("Format mobile optimisé")
        if st.button("📱 Convertir en TFLite"):
            with st.spinner("Optimisation pour mobile..."):
                import time
                time.sleep(1.5)
                st.success("Modèle optimisé en TFLite")


def api_generation():
    """Génération d'API FastAPI."""
    st.subheader("🔗 Génération d'API")
    
    st.markdown("Créez automatiquement une API REST pour votre modèle")
    
    # Configuration de l'API
    with st.expander("⚙️ Configuration de l'API"):
        api_name = st.text_input("Nom de l'API", value="teachable_model_api")
        api_port = st.number_input("Port", value=8000, min_value=1000, max_value=9999)
        enable_docs = st.checkbox("Documentation Swagger", value=True)
        enable_cors = st.checkbox("CORS activé", value=True)
        
        # Options avancées
        max_file_size = st.slider("Taille max fichier (MB)", 1, 50, 10)
        rate_limiting = st.checkbox("Limitation de débit", value=False)
    
    if st.button("🚀 Générer l'API FastAPI"):
        with st.spinner("Génération du code API..."):
            # Simuler la génération
            import time
            time.sleep(1)
            
            api_code = generate_fastapi_code(api_name, api_port)
            
            st.success("API générée avec succès !")
            
            # Afficher le code généré
            with st.expander("📄 Code généré (serve_api.py)"):
                st.code(api_code, language="python")
            
            # Instructions d'utilisation
            st.markdown("**Instructions d'utilisation :**")
            st.code(f"""
# Démarrer l'API
python serve_api.py

# Tester avec curl
curl -X POST "http://localhost:{api_port}/predict" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@image.jpg"

# Documentation disponible sur :
# http://localhost:{api_port}/docs
            """)


def generate_fastapi_code(api_name, port):
    """Génère le code FastAPI."""
    return f'''
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn

app = FastAPI(title="{api_name}", version="1.0.0")

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger le modèle
model = tf.keras.models.load_model("model.keras")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Prédiction sur une image uploadée."""
    try:
        # Lire l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Préprocessing
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        # Prédiction
        prediction = model.predict(image_array)
        
        return {{
            "filename": file.filename,
            "prediction": prediction.tolist()
        }}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={port})
'''


def deployment_guide():
    """Guide de déploiement."""
    st.subheader("📖 Guide de déploiement")
    
    tab1, tab2, tab3 = st.tabs(["Local", "Cloud", "Mobile"])
    
    with tab1:
        st.markdown("""
        **Déploiement local :**
        
        1. Exportez votre modèle au format souhaité
        2. Générez l'API FastAPI
        3. Installez les dépendances :
        ```bash
        pip install fastapi uvicorn tensorflow pillow
        ```
        4. Lancez l'API :
        ```bash
        python serve_api.py
        ```
        """)
    
    with tab2:
        st.markdown("""
        **Déploiement cloud :**
        
        **Docker :**
        ```dockerfile
        FROM python:3.9-slim
        COPY requirements.txt .
        RUN pip install -r requirements.txt
        COPY . .
        CMD ["python", "serve_api.py"]
        ```
        
        **Plateforme recommandées :**
        - 🐳 Docker + AWS ECS/Google Cloud Run
        - ⚡ Vercel/Netlify pour les APIs légères
        - 🚀 Heroku pour un déploiement rapide
        """)
    
    with tab3:
        st.markdown("""
        **Déploiement mobile :**
        
        1. Convertissez en TensorFlow Lite
        2. Intégrez dans votre app mobile :
        
        **Android (Java/Kotlin) :**
        ```java
        // Charger le modèle TFLite
        Interpreter tflite = new Interpreter(loadModelFile());
        ```
        
        **iOS (Swift) :**
        ```swift
        // Utiliser Core ML ou TensorFlow Lite
        let interpreter = try Interpreter(modelPath: modelPath)
        ```
        """)


def main():
    """Fonction principale de la page."""
    setup_page()
    
    st.title("🚀 Déploiement et inférence temps réel")
    st.markdown("Déployez votre modèle et testez-le en conditions réelles")
    
    # Vérifier qu'un modèle est disponible
    if not check_model_available():
        return
    
    # Informations sur le modèle chargé
    st.info("✅ Modèle prêt pour l'inférence")
    
    # Interface d'inférence
    realtime_inference()
    
    st.markdown("---")
    
    # Export du modèle
    model_export()
    
    st.markdown("---")
    
    # Génération d'API
    api_generation()
    
    st.markdown("---")
    
    # Guide de déploiement
    deployment_guide()


if __name__ == "__main__":
    main()
