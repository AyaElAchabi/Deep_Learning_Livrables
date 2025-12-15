import streamlit as st
import os
from dotenv import load_dotenv
from utils import process_documents, create_vectorstore, get_conversation_chain

def main():
    load_dotenv()
    st.set_page_config(page_title="Document Intelligence Assistant", page_icon="📚")

    # Custom CSS for a unique look
    st.markdown("""
        <style>
        .main {
            background-color: #f5f7fa;
        }
        .stTextInput > div > div > input {
            background-color: #ffffff;
            border: 2px solid #4a90e2;
            border-radius: 10px;
        }
        .stButton > button {
            background-color: #4a90e2;
            color: white;
            border-radius: 20px;
            padding: 10px 25px;
            font-weight: bold;
            border: none;
            transition: all 0.3s;
        }
        .stButton > button:hover {
            background-color: #357abd;
            box-shadow: 0 5px 15px rgba(74, 144, 226, 0.3);
        }
        h1 {
            color: #2c3e50;
            font-family: 'Helvetica Neue', sans-serif;
        }
        h2, h3 {
            color: #34495e;
        }
        .upload-text {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center;'>📚 Document Intelligence Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 18px;'>Explorez vos documents PDF avec l'intelligence artificielle</p>", unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    # Sidebar for configuration and file upload
    with st.sidebar:
        st.markdown("### ⚙️ Paramètres")
        st.markdown("---")
        api_key = st.text_input("🔑 Clé API Cohere", type="password", help="Entrez votre clé API Cohere pour commencer")
        if api_key:
            os.environ["COHERE_API_KEY"] = api_key
        
        st.markdown("### 📄 Vos Fichiers")
        st.markdown("---")
        pdf_docs = st.file_uploader(
            "Déposez vos fichiers PDF ici", accept_multiple_files=True, help="Vous pouvez télécharger plusieurs fichiers PDF")
        
        if st.button("🚀 Analyser les documents"):
            if not api_key and not os.getenv("COHERE_API_KEY"):
                st.error("⚠️ Veuillez fournir une clé API Cohere.")
            elif not pdf_docs:
                st.error("⚠️ Veuillez télécharger au moins un fichier PDF.")
            else:
                with st.spinner("🔄 Analyse en cours..."):
                    # Get PDF text
                    raw_documents = process_documents(pdf_docs)
                    
                    # Create vector store
                    vectorstore = create_vectorstore(raw_documents)
                    
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    st.session_state.processComplete = True
                    st.success("✅ Documents analysés avec succès!")
        
        st.markdown("---")
        st.markdown("<div style='text-align: center; color: #95a5a6; font-size: 12px;'>Powered by Cohere AI</div>", unsafe_allow_html=True)

    # Chat Interface
    if st.session_state.processComplete:
        st.markdown("<div class='upload-text'><h3>💬 Conversation Interactive</h3><p>Posez vos questions sur les documents analysés</p></div>", unsafe_allow_html=True)
        
        # Display chat messages from history on app rerun
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("💭 Quelle est votre question ?"):
            # Display user message in chat message container
            st.chat_message("user", avatar="👤").markdown(prompt)
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.spinner("🤔 Réflexion en cours..."):
                response = st.session_state.conversation({"question": prompt})
                answer = response['answer']
                
                # Display assistant response in chat message container
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(answer)
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
    else:
        st.markdown("""
            <div style='text-align: center; padding: 50px;'>
                <h2 style='color: #7f8c8d;'>👋 Bienvenue !</h2>
                <p style='font-size: 18px; color: #95a5a6;'>
                    Pour commencer, veuillez :<br><br>
                    1️⃣ Entrer votre clé API Cohere dans la barre latérale<br>
                    2️⃣ Télécharger vos documents PDF<br>
                    3️⃣ Cliquer sur "🚀 Analyser les documents"
                </p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
