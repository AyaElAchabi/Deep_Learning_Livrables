import os
import cohere
import faiss
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

def get_cohere_client():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        return None
    return cohere.Client(api_key)

def process_documents(pdf_docs):
    """
    Load and process uploaded PDF documents.
    Returns a list of text chunks (strings).
    """
    text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            print(f"Error reading PDF {pdf.name}: {e}")
    
    # Simple text splitting
    chunk_size = 1000
    overlap = 200
    
    chunks = []
    start = 0
    if len(text) == 0:
        return []
        
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
        if start >= len(text):
            break
            
    return chunks

def create_vectorstore(chunks):
    """
    Create a vector store from text chunks.
    """
    co = get_cohere_client()
    if not co:
        # If no API key yet, we can't create embeddings. 
        # But app.py checks for API key before calling this.
        return None
        
    if not chunks:
        return None

    # Get embeddings
    batch_size = 90
    embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        try:
            response = co.embed(texts=batch, model="embed-english-v3.0", input_type="search_document")
            embeddings.extend(response.embeddings)
        except Exception as e:
            print(f"Error embedding batch: {e}")
            
    if not embeddings:
        return None
        
    embed_arr = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    d = embed_arr.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embed_arr)
    
    return {"index": index, "chunks": chunks}

class CohereChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.chat_history = []
        self.co = get_cohere_client()

    def __call__(self, inputs):
        if not self.vectorstore:
            return {'answer': "No documents processed."}
            
        question = inputs["question"]
        index = self.vectorstore["index"]
        chunks = self.vectorstore["chunks"]
        
        # Retrieve relevant docs
        response = self.co.embed(texts=[question], model="embed-english-v3.0", input_type="search_query")
        q_embed = np.array(response.embeddings).astype('float32')
        
        k = 3
        D, I = index.search(q_embed, k)
        
        relevant_docs = []
        for i in I[0]:
            if i < len(chunks):
                relevant_docs.append({"text": chunks[i]})
        
        # Generate answer using Cohere Chat
        response = self.co.chat(
            message=question,
            documents=relevant_docs,
            chat_history=self.chat_history,
            model="command-a-03-2025"
        )
        
        answer = response.text
        
        # Update history
        self.chat_history.append({"role": "USER", "message": question})
        self.chat_history.append({"role": "CHATBOT", "message": answer})
        
        return {'answer': answer}

def get_conversation_chain(vectorstore):
    return CohereChain(vectorstore)
