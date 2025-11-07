
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

def load_vector_store(db_name, db_dir='./db', embedding_model='sentence-transformers/all-mpnet-base-v2'):

    os.makedirs(db_dir, exist_ok=True)
    embedder = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = Chroma(
        collection_name = db_name,
        embedding_function = embedder,
        persist_directory = db_dir
    )
    return vector_store

if __name__ == "__main__":
    db_name = 'biblio'
    vector_store = load_vector_store(db_name)