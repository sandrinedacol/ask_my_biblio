import glob
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from database_loader import load_vector_store


def get_files_path(filenames, pdf_dir='./pdf'):
    if filenames == None:
        files_path = glob.glob(f'{pdf_dir}/*.pdf')
    else:
        if type(filenames) == list:
            files_path = [f'{pdf_dir}/{filename}' for filename in filenames]
        elif type(filenames) == str:
            files_path = [f'{pdf_dir}/{filenames}']
    if len(files_path) == 1:
        print(f'1 pdf file about to be stored:\n')
    else:
        print(f'{len(files_path)} pdf files about to be stored:\n')
    return files_path


def load_file(file_path):
    print(file_path.split('/')[-1])
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def split_document(doc, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = text_splitter.split_documents(doc)
    print(f'--> {len(chunks)} chunks\n')
    return chunks


def store_embeddings(chunks, vector_store):
    _ = vector_store.add_documents(documents=chunks)
    return vector_store


def store_document(filenames=None, db_name='biblio', reset_pkl=True):
    vector_store = load_vector_store(db_name)
    files_path = get_files_path(filenames)
    n_chunks = 0
    for file_path in files_path:
        docs = load_file(file_path)
        chunks = split_document(docs)
        n_chunks += len(chunks)
        vector_store = store_embeddings(chunks, vector_store)
    print(f'{n_chunks} chunks successfully stored!')


if __name__ == "__main__":
    store_document()
    