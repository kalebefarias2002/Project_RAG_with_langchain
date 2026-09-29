from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


PASTA_BASE = "base"

def criar_db():
    documentos = carregar_documanetos()
    chunks = dividir_chunks(documentos)
    #vetorizar_chunks(chunks)

def carregar_documanetos():
    carreggar = PyPDFDirectoryLoader(PASTA_BASE, glob="**/*.pdf")
    documentos = carreggar.load()
    return documentos

def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500, length_function=len, add_start_index=True)
    chunks = separador_documentos.split_documents(documentos)
    return chunks
