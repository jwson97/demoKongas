import pickle
from pathlib import Path
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv

load_dotenv()

print("Document 로드")

# 문서 로드
doc_paths = [
    "docs/가스계통_운영규정.pdf",
    "docs/여비규정.pdf",
    "docs/취업규칙.pdf",
]

docs = []
for doc_file in doc_paths:
    file_path = Path(doc_file)
    try:
        if doc_file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        docs.extend(loader.load())
    except Exception as e:
        print(f"Error loading document {doc_file}: {e}")

print("문서 분할")

# 문서 분할
text_splitter = SemanticChunker(OpenAIEmbeddings())
document_chunks = text_splitter.split_documents(docs)

# Vector DB 생성 및 저장

print("Vector DB 생성 및 저장")
persist_directory = "./chroma_db"
vector_db = Chroma.from_documents(
    documents=document_chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory=persist_directory
)
vector_db.persist()

print("BM25Retriever 생성 및 저장")
# BM25Retriever 생성 및 저장
bm25_retriever = BM25Retriever.from_documents(document_chunks)
bm25_retriever.k = 5

# document_chunks와 bm25_retriever 저장
with open('document_chunks.pkl', 'wb') as f:
    pickle.dump(document_chunks, f)

with open('bm25_retriever.pkl', 'wb') as f:
    pickle.dump(bm25_retriever, f)

print("Data preparation completed. Vector DB, document chunks, and BM25Retriever have been saved.")


pdf_metadata = {}
for doc_file in doc_paths: 
    file_path = Path(doc_file)
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    for page in pages:
        content = page.page_content
        metadata = page.metadata
        metadata['file_path'] = str(file_path)
        
        # Use the content as the key and store metadata
        pdf_metadata[content] = metadata

# Save the pdf_metadata
with open('pdf_metadata.pkl', 'wb') as f:
    pickle.dump(pdf_metadata, f)