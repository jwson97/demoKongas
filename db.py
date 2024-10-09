import pickle
from pathlib import Path
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

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


####################################### semantic
# Semantic 문서 분할
semantic_text_splitter = SemanticChunker(OpenAIEmbeddings())
semantic_document_chunks = semantic_text_splitter.split_documents(docs)

# Semantic Vector DB 생성 및 저장

print("Vector DB 생성 및 저장")
semantic_persist_directory = "./semantic_chroma_db"
semantic_vector_db = Chroma.from_documents(
    documents=semantic_document_chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory=semantic_persist_directory
)
semantic_vector_db.persist()

print("BM25Retriever 생성 및 저장")
# BM25Retriever 생성 및 저장
semantic_bm25_retriever = BM25Retriever.from_documents(semantic_document_chunks)
semantic_bm25_retriever.k = 5

# document_chunks와 bm25_retriever 저장
with open('semantic_document_chunks.pkl', 'wb') as f:
    pickle.dump(semantic_document_chunks, f)

with open('semantic_bm25_retriever.pkl', 'wb') as f:
    pickle.dump(semantic_bm25_retriever, f)

print("Data preparation completed. Vector DB, semantic document chunks, and semantic BM25Retriever have been saved.")


####################################### semantic
# RecursiveCharacterTextSplitter를 사용한 문서 분할
recursive_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
recursive_document_chunks = recursive_text_splitter.split_documents(docs)

print("Vector DB 생성 및 저장")
recursive_persist_directory = "./recursive_chroma_db"
recursive_vector_db = Chroma.from_documents(
    documents=recursive_document_chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory=recursive_persist_directory
)
recursive_vector_db.persist()

# BM25Retriever 생성 및 저장
recursive_bm25_retriever = BM25Retriever.from_documents(recursive_document_chunks)
recursive_bm25_retriever.k = 3


with open('recursive_document_chunks.pkl', 'wb') as f:
    pickle.dump(recursive_document_chunks, f)

with open('recursive_bm25_retriever.pkl', 'wb') as f:
    pickle.dump(recursive_bm25_retriever, f)



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