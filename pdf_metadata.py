import pickle
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

doc_paths = [
    "docs/가스계통_운영규정.pdf",
    "docs/여비규정.pdf",
    "docs/취업규칙.pdf",
]

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
   