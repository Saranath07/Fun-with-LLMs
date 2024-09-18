import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

chroma_client = chromadb.Client(Settings(allow_reset=True))
collection_name = "client_documents"
chroma_client.reset()  # Be careful with this in production!
collection = chroma_client.create_collection(name=collection_name)

# Document processing function
def process_documents(directory: str) -> List[dict]:

    document_types = ['.txt', '.md', '.pdf', '.docx'] 

    loader = DirectoryLoader(
        directory,
        glob="**/*",  
        loader_cls=PyPDFLoader if directory.endswith('.pdf') else TextLoader,  # Use PDFLoader for PDFs
        loader_kwargs={'autodetect_encoding': True},
        use_multithreading=True,
        show_progress=True,
    )
    
    documents = []
    for doc in loader.load():
        if any(doc.metadata['source'].lower().endswith(ext) for ext in document_types):
            documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    return [{"id": str(i), "text": split.page_content, "metadata": split.metadata} for i, split in enumerate(splits)]