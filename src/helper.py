from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


# Extract text from PDF file 
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob= "*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

# filetring the data as need only source and page content
from typing import List
from langchain.schema import Document

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Documents objects, return a new list of documents objects
    containing only 'source' in metadata and the original page content.
    """
    minimal_docs: List[Document]= []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs



# split the documents into smaller chunks
def text_splitter(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= 500,
        chunk_overlap=20,
    )
    texts_chunks = text_splitter.split_documents(minimal_docs)
    return texts_chunks


# Embeddings 

def download_embeddings():
    """
    Download and retrun the huggingfsce embeddings models.
    """

    model_name= "sentence-transformers/all-MiniLM-L6-v2"
    embeddings= HuggingFaceEmbeddings(
        model_name= model_name
    )
    return embeddings

embedding= download_embeddings()

