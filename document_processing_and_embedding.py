from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm
from langchain_community.vectorstores import Chroma

def split_documents(documents, chunk_size=7500, chunk_overlap=100):
    """
    Splits documents into smaller chunks for embedding if necessary.
    
    Parameters:
    - documents: List of Document objects to be split.
    - chunk_size: Maximum size of each document chunk.
    - chunk_overlap: Number of characters that overlap between chunks.
    
    Returns:
    - List of split document chunks.
    """
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    
    # Split the documents into chunks
    doc_chunks = text_splitter.split_documents(documents)
    
    print(f"Documents split into {len(doc_chunks)} chunks.")
    return doc_chunks


def make_document_embeddings(documents):
    """
    Generates vector embeddings for each document.
    
    Parameters:
    - documents: List of Document objects (or document chunks) to be embedded.
    
    Returns:
    - List of tuples containing document id and its corresponding embedding.
    """
    # Initialize the embedding model
    embedding_model = OpenAIEmbeddings()
    
    vectorstore = Chroma.from_documents(
    documents=documents,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
    retriever = vectorstore.as_retriever()