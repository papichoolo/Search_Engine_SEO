from pymongo import MongoClient
# from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from tqdm import tqdm


class VectorStoring:
    def __init__(self, mongo_uri='mongodb://localhost:27017/', mongo_db='embedding_database'):
        # MongoDB setup
        self.mongo_client = MongoClient(mongo_uri)
        self.db = self.mongo_client[mongo_db]
        self.documents_collection = 'documents'
        self.embeddings_collection = 'embeddings'
        
        # ChromaDB setup
        # self.chroma_vector_store = None  # ChromaDB instance will be initialized when storing embeddings

    # 1. MongoDB Functions
    def check_and_create_collection(self, collection_name):
        """Check if a collection exists in MongoDB; create it if not."""
        if collection_name not in self.db.list_collection_names():
            self.db.create_collection(collection_name)
            print(f"Collection '{collection_name}' created in MongoDB.")
        else:
            print(f"Collection '{collection_name}' already exists in MongoDB.")

    def store_documents_in_mongodb(self, documents):
        """Store documents with metadata in MongoDB collection."""
        self.check_and_create_collection(self.documents_collection)
        collection = self.db[self.documents_collection]

        # Insert documents into MongoDB
        for doc in tqdm(documents, total=len(documents)):
            document_data = {
                "document": doc.page_content,
                "metadata": doc.metadata,
                "id": doc.id
            }
            collection.insert_one(document_data)
        
        print(f"Stored {len(documents)} documents in MongoDB collection '{self.documents_collection}'.")

    def store_embeddings_in_mongodb(self, embeddings):
        """Store document embeddings in MongoDB collection."""
        self.check_and_create_collection(self.embeddings_collection)
        collection = self.db[self.embeddings_collection]

        # Insert embeddings into MongoDB
        for doc_id, embedding in tqdm(embeddings, total=len(embeddings)):
            embedding_data = {
                "document_id": doc_id,
                "embedding": embedding
            }
            collection.insert_one(embedding_data)
        
        print(f"Stored {len(embeddings)} embeddings in MongoDB collection '{self.embeddings_collection}'.")

    # 2. ChromaDB Functions
    # def store_embeddings_in_chromadb(self, documents):
    #     """Generate embeddings and store them in ChromaDB."""
    #     # Initialize embedding model
    #     embedding_model = OllamaEmbeddings(model='nomic-embed-text')
        
    #     # Store document embeddings in ChromaDB
    #     self.chroma_vector_store = Chroma.from_documents(
    #         documents=documents,
    #         collection_name="chroma-collection",
    #         embedding=embedding_model
    #     )
    #     print(f"Embeddings stored in ChromaDB collection 'chroma-collection'.")

    # Full process flow for storing both documents and embeddings
    def store_all(self, documents, embeddings):
        """Store both documents and embeddings in MongoDB and ChromaDB."""
        # 1. Store documents in MongoDB
        self.store_documents_in_mongodb(documents)
        
        # 2. Store embeddings in MongoDB
        self.store_embeddings_in_mongodb(embeddings)
        
        # 3. Store embeddings in ChromaDB
        # self.store_embeddings_in_chromadb(documents)


# Example usage:
if __name__ == "__main__":
    # Assuming you have `documents` and `embeddings` prepared from previous steps
    vector_storing = VectorStoring()

    # Sample documents and embeddings
    documents = [...]  # A list of Document objects
    embeddings = [...]  # A list of (document_id, embedding) tuples

    # Store documents and embeddings in MongoDB and ChromaDB
    vector_storing.store_all(documents, embeddings)
