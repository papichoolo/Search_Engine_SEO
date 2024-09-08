import pandas as pd
from duckduckgo_search import DDGS
from tqdm import tqdm
from langchain.schema import Document
# from search_service import get_urls
from document_loading import load_documents_from_df
from search_querier import search_query
from Storings import VectorStoring
from document_processing_and_embedding import split_documents, make_document_embeddings


# Assuming the previously defined functions are already available

# Step 1: Search Query Function
# def search_query(query):
#     """Fetch search results from DuckDuckGo and return as a DataFrame."""
#     results = DDGS().text(
#         keywords=str(query),
#         max_results=10,
#         region='wt-wt',
#         timelimit='7d'
#     )
    
#     # Convert the results to a DataFrame
#     results_df = pd.DataFrame(results)
#     return results_df


# Main Function to process the workflow
def main():
    query = input("Enter the search query: ")

    # Step 1: Fetch Search Results
    print("Fetching search results...")
    search_results_df = search_query(query)
    
    if search_results_df.empty:
        print("No results found. Exiting...")
        return

    # Step 2: Load Documents from Search Results DataFrame
    print("Loading documents...")
    documents = load_documents_from_df(search_results_df)

    # Step 3: Split Documents into Chunks (if necessary)
    print("Splitting documents into chunks...")
    document_chunks = split_documents(documents)

    # Step 4: Generate Embeddings for Each Document Chunk
    print("Generating document embeddings...")
    document_embeddings = make_document_embeddings(document_chunks)

    # Step 5: Store Documents and Embeddings in MongoDB and ChromaDB
    print("Storing documents and embeddings in databases...")
    vector_storing = VectorStoring()  # Initialize the VectorStoring class

    # Store both documents and embeddings in MongoDB and ChromaDB
    vector_storing.store_all(documents=document_chunks, embeddings=document_embeddings)

    print("Process complete. Documents and embeddings have been stored.")


if __name__ == "__main__":
    main()
