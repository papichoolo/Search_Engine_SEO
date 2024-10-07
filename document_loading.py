from langchain.schema import Document
from tqdm import tqdm

def load_documents_from_df(df):
    """Load documents from a DataFrame with necessary fields."""
    documents = []
    
    # Iterate over DataFrame rows with a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Create a Document with page content, meta-data, and unique id
        document = Document(
            page_content=row['Abstract'],  # Text content for embedding
            meta_data={'Title': row['Title'], 'weblink': row['Link-pubmed']},  # Additional meta-data
            id=str(index)  # Unique identifier as string
        )
        
        documents.append(document)
    
    return documents
