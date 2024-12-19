from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Initialize the embedding model
model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()

def embed_documents(docs, model=model):
    """
    Embeds documents using the specified SentenceTransformer model.
    Args:
        docs (list): List of Document objects.
        model: SentenceTransformer model.
    Returns:
        dict: A dictionary where keys are doc_no and values are embeddings.
    """
    doc_embeddings = {}
    for doc in tqdm(docs, desc="Embedding Documents"):
        if doc.text:  # Ensure there's text to embed
            embedding = model.encode(doc.text, convert_to_tensor=True, device='cuda')
            doc_embeddings[doc.doc_no] = embedding.cpu()  # Move to CPU for storage
    return doc_embeddings

def embed_queries(queries, model=model):
    """
    Embeds queries using the specified SentenceTransformer model.
    Args:
        queries (list): List of Query objects.
        model: SentenceTransformer model.
    Returns:
        dict: A dictionary where keys are query_no and values are embeddings.
    """
    query_embeddings = {}
    for query in tqdm(queries, desc="Embedding Queries"):
        if query.query:  # Ensure there's a query string to embed
            embedding = model.encode(query.query, convert_to_tensor=True, device='cuda')
            query_embeddings[query.query_no] = embedding.cpu()  # Move to CPU for storage
    return query_embeddings