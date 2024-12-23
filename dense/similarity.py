import faiss
from parser import doc_embed_parser, query_embed_parser, normalize_embeds
import torch

root_path_data = "../embeds" # Run the code in dense folder
docs_file_path = f"{root_path_data}/doc_embeddings.pkl"
query_file_path = f"{root_path_data}/query_embeddings.pkl"

doc_embeds = doc_embed_parser(docs_file_path)
query_embeds = query_embed_parser(query_file_path)

normalized_doc_embeds = normalize_embeds(doc_embeds)
normalized_query_embeds = normalize_embeds(query_embeds)

dimension = normalized_doc_embeds.shape[1]
index = faiss.IndexFlatIP(dimension)  
index.add(normalized_doc_embeds)  

k = 10

query = normalized_query_embeds[0].reshape(1, -1)
similarities, indices = index.search(query, k)

similarities = torch.tensor(similarities)
indices = torch.tensor(indices)

print("Cosine Similarity Matrix (as sorted indices):")
print(indices)


"""
for query_idx, doc_indices in enumerate(indices):
    print(f"Query {query_idx + 1}:")
    for rank, doc_idx in enumerate(doc_indices):
        print(f"  Rank {rank + 1}: Document {doc_idx}, Similarity: {similarities[query_idx][rank]}")
"""