# %%
import pickle

# Load document embeddings
with open("doc_embeddings.pkl", "rb") as f:
    doc_data = pickle.load(f)
doc_embeddings = doc_data["embeddings"]
doc_ids = doc_data["doc_ids"]

# Load query embeddings
with open("query_embeddings.pkl", "rb") as f:
    query_data = pickle.load(f)
query_embeddings = query_data["embeddings"]
query_ids = query_data["query_ids"]

print("Document Embeddings Shape:", doc_embeddings.shape)
print("Query Embeddings Shape:", query_embeddings.shape)

# Load queries
with open("../data/queries.pkl", "rb") as f:
    queries = pickle.load(f)

print("Number of Queries:", len(queries))

# %%
# preprocess queries
queries_filtered = [query for query in queries if query.number_of_relevant_docs > 0]

print("Number of Queries with Relevant Documents:", len(queries_filtered))

# %%
import faiss

embedding_dim = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)

faiss.normalize_L2(doc_embeddings)  # normalize before adding to index
faiss.normalize_L2(query_embeddings)  # normalize before searching

index.add(doc_embeddings)

print(f"FAISS index contains {index.ntotal} embeddings.")


# %%
# Number of nearest neighbors to retrieve
top_k = 1000

# Search the index with normalized query embeddings
distances, indices = index.search(query_embeddings, top_k)

print("Distances Shape:", distances.shape)  # (num_queries, top_k)
print("Indices Shape:", indices.shape)  # (num_queries, top_k)

# %%
# Map indices to document IDs and reverse the order by similarity
results_map = [
    {
        "query_id": query_ids[i],
        "top_docs": sorted(
            [
                {"doc_id": doc_ids[idx], "similarity": distances[i][j]}
                for j, idx in enumerate(indices[i])
            ],
            key=lambda x: x["similarity"],
            reverse=True,
        ),
    }
    for i in range(len(query_ids))
]

# # Print results for each query
# for result in results:
#     print(f"Query ID: {result['query_id']}")
#     for doc in result["top_docs"]:
#         print(f"  Doc ID: {doc['doc_id']}\t Similarity: {doc['similarity']:.6f}")


# %%
# Evaluation
import pytrec_eval

# Create qrels from queries
qrels = {
    query.query_no: {
        doc_id: 1 for doc_id in query.relevant_docs
    }  # Relevance score = 1 for relevant documents
    for query in queries_filtered
}


# print(qrels)

# %%
run = {
    query_ids[i]: {
        doc_ids[idx]: float(distances[i][j]) for j, idx in enumerate(indices[i])
    }
    for i in range(len(query_ids))
}

# %%
# Define evaluation metrics
metrics = {
    "map": "Mean Average Precision",
    "ndcg_cut_10": "NDCG@10",
    "ndcg_cut_20": "NDCG@20",
    "P_5": "Precision@5",
    "P_10": "Precision@10",
    "P_20": "Precision@20",
    "P_100": "Precision@100",
    "recall_100": "Recall@100",
    "recall_1000": "Recall@1000",
    "recip_rank": "Reciprocal Rank",
    "iprec_at_recall_0.00": "Interpolated Precision at 0.00 Recall",
    "iprec_at_recall_0.10": "Interpolated Precision at 0.10 Recall",
    "iprec_at_recall_0.20": "Interpolated Precision at 0.20 Recall",
    "iprec_at_recall_0.30": "Interpolated Precision at 0.30 Recall",
    "iprec_at_recall_0.40": "Interpolated Precision at 0.40 Recall",
    "iprec_at_recall_0.50": "Interpolated Precision at 0.50 Recall",
    "iprec_at_recall_0.60": "Interpolated Precision at 0.60 Recall",
    "iprec_at_recall_0.70": "Interpolated Precision at 0.70 Recall",
    "iprec_at_recall_0.80": "Interpolated Precision at 0.80 Recall",
    "iprec_at_recall_0.90": "Interpolated Precision at 0.90 Recall",
    "iprec_at_recall_1.00": "Interpolated Precision at 1.00 Recall",
    "Rprec": "R-Precision",
    "bpref": "Binary Preference",
}

# Initialize evaluator
evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics.keys())

# Compute metrics
results = evaluator.evaluate(run)

mean_metrics = {}

for metric in results[next(iter(results))].keys():  # Get metrics from the first query
    mean_metrics[metric] = sum(
        query_metrics[metric] for query_metrics in results.values()
    ) / len(results)

for metric, value in mean_metrics.items():
    print(f"{metric}: {value:.4f}")


# %%
# Display results
for query_id, query_metrics in results.items():
    print(f"Query ID: {query_id}")
    for metric, value in query_metrics.items():
        print(f"  {metrics[metric]}: {value:.4f}")

# %%
