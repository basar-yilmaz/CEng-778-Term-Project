import pickle

from parser import parse_documents, parse_queries, parse_relevance
from embedding import embed_queries, embed_documents
import os

data_path = "data"  # Put the data folders into the root folder.
doc_path = f"{data_path}/ft/all"
query_path = f"{data_path}/query-relJudgements/q-topics-org-SET1.txt"

def parsing_phase():
    # Read the documents (doc_ids are used to check if a document is relevant)
    docs, doc_ids = parse_documents(doc_path)
    print(f"Total documents: {len(docs)}")

    # Read the queries
    queries = parse_queries(
        [
            f"{data_path}/query-relJudgments/q-topics-org-SET1.txt",
            f"{data_path}/query-relJudgments/q-topics-org-SET2.txt",
            f"{data_path}/query-relJudgments/q-topics-org-SET3.txt",
        ]
    )
    print(f"Total queries: {len(queries)}")

    # print("Example query #1:")
    # print(queries[0])

    # Read the relevance judgments and add them to the queries
    parse_relevance(
        f"{data_path}/query-relJudgments/qrel_301-350_complete.txt",
        queries,
        doc_ids,
    )

    return docs, queries

def embedding_phase(docs, queries):
    # Embed the queries
    query_embeddings = embed_queries(queries)

    # Embed the documents
    doc_embeddings = embed_documents(docs)

    return doc_embeddings, query_embeddings

# Read the documents
if __name__ == "__main__":
   docs, queries = parsing_phase()

   doc_embs, query_embs = embedding_phase(docs, queries)

   # Save embeddings to files
   with open("doc_embeddings.pkl", "wb") as f:
       pickle.dump(doc_embs, f)

   with open("query_embeddings.pkl", "wb") as f:
       pickle.dump(query_embs, f)

   # Example document embedding
   doc_id = list(doc_embs.keys())[0]
   print(f"Embedding for doc_id {doc_id}: {doc_embs[doc_id]}")

   # Example query embedding
   query_id = list(query_embs.keys())[0]
   print(f"Embedding for query_id {query_id}: {query_embs[query_id]}")
