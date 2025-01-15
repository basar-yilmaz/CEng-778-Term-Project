import pickle
import random
from parser import (
    filter_relevance_file,
    parse_documents,
    parse_queries,
    parse_relevance,
)

data_path = "../data"
doc_path = f"{data_path}/ft/all"
query_paths = [
    f"{data_path}/query-relJudgments/q-topics-org-SET1.txt",
    f"{data_path}/query-relJudgments/q-topics-org-SET2.txt",
    f"{data_path}/query-relJudgments/q-topics-org-SET3.txt",
]

relevance_path = [
    f"{data_path}/query-relJudgments/qrel_301-350_complete.txt",
    f"{data_path}/query-relJudgments/qrels.trec7.adhoc_350-400.txt",
    f"{data_path}/query-relJudgments/qrels.trec8.adhoc.parts1-5_400-450",
]


def parsing_phase():
    # Read the documents (doc_ids are used to check if a document is relevant)
    docs, doc_ids = parse_documents(doc_path)
    print(f"Total documents: {len(docs)}")

    queries = parse_queries(query_paths)
    print(f"Total queries: {len(queries)}")

    # Read the relevance judgments and add them to the queries
    parse_relevance(
        relevance_path,
        queries,
        doc_ids,
    )

    return docs, doc_ids, queries


def split_queries(queries, test_size=50, min_relevant_docs=5):
    # Filter queries that have at least min_relevant_docs relevant documents
    eligible_queries = []
    for query in queries:
        if query.relevant_docs and len(query.relevant_docs) >= min_relevant_docs:
            eligible_queries.append(query)

    print(f"\nTotal queries: {len(queries)}")
    print(f"Queries with {min_relevant_docs}+ relevant docs: {len(eligible_queries)}")

    # Ensure we have enough eligible queries
    if len(eligible_queries) < test_size:
        raise ValueError(f"Not enough queries with {min_relevant_docs}+ relevant documents. Found only {len(eligible_queries)}, need {test_size}.")

    # Randomly select test_size queries for test set
    random.seed(42)  # For reproducibility
    test_queries = random.sample(eligible_queries, test_size)
    train_queries = [q for q in queries if q not in test_queries]

    # Print statistics
    test_rel_docs_counts = [len(q.relevant_docs) for q in test_queries]
    print(f"\nTrain set size: {len(train_queries)}")
    print(f"Test set size: {len(test_queries)}")
    print("\nTest set statistics:")
    print(f"Min relevant docs: {min(test_rel_docs_counts)}")
    print(f"Max relevant docs: {max(test_rel_docs_counts)}")
    print(f"Avg relevant docs: {sum(test_rel_docs_counts)/len(test_rel_docs_counts):.2f}")

    return train_queries, test_queries


def save_data(docs, queries_train, queries_test):
    # Save documents
    print("\nSaving documents...")
    with open(f"{data_path}/docs.pkl", "wb") as f:
        pickle.dump(docs, f)
    
    # Save train queries
    print("Saving train queries...")
    with open(f"{data_path}/queriesTrainWithNonRelevant.pkl", "wb") as f:
        pickle.dump(queries_train, f)
    
    # Save test queries
    print("Saving test queries...")
    with open(f"{data_path}/queriesTestWithNonRelevant.pkl", "wb") as f:
        pickle.dump(queries_test, f)
    
    print("\nAll data saved successfully!")
    print(f"- Documents: {len(docs)} saved to docs.pkl")
    print(f"- Train queries: {len(queries_train)} saved to queriesTrainWithNonRelevant.pkl")
    print(f"- Test queries: {len(queries_test)} saved to queriesTestWithNonRelevant.pkl")


if __name__ == "__main__":
    # Parse all data
    docs, doc_ids, queries = parsing_phase()

    # Split queries into train and test sets
    queries_train, queries_test = split_queries(queries, test_size=50, min_relevant_docs=5)

    # Save all data
    save_data(docs, queries_train, queries_test)
