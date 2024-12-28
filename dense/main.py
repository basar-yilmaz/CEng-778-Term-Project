import pickle
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


def save_data(docs, doc_ids, queries):
    with open(f"{data_path}docs.pkl", "wb") as f:
        pickle.dump(docs, f)
    with open(f"{data_path}queries.pkl", "wb") as f:
        pickle.dump(queries, f)


if __name__ == "__main__":
    docs, doc_ids, queries = parsing_phase()

    # filter_relevance_file(relevance_path, doc_ids) this creates qrels with existing doc_ids

    save_data(docs, doc_ids, queries)
