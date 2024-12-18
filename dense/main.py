from parser import parse_documents, parse_queries, parse_relevance
import os

data_path = ".." # Put the data folders into the root folder.
doc_path = f"{data_path}/ft/all"
query_path = f"{data_path}/query-relJudgements/q-topics-org-SET1.txt"

# Read the documents
if __name__ == "__main__":
    # stopwords = parse_stopwords("data/ft/all/stopword.lst")
    # print(f"Total stopwords: {len(stopwords)}")

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

    print("Example query #1:")
    print(queries[0])

    # Read the relevance judgments and add them to the queries
    parse_relevance(
        f"{data_path}/query-relJudgments/qrel_301-350_complete.txt",
        queries,
        doc_ids,
    )
    
    with open("queries.txt", "w") as file:
        for query in queries:
            file.write(str(query.query) + "\n")

    with open("docs.txt", "w") as file:
        for doc in docs:
            file.write(str(doc.text) + " " + str(doc.doc_no) + "\n")

    
