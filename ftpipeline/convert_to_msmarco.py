import pickle
import random
import json
import os

# Load your dataset
data_path = "../data"
with open(f"{data_path}/docs.pkl", "rb") as f:
    documents = pickle.load(f)

with open(f"{data_path}/queriesTrainWithNonRelevant.pkl", "rb") as f:
    train_queries = pickle.load(f)

with open(f"{data_path}/queriesTestWithNonRelevant.pkl", "rb") as f:
    test_queries = pickle.load(f)

# Debug: Print first few documents to understand structure
print("Debug: First document structure:")
first_doc = documents[0]
print(f"Doc attributes: {dir(first_doc)}")
print(f"Doc no: {first_doc.doc_no}")
print(f"Headline: {first_doc.headline}")
print(f"Text: {first_doc.text[:100] if first_doc.text else None}")

# Convert to MS MARCO format
# 1. Create corpus dictionary (pid -> passage)
corpus = {}
for doc in documents:
    # Combine headline and text if both exist
    text = ""
    if doc.headline:
        text += doc.headline + " "
    if doc.text:
        text += doc.text
    
    # Clean the text: remove tabs and newlines to ensure proper TSV format
    text = text.strip().replace("\t", " ").replace("\n", " ")
    if not text:  # Skip empty documents
        print(f"Warning: Empty text for document {doc.doc_no}")
        continue
        
    corpus[doc.doc_no] = text

# 2. Create train and test queries dictionaries
train_queries_dict = {}
for query in train_queries:
    clean_query = query.query.strip().replace("\t", " ").replace("\n", " ")
    train_queries_dict[query.query_no] = clean_query

test_queries_dict = {}
for query in test_queries:
    clean_query = query.query.strip().replace("\t", " ").replace("\n", " ")
    test_queries_dict[query.query_no] = clean_query

# 3. Create training data structure
train_data = {}
for query in train_queries:
    qid = query.query_no
    if not query.relevant_docs:  # Skip queries without relevant documents
        continue
        
    # Get positive passage IDs
    pos_pids = query.relevant_docs
    
    # Get negative passages from non_relevant_docs
    if query.non_relevant_docs:
        # Use all non-relevant documents
        neg_pids = query.non_relevant_docs
    else:
        # Fallback to random sampling if no non-relevant docs available
        print(f"Warning: No non-relevant documents found for query {qid}, using random sampling.")
        all_doc_ids = set(corpus.keys())
        available_negs = list(all_doc_ids - set(pos_pids))
        # Use a reasonable number of random negatives as fallback
        num_negs = min(len(available_negs), 100)
        neg_pids = random.sample(available_negs, num_negs)
    
    train_data[qid] = {
        'qid': qid,
        'query': train_queries_dict[qid],
        'pos': pos_pids,
        'neg': neg_pids
    }

# Save in MS MARCO format
output_dir = 'msmarco-data'
os.makedirs(output_dir, exist_ok=True)

# Save corpus
print("\nSaving collection.tsv...")
with open(os.path.join(output_dir, 'collection.tsv'), 'w', encoding='utf8') as f:
    for pid, passage in corpus.items():
        if not isinstance(pid, str):
            pid = str(pid)
        f.write(f"{pid}\t{passage}\n")

# Save train queries
print("Saving queries.train.tsv...")
with open(os.path.join(output_dir, 'queries.train.tsv'), 'w', encoding='utf8') as f:
    for qid, query in train_queries_dict.items():
        f.write(f"{qid}\t{query}\n")

# Save test queries
print("Saving queries.test.tsv...")
with open(os.path.join(output_dir, 'queries.test.tsv'), 'w', encoding='utf8') as f:
    for qid, query in test_queries_dict.items():
        f.write(f"{qid}\t{query}\n")

# Save training data with hard negatives
print("Saving msmarco-hard-negatives.jsonl...")
with open(os.path.join(output_dir, 'msmarco-hard-negatives.jsonl'), 'w', encoding='utf8') as f:
    for qid, data in train_data.items():
        entry = {
            'qid': qid,
            'pos': [{'pid': pid, 'ce-score': 1.0} for pid in data['pos']],
            'neg': {
                'custom': [{'pid': pid, 'ce-score': 0.0} for pid in data['neg']]
            }
        }
        f.write(json.dumps(entry) + '\n')

# Save test set ground truth (qrels format)
print("Saving test.qrels...")
with open(os.path.join(output_dir, 'test.qrels'), 'w', encoding='utf8') as f:
    for query in test_queries:
        qid = query.query_no
        # Save relevant documents
        for doc_id in query.relevant_docs:
            f.write(f"{qid}\t0\t{doc_id}\t1\n")
        # Save non-relevant documents
        if query.non_relevant_docs:
            for doc_id in query.non_relevant_docs:
                f.write(f"{qid}\t0\t{doc_id}\t0\n")

print(f"\nConversion complete. Files saved in {output_dir}/")
print(f"Total documents: {len(corpus)}")
print(f"Total train queries: {len(train_queries_dict)}")
print(f"Total test queries: {len(test_queries_dict)}")
print(f"Total training examples: {len(train_data)}")

# Print some statistics about test set
test_rel_counts = [len(q.relevant_docs) if q.relevant_docs else 0 for q in test_queries]
test_nonrel_counts = [len(q.non_relevant_docs) if q.non_relevant_docs else 0 for q in test_queries]

print("\nTest Set Statistics:")
print(f"Average relevant docs per query: {sum(test_rel_counts)/len(test_rel_counts):.2f}")
print(f"Average non-relevant docs per query: {sum(test_nonrel_counts)/len(test_nonrel_counts):.2f}")
print(f"Min relevant docs: {min(test_rel_counts)}")
print(f"Max relevant docs: {max(test_rel_counts)}") 