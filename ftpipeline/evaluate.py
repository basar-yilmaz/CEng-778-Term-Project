import os
import json
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pytrec_eval
import time
from datetime import datetime, timedelta

# FAISS import
try:
    import faiss
    print("FAISS GPU version loaded successfully!")
except ImportError:
    try:
        import faiss.contrib.torch_utils
        print("FAISS CPU version loaded successfully!")
    except ImportError:
        raise ImportError("FAISS could not be loaded. Please run 'pip install faiss-gpu' or 'pip install faiss-cpu'")

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

# Configuration
model_path = '/kaggle/working/train_bi-encoder-margin_mse_en-custom_bert_dot_v5-sentence-transformers-msmarco-bert-base-dot-v5-batch_size_8-2025-01-15_15-47-06'  # Trained model path
data_folder = 'msmarco-data'  # Changed to msmarco-data folder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
top_k = 1000  # Sufficient results for Recall@1000

print(f"Device: {device}")

# Loading model
print("Loading model...")
model = SentenceTransformer(model_path)
model.to(device)
model.eval()

# Loading documents
print("\nLoading documents...")
corpus = {}
error_lines = []
total_lines = 0
valid_lines = 0

with open(os.path.join(data_folder, 'collection.tsv'), 'r', encoding='utf8') as f:
    for line_num, line in enumerate(tqdm(f, desc="Reading documents"), 1):
        total_lines += 1
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        parts = line.split('\t')
        if len(parts) != 2:
            error_lines.append(f"Line {line_num}: {line[:100]}...")  # Show first 100 characters
            continue
            
        pid, passage = parts
        if pid and passage:  # If both fields are non-empty
            corpus[pid] = passage
            valid_lines += 1

print(f"\nTotal lines: {total_lines}")
print(f"Valid document count: {valid_lines}")
if error_lines:
    print(f"\nError line count: {len(error_lines)}")
    print("First 5 error line examples:")
    for err in error_lines[:5]:
        print(err)

if not corpus:
    raise ValueError("No valid documents loaded! Please check file format.")

# Loading test queries instead of train queries
print("\nLoading test queries...")
queries = {}
error_lines = []
total_lines = 0
valid_lines = 0

with open(os.path.join(data_folder, 'queries.test.tsv'), 'r', encoding='utf8') as f:
    for line_num, line in enumerate(tqdm(f, desc="Reading queries"), 1):
        total_lines += 1
        line = line.strip()
        if not line:  # Skip empty lines
            continue
            
        parts = line.split('\t')
        if len(parts) != 2:
            error_lines.append(f"Line {line_num}: {line[:100]}...")
            continue
            
        qid, query = parts
        if qid and query:  # If both fields are non-empty
            queries[qid] = query
            valid_lines += 1

print(f"\nTotal lines: {total_lines}")
print(f"Valid query count: {valid_lines}")
if error_lines:
    print(f"\nError line count: {len(error_lines)}")
    print("First 5 error line examples:")
    for err in error_lines[:5]:
        print(err)

if not queries:
    raise ValueError("No valid queries loaded! Please check file format.")

# Loading ground truth from test.qrels instead of msmarco-hard-negatives.jsonl
print("\nLoading ground truth...")
qrels = {}
with open(os.path.join(data_folder, 'test.qrels'), 'r') as f:
    for line in tqdm(f, desc="Reading ground truth"):
        qid, _, doc_id, relevance = line.strip().split('\t')
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][doc_id] = int(relevance)

# Encoding documents
print("\nEncoding documents...")
doc_ids = list(corpus.keys())
doc_texts = [corpus[did] for did in doc_ids]

doc_embeddings = []
for i in tqdm(range(0, len(doc_texts), batch_size), desc="Document encoding"):
    batch_texts = doc_texts[i:i + batch_size]
    with torch.no_grad():
        embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
        doc_embeddings.append(embeddings.cpu().numpy())
doc_embeddings = np.vstack(doc_embeddings)

# Creating FAISS index
print("\nCreating FAISS index...")
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # For inner product
index.add(doc_embeddings)

# Encoding queries and performing search
print("\nEncoding queries and performing search...")
results = {}
query_ids = list(queries.keys())
query_texts = [queries[qid] for qid in query_ids]

for i in tqdm(range(0, len(query_texts), batch_size), desc="Query search"):
    batch_texts = query_texts[i:i + batch_size]
    batch_ids = query_ids[i:i + batch_size]
    
    with torch.no_grad():
        q_embeddings = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
        scores, indices = index.search(q_embeddings.cpu().numpy(), top_k)
        
        # Save results for each query
        for qid, query_scores, query_indices in zip(batch_ids, scores, indices):
            results[qid] = {doc_ids[idx]: float(score) for idx, score in zip(query_indices, query_scores)}

# Prepare qrels format for evaluation
trec_qrels = {qid: {pid: rel for pid, rel in rels.items()} for qid, rels in qrels.items()}
trec_results = {qid: {pid: score for pid, score in sorted(res.items(), key=lambda x: x[1], reverse=True)} 
                for qid, res in results.items()}

# Define metrics
metrics_dict = {
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

# Evaluation with pytrec_eval
evaluator = pytrec_eval.RelevanceEvaluator(trec_qrels, set(metrics_dict.keys()))
scores = evaluator.evaluate(trec_results)

# Print results
metrics_values = {metric: [] for metric in metrics_dict.keys()}
for qid, query_scores in sorted(scores.items()):
    for metric in metrics_dict.keys():
        if metric in query_scores:
            metrics_values[metric].append(query_scores[metric])

print("\n=== Evaluation Results ===")
print(f"{'Metric':<40} {'Average Score':<10}")
print("="*50)
for metric, values in metrics_values.items():
    if values:  # If there are values for the metric
        mean_score = np.mean(values)
        print(f"{metrics_dict[metric]:<40} {mean_score:.4f}")

# Save detailed results
print("\nSaving detailed results...")
with open('evaluation_results.json', 'w') as f:
    json.dump({
        'per_query_scores': scores,
        'average_scores': {metric: float(np.mean(values)) for metric, values in metrics_values.items() if values},
        'metric_descriptions': metrics_dict
    }, f, indent=2)

# Summary statistics
print("\n=== Summary Statistics ===")
print(f"Total number of queries: {len(queries)}")
print(f"Total number of documents: {len(corpus)}")
print(f"Number of evaluated queries: {len(scores)}")
print(f"Number of results returned per query: {top_k}")

print("\nEvaluation completed! Results saved to 'evaluation_results.json'") 