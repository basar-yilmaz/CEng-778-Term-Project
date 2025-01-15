import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import os
from collections import defaultdict
from torch.utils.data import IterableDataset
import tqdm
from torch.utils.data import Dataset
import random
import torch
import transformers

# Disable Wandb
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*50}")
print(f"Device used: {device}")
if torch.cuda.is_available():
    print(f"GPU model: {torch.cuda.get_device_name(0)}")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    # Clear GPU memory
    torch.cuda.empty_cache()

# Fixed Parameters
train_batch_size = 8  # Reduced batch size
max_seq_length = 300  # Maximum sequence length
model_name = 'sentence-transformers/msmarco-bert-base-dot-v5'  # Model name
max_passages = 0  # Maximum passages
epochs = 5  # Number of epochs
pooling = 'mean'  # Pooling type - mean pooling is used for dot product model
negs_to_use = 'custom'  # System for negative examples
warmup_steps = 100  # Reduced warmup steps
lr = 1e-5  # Reduced learning rate
name = 'custom_bert_dot_v5'  # Model name
num_negs_per_system = 5  # Number of negative examples
use_pre_trained_model = True  # Use pre-trained model
use_all_queries = False  # Use all queries

# Kaggle paths
data_folder = '/kaggle/input/msmarcobase1'
model_save_path = f'/kaggle/working/train_bi-encoder-margin_mse_en-{name}-{model_name.replace("/", "-")}-batch_size_{train_batch_size}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

print("\n=== Initial Configuration ===")
print(f"Data folder: {data_folder}")
print(f"Model save path: {model_save_path}")
print(f"Batch size: {train_batch_size}")
print(f"Epochs: {epochs}")
print(f"Learning rate: {lr}")
print(f"Warmup steps: {warmup_steps}")
print("==============================")

# Creating model
if use_pre_trained_model:
    print(f"\nLoading pre-trained SBERT model: {model_name}")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
else:
    print("\nCreating a new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Create directory for saving model
os.makedirs(model_save_path, exist_ok=True)

# Load corpus data
corpus = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')

print("\nReading corpus: collection.tsv")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    # Find total number of lines
    total_lines = sum(1 for _ in fIn)
    fIn.seek(0)  # Return to start of file
    
    for line_num, line in tqdm.tqdm(enumerate(fIn, 1), total=total_lines, desc="Loading corpus"):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            print(f"Warning: Line {line_num} has incorrect format: {line}")
            continue
        pid, passage = parts
        corpus[pid] = passage

print(f"Corpus loaded. Total number of documents: {len(corpus)}")

# Training data: train queries
queries = {}
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')

print("\nReading queries: queries.train.tsv")
with open(queries_filepath, 'r', encoding='utf8') as fIn:
    # Find total number of lines
    total_lines = sum(1 for _ in fIn)
    fIn.seek(0)  # Return to start of file
    
    for line_num, line in tqdm.tqdm(enumerate(fIn, 1), total=total_lines, desc="Loading queries"):
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            print(f"Warning: Line {line_num} has incorrect format: {line}")
            continue
        qid, query = parts
        queries[qid] = query

print(f"Queries loaded. Total number of queries: {len(queries)}")

# Load training data
train_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl')

train_queries = {}
ce_scores = {}

print("\nLoading training data...")
with open(train_filepath, 'rt') as fIn:
    # Find total number of lines
    total_lines = sum(1 for _ in fIn)
    fIn.seek(0)  # Return to start of file
    
    for line in tqdm.tqdm(fIn, total=total_lines, desc="Loading training data"):
        if max_passages > 0 and len(train_queries) >= max_passages:
            break
            
        data = json.loads(line)
        
        if data['qid'] not in ce_scores:
            ce_scores[data['qid']] = {}
        
        # Positive ce_scores
        for item in data['pos']:
            ce_scores[data['qid']][item['pid']] = item['ce-score']

        # Get positive passage IDs
        pos_pids = [item['pid'] for item in data['pos']]
       
        # Get negative passages
        neg_pids = set()
        if negs_to_use not in data['neg']:
            continue
                
        system_negs = data['neg'][negs_to_use]
        
        negs_added = 0
        for item in system_negs:
            ce_scores[data['qid']][item['pid']] = item['ce-score']
            
            pid = item['pid']
            if pid not in neg_pids:
                neg_pids.add(pid)
                negs_added += 1
                if negs_added >= num_negs_per_system:
                    break

        if use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
            train_queries[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids, 'neg': neg_pids}

print(f"Training data loaded. Total number of training queries: {len(train_queries)}")

# Custom Dataset Class
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = []
        self.corpus = corpus
        
        # Flatten data: Create an example for each query-positive pair
        for qid, query_data in queries.items():
            query_text = query_data['query']
            # Create an entry for each positive example
            for pos_pid in query_data['pos']:
                if pos_pid in corpus:  
                    self.queries.append({
                        'query': query_text,
                        'positive': corpus[pos_pid]
                    })

    def __getitem__(self, idx):
        query_data = self.queries[idx]
        return InputExample(texts=[query_data['query'], query_data['positive']])

    def __len__(self):
        return len(self.queries)

# DataLoader and Loss function
print("\nPreparing DataLoader...")
train_dataset = MSMARCODataset(queries=train_queries, corpus=corpus)
train_dataloader = DataLoader(
    train_dataset, 
    shuffle=True, 
    batch_size=train_batch_size, 
    drop_last=True,
    num_workers=0
)

# Loss function
train_loss = losses.MultipleNegativesRankingLoss(
    model=model,
    scale=20.0, 
    similarity_fct=util.dot_score 
)

class TrainingProgress:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.start_time = datetime.now()
        self.losses = []
        self.epoch_losses = [] 
        
    def __call__(self, score, epoch, steps):
        self.current_epoch = epoch
        elapsed_time = datetime.now() - self.start_time
        estimated_time = (elapsed_time / epoch) * (self.total_epochs - epoch) if epoch > 0 else None
        
        # Check and save loss value
        if not (torch.isinf(torch.tensor(score)) or torch.isnan(torch.tensor(score))):
            self.losses.append(score)
            if len(self.losses) % len(train_dataloader) == 0:  # End of epoch
                epoch_avg = sum(self.losses[-len(train_dataloader):]) / len(train_dataloader)
                self.epoch_losses.append(epoch_avg)
                if epoch_avg < self.best_loss:
                    self.best_loss = epoch_avg
        
        print(f"\n{'='*20} Epoch {epoch}/{self.total_epochs} {'='*20}")
        print(f"Step: {steps}/{len(train_dataloader)}")
        print(f"Current loss: {score:.4f}")
        
        if len(self.epoch_losses) > 0:
            print(f"Last epoch average loss: {self.epoch_losses[-1]:.4f}")
            print(f"Best epoch loss: {self.best_loss:.4f}")
        
        print(f"Remaining epochs: {self.total_epochs - epoch}")
        print(f"Elapsed time: {elapsed_time}")
        if estimated_time:
            print(f"Estimated remaining time: {estimated_time}")
        
        if torch.cuda.is_available():
            print(f"GPU Memory Usage: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            print(f"GPU Cache Usage: {torch.cuda.memory_reserved()/1024**2:.1f}MB")

progress_tracker = TrainingProgress(epochs)

try:
    print("\n=== Training Starting ===")
    print(f"Total number of examples: {len(train_dataset)}")
    print(f"Batch size: {train_batch_size}")
    print(f"Total number of batches (per epoch): {len(train_dataloader)}")
    print(f"Total number of steps: {epochs * len(train_dataloader)}")
    print("=====================")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        use_amp=True,
        checkpoint_path=model_save_path,
        checkpoint_save_steps=len(train_dataloader),
        checkpoint_save_total_limit=1,
        optimizer_params={'lr': lr},
        max_grad_norm=1.0,
        show_progress_bar=True,
        callback=progress_tracker
    )
except Exception as e:
    print(f"\nError occurred during training: {str(e)}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    raise e

print("\n=== Training Completed! ===")
if len(progress_tracker.losses) > 0:
    print(f"Last average loss: {sum(progress_tracker.losses[-10:])/min(len(progress_tracker.losses),10):.4f}")
print(f"Best loss: {progress_tracker.best_loss:.4f}")
print(f"Model saved: {model_save_path}")

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Save Model
model.save(model_save_path)

# Training Summary
print("\n=== Training Summary ===")
print(f"Total number of documents: {len(corpus)}")
print(f"Total number of queries: {len(queries)}")
print(f"Total training queries: {len(train_queries)}")
print(f"Best loss: {progress_tracker.best_loss:.4f}")
print(f"Model save path: {model_save_path}")
if torch.cuda.is_available():
    print(f"Final GPU Memory Usage: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
print("===================")
