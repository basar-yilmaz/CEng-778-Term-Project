# %%
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
import torch
import pickle
from tqdm import tqdm

# %%
# Load data from data path
data_path = "../data"

# load docs
with open(f"{data_path}/docs.pkl", "rb") as f:
    docs = pickle.load(f)

# load queries
with open(f"{data_path}/queries.pkl", "rb") as f:
    queries = pickle.load(f)

# %%
print(f"Number of documents: {len(docs)}")
print(f"Number of queries: {len(queries)}")

# %%
# Preprocess documents without any text
docs = [doc for doc in docs if doc.text is not None]


# %%
class DocumentDataset(Dataset):
    def __init__(self, documents):
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        # Return only the `text` attribute
        return self.documents[idx].text


class QueryDataset(Dataset):
    def __init__(self, queries):
        self.queries = queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        # Return only the `query` attribute
        return self.queries[idx].query


def collate_and_tokenize(batch, tokenizer, max_length=512):
    return tokenizer(
        batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )


# %%
# Initialize PyTorch dataset
query_dataset = QueryDataset(queries)
doc_dataset = DocumentDataset(docs)

# %%
# Initialize model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/msmarco-bert-base-dot-v5"
)
model = AutoModel.from_pretrained("sentence-transformers/msmarco-bert-base-dot-v5")
model = torch.nn.DataParallel(model)

# %%
batch_size = 4096

# Create DataLoaders
document_loader = DataLoader(
    doc_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda batch: collate_and_tokenize(batch, tokenizer),
)

query_loader = DataLoader(
    query_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda batch: collate_and_tokenize(batch, tokenizer),
)


# %%
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # Extract the last hidden state
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def cls_pooling(model_output):
    # The [CLS] token is at index 0
    return model_output.last_hidden_state[:, 0, :].cpu()  # Output the [CLS] token


def compute_embeddings(
    loader, model, device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing embeddings"):
            # Pass tokenized data to the model
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                model_output = model(**batch)
            # batch_embeddings = mean_pooling(model_output, batch["attention_mask"])
            batch_embeddings = cls_pooling(model_output)
            embeddings.append(batch_embeddings)

    return torch.cat(embeddings, dim=0)


# %%
# Compute embeddings
query_embeddings = compute_embeddings(query_loader, model)
print("Query Embeddings Shape:", query_embeddings.shape)

doc_embeddings = compute_embeddings(document_loader, model)
print("Document Embeddings Shape:", doc_embeddings.shape)

# %%
with open("doc_embeddings.pkl", "wb") as f:
    pickle.dump(
        {
            "embeddings": doc_embeddings.cpu().numpy(),
            "doc_ids": [doc.doc_no for doc in docs],
        },
        f,
    )

with open("query_embeddings.pkl", "wb") as f:
    pickle.dump(
        {
            "embeddings": query_embeddings.cpu().numpy(),
            "query_ids": [query.query_no for query in queries],
        },
        f,
    )
