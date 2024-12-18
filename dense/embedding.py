import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True).cuda()

def encode_file(file_path, model):
    embeddings = []
    with open(file_path, "r") as file:
        for line in file:
            embedding = model.encode(line.strip())
            embeddings.append(embedding)
    return embeddings

def save_embeddings(embeddings, file_path):
    np.save(file_path, embeddings)

def load_embeddings(file_path):
    return np.load(file_path, allow_pickle=True)

doc_embeddings = encode_file("./docs.txt", model)
save_embeddings(doc_embeddings, "./doc_embeddings.npy")

query_embeddings = encode_file("./queries.txt", model)
save_embeddings(query_embeddings, "./query_embeddings.npy")

loaded_doc_embeddings = load_embeddings("./doc_embeddings.npy")
loaded_query_embeddings = load_embeddings("./query_embeddings.npy")
