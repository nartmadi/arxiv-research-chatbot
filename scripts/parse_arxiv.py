import json
import faiss
import numpy as np
import time
import psutil
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

INPUT_DIR = "../data/arxiv-dataset"
OUTPUT_DIR = "../data"

JSON_FILE = f"{INPUT_DIR}/arxiv_dataset.json"
FAISS_FLATL2_INDEX_FILE = f"{OUTPUT_DIR}/arxiv_flatl2.index"
FAISS_IVF_INDEX_FILE = f"{OUTPUT_DIR}/arxiv_ivf.index"
FAISS_HNSW_INDEX_FILE = f"{OUTPUT_DIR}/arxiv_hnsw.index"
METADATA_FILE = f"{OUTPUT_DIR}/arxiv_metadata.json"

NUM_QUERIES = 100
NUM_CLUSTERS = 4096
NUM_NEIGHBORS = 10
BATCH_SIZE = 32

print("Loading Sentence-BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = model.get_sentence_embedding_dimension()

print("Initializing FAISS GPU resources...")
gpu_res = faiss.StandardGpuResources()

# Process dataset
metadata_store = {}
paper_ids = []
embeddings = []
batch_texts = []

print("Processing JSON file and generating embeddings...")

with open(JSON_FILE, 'r', encoding='utf-8') as file:
    for paper in tqdm(file, desc='Processing Papers'):
        try:
            data = json.loads(paper)

            paper_id = data.get('id', None)
            title = data.get('title', '')
            abstract = data.get('abstract', '').strip()
            authors = data.get('authors_parsed', [])
            categories = data.get('categories', '')
            journal_ref = data.get('journal_ref', None)
            doi = data.get('doi', None)
            update_date = data.get('update_date', None)

            if authors:
                authors = ", ".join([" ".join(a[:2]) for a in authors])
            
            if not abstract or not title:
                continue

            batch_texts.append(abstract)
            paper_ids.append(paper_id)
            metadata_store[paper_id] = {
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'categories': categories,
                'journal_ref': journal_ref,
                'doi': doi,
                'update_date': update_date
            }

            if len(batch_texts) >= BATCH_SIZE:
                batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
                embeddings.extend(batch_embeddings)
                batch_texts = []

        except json.JSONDecodeError:
            print('Skipping invalid JSON paper entry')

if batch_texts:
    batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
    embeddings.extend(batch_embeddings)

embeddings = np.array(embeddings, dtype=np.float32)

query_vectors = np.random.rand(NUM_QUERIES, embedding_dim).astype("float32")

quantizer = faiss.IndexFlatL2(embedding_dim)
faiss_ivf = faiss.IndexIVFFlat(quantizer, embedding_dim, NUM_CLUSTERS, faiss.METRIC_L2)

print("\n🔹 Training FAISS IVFFlat (Clustered FAISS Index)...")
faiss_ivf.train(embeddings[:100_000])
faiss_ivf.add(embeddings)

faiss.write_index(faiss.index_gpu_to_cpu(faiss_ivf), FAISS_IVF_INDEX_FILE)
print("✅ FAISS IVFFlat index saved successfully!")

print("Saving full metadata...")

with open(METADATA_FILE, 'w', encoding='utf-8') as file:
    for paper_id in paper_ids:
        entry = metadata_store[paper_id]
        entry['id'] = paper_id
        file.write(json.dumps(entry) + '\n')

print("✅ Full metadata saved successfully!")