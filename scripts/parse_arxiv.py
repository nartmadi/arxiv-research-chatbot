import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Config
JSON_FILE = '../data/arxiv_dataset.json'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
FAISS_INDEX_FILE = '../data/arxiv_dataset.index'
METADATA_FILE = "../data/arxiv_metadata.json"

# Load Sentence-BERT model
model = SentenceTransformer(EMBEDDING_MODEL)

# Initialize FAISS index
embedding_dim = model.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(embedding_dim)

# Process JSON file
metadata_store = {}
paper_ids = []

with open(JSON_FILE, 'r', encoding='utf-8') as file:
    for paper in tqdm(file, desc='Processing Paper'):
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

            embedding = model.encode(abstract, convert_to_numpy=True, normalize_embeddings=True)

            faiss_index.add(np.array([embedding], dtype=np.float32))

            metadata_store[paper_id] = {
                'title': title,
                'abstract': abstract,
                'authors': authors,
                'categories': categories,
                'journal_ref': journal_ref,
                'doi': doi,
                'update_date': update_date
            }
            paper_ids.append(paper_id)
        except json.JSONDecodeError:
            print('Skipping invalid JSON paper entry')

faiss.write_index(faiss_index, FAISS_INDEX_FILE)

with open(METADATA_FILE, 'w', encoding='utf-8') as file:
    for paper_id in paper_ids:
        entry = metadata_store[paper_id]
        entry['id'] = paper_id
        file.write(json.dumps(entry) + '\n')
