import faiss
import numpy as np
import json
from embedding_generator import generate_embeddings
from preprocess import process_data
import pickle

def create_and_populate_faiss_index(embeddings, index_path):
    print("creating vector db...")
    embeddings_array = np.array([item['embedding'] for item in embeddings]).astype('float32')    
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    with open(index_path, 'wb') as f:
        pickle.dump(index, f)

    metadata = [{
        "question": item["question"],
        "answer": item["answer"]
    } for item in embeddings]

    return index, metadata

if __name__ == "__main__":
    print("running vector store....")

    faq_file_path = "./data/faq.json"
    text_data = process_data(faq_file_path)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = generate_embeddings(text_data, model_name)

    index_path = "./data/faiss_index.pkl"
    index, metadata = create_and_populate_faiss_index(embeddings, index_path)

    print(f"Created and populated FAISS index at {index_path}")
    print(f"metadata count:{len(metadata)}")