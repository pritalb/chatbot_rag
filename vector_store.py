import faiss
import numpy as np
from embedding_generator import generate_embeddings
from preprocess import process_data
import pickle
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document


def create_and_populate_faiss_index(embeddings, index_path, docstore_path):
    embeddings_array = np.array([item["embedding"] for item in embeddings]).astype(
        "float32"
    )
    dimension = embeddings_array.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    with open(index_path, "w") as f:
        pickle.dump(index, f)

    documents = {
        str(i): Document(
            page_content=item["question"], metadata={"answer": item["answer"]}
        )
        for i, item in enumerate(embeddings)
    }

    docstore = InMemoryDocstore()
    docstore.add(texts=documents)
    with open(docstore_path, "wb") as f:
        pickle.dump(docstore, f)

    return index, docstore


if __name__ == "__main__":
    print("running vector store....")

    faq_file_path = "./data/faq.json"
    text_data = process_data(faq_file_path)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = generate_embeddings(text_data, model_name)

    index_path = "./data/faiss_index.pkl"
    metadata_path = "./data/faiss_docstore.pkl"
    index, docstore = create_and_populate_faiss_index(
        embeddings, index_path, metadata_path
    )

    print(f"Created and populated FAISS index at {index_path}")
    print(f"metadata count:{len(docstore._dict)}")
