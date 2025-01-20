from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import huggingface_hub
import pickle
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv


# Ignore the deprecation warning for HuggingFaceHub. HuggingFaceEndpoint has a bug that stops it from working as mentioned in the forum post:
# https://github.com/langchain-ai/langchain/issues/18321

load_dotenv()

def create_rag_chain_faiss(model_name, index_path, docstore_path, hf_llm_id):
    embeddings = HuggingFaceEmbeddings(model_name = model_name)

    with open(index_path, 'rb') as f:
        index = pickle.load(f)

    with open(docstore_path, 'rb') as f:
        docstore = pickle.load(f)

    vectorstore = FAISS(
        embedding_function = embeddings,
        index = index,
        docstore = docstore,
        index_to_docstore_id = {i : str(i) for i in range(len(docstore._dict))}
    )

    prompt_template = """
    You are a helpful customer support chatbot. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you do not know, don't try to make up an answer.
    Context:
    {context}
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("human", "{input}"),
        ]
    )

    llm = huggingface_hub.HuggingFaceHub(
        repo_id=hf_llm_id,
        huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        model_kwargs={"temperature": 0, "max_length": 1024, "max_new_tokens": 250}
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(vectorstore.as_retriever(), document_chain)

    return chain

if __name__ == "__main__":
    model_name = "sentence-transformers/all-mpnet-base-v2"
    index_path = "./data/faiss_index.pkl"
    docstore_path = "./data/faiss_docstore.pkl"
    hf_llm_id = "google/flan-t5-small"
    qa_chain = create_rag_chain_faiss(model_name, index_path, docstore_path, hf_llm_id)

    question = "How can I save my password?"
    response = qa_chain.invoke({"input": question})

    print(f"Question: {question}")
    print(f"Response: {response}")