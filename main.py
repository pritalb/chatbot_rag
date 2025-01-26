from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline_faiss import get_chatbot_response
from dotenv import load_dotenv
import os
import uvicorn
import logging

load_dotenv()
app = FastAPI()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Question(BaseModel):
    question: str


@app.post("/chat/")
async def chat_endpoint(question: Question):
    logging.info(f"Received Question: {question.question}")

    model_name = "sentence-transformers/all-mpnet-base-v2"
    index_path = "./data/faiss_index.pkl"
    docstore_path = "./data/faiss_docstore.pkl"
    hf_llm_id = "google/flan-t5-small"

    response = get_chatbot_response(
        question.question, model_name, index_path, docstore_path, hf_llm_id
    )

    logging.info(f"Response: {response}")
    return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
