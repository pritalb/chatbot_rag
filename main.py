from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from kafka import KafkaProducer, KafkaConsumer
from rag_pipeline_faiss import get_chatbot_response
from dotenv import load_dotenv
import json
import time
from uuid import uuid4
import os
import uvicorn
import logging


load_dotenv()
app = FastAPI()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda x: json.dumps(x).encode("utf-8"),
)
consumer = KafkaConsumer(
    "llm_responses",
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="api-consumer",
    value_deserializer=lambda x: json.loads(x.decode("utf-8")),
)


class Question(BaseModel):
    question: str


@app.post("/chat/")
async def chat_endpoint(question: Question):
    try:
        message_id = str(uuid4())
        logging.info(f"Received Question: {question.question}")
        data = {
            "message_id": message_id,
            "query": question.question,
        }
        producer.send("user_queries", value=data)

        start_time = time.time()
        while True:
            for message in consumer:
                if message.value["original_message"]["message_id"] == message_id:
                    logging.info(f"Response received for message id: {message_id} in {time.time() - start_time} seconds")
                    return {
                        "response": message.value["transformed_message"]
                    }
                if time.time() - start_time > 60:
                    raise HTTPException(
                        status_code=504,
                        detail="Timeout waiting for response from LLM",
                    )
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )
        # model_name = "sentence-transformers/all-mpnet-base-v2"
        # index_path = "./data/faiss_index.pkl"
        # docstore_path = "./data/faiss_docstore.pkl"
        # hf_llm_id = "google/flan-t5-small"

        # response = get_chatbot_response(
        #     question.question, model_name, index_path, docstore_path, hf_llm_id
        # )

        # logging.info(f"Response: {response}")
        # return {"response": response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
