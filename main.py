from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from kafka import KafkaProducer, KafkaConsumer
from dotenv import load_dotenv
import os
import json
import logging
import time
from uuid import uuid4

load_dotenv()
app = FastAPI()

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

producer = KafkaProducer(
    bootstrap_servers=os.getenv("KAFKA_HOST", "127.0.0.1") + ':9093',
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )
consumer = KafkaConsumer(
    'llm_responses',
    bootstrap_servers=os.getenv("KAFKA_HOST", "127.0.0.1") + ':9093',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='api-consumer',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

class Question(BaseModel):
    question: str


@app.post("/chat/")
async def chat_endpoint(question: Question):
    try:
        message_id = str(uuid4())
        logging.info(f"Received question: {question.question} id:{message_id}")
        data = {"message_id": message_id, "query": question.question}
        producer.send('user_queries', value = data)
        start_time = time.time()
        while True:
           for message in consumer:
               if message.value['original_message']['message_id'] == message_id:
                   logging.info(f"Response received for message id:{message_id} in {time.time() - start_time} seconds")
                   return {"response": message.value["transformed_message"]}
               if time.time() - start_time > 30:
                  raise HTTPException(status_code=504, detail="Timeout waiting for response from LLM")


    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)