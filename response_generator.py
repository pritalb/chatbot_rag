from kafka import KafkaConsumer, KafkaProducer
import json
from rag_pipeline_faiss import create_rag_chain_faiss
import time
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_queries(input_topic, output_topic, model_name, index_path, docstore_path, hf_llm_id, bootstrap_servers="localhost:9092"):
    consumer = KafkaConsumer(
        input_topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='response-generator-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    )
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda x: json.dumps(x).encode("utf-8"),
    )
    qa_chain = create_rag_chain_faiss(model_name, index_path, docstore_path, hf_llm_id)

    for message in consumer:
        try:
            data = message.value
            logging.info(f"Processing message: {data['message_id']}")
            response = qa_chain.invoke({"input": data["query"]})
            logging.info(f"The full response created by the rag chain is {response}")

            transformed_data = {
                "original_message": data,
                "transformed_message": response["answer"],
            }
            producer.send(output_topic, value=transformed_data)
            logging.info(f"Generated response for message id:{data['message_id']}")
        except Exception as e:
            logging.error(f"Error processing message: {e}")
        producer.flush()
        print("finished")


if __name__ == '__main__':
    input_topic = "user_queries"
    output_topic = "llm_responses"
    model_name = "sentence-transformers/all-mpnet-base-v2"
    index_path = "data/faiss_index.pkl"
    docstore_path = "data/faiss_docstore.pkl"
    hf_llm_id = "google/flan-t5-small"
    process_queries(input_topic, output_topic, model_name, index_path, docstore_path, hf_llm_id)
