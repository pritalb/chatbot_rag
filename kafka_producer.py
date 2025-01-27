from kafka import KafkaProducer
import json
import time
import random


def produce_data(topic_name, num_messages, bootstrap_servers="localhost:9092"):
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda x: json.dumps(x).encode("utf-8"),
    )

    for i in range(num_messages):
        data = {
            "message_id": i,
            "user_id": random.randint(100, 200),
            "query": f"Test Query {i}",
            "timestamp": time.time(),
        }
        producer.send(topic_name, value=data)
        print(f"sent message {i}")
        time.sleep(1)
    producer.flush()
    print("finished sending data")


if __name__ == "__main__":
    topic_name = "user_queries"
    num_messages = 10
    produce_data(topic_name, num_messages)
    