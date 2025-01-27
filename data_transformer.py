from kafka import KafkaConsumer, KafkaProducer
import json
import time


def transform_data(input_topic, output_topic, bootstrap_servers="localhost:9092"):
    consumer = KafkaConsumer(
        input_topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        group_id="transformer-group",
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda x: json.dumps(x).encode("utf-8"),
    )

    for message in consumer:
        data = message.value
        transformed_data = {
            "transformed_message": f"{data['query'].upper()}",
            "original_message": data,
        }
        print(f"transformed message id: {data['message_id']}")
        producer.send(output_topic, value=transformed_data)

    producer.flush()
    print("completed transformation")


if __name__ == "__main__":
    input_topic = "user_queries"
    output_topic = "transformed_queries"
    transform_data(input_topic, output_topic)
