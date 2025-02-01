from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess


def kafka_producer_task():
    subprocess.run(["python", "opt/airflow/kafka_producer.py"], check=True)


def kafka_transformer_task():
    subprocess.run(["python", "/opt/airflow/data_transformer.py"], check=True)


def generate_faiss_index():
    subprocess.run(["python", "opt/airflow/vector_store_faiss.py"], check=True)


with DAG(
    dag_id="kafka_data_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=["kafka-airflow"],
) as dag:
    produce_data = PythonOperator(
        task_id="produce_data",
        python_callable=kafka_producer_task,
    )

    transform_data = PythonOperator(
        task_id="transform_data",
        python_callable=kafka_transformer_task,
    )

    generate_faiss_index = PythonOperator(
        task_id="generate_faiss_index",
        python_callable=generate_faiss_index,
    )

    produce_data >> transform_data >> generate_faiss_index
