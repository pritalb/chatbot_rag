import json
import re
import string


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return  text


def load_faq(filepath):
    with open(filepath, "r") as file:
        faq_data = json.load(file)
    return faq_data


def process_data(filepath):
    faq_data = load_faq(filepath)
    processed_data = []

    for item in faq_data:
        processed_data.append(
            {
                "question": preprocess_text(item["question"]),
                "answer": preprocess_text(item["answer"])
            }
        )

    return processed_data


if __name__ == "__main__":
    faq_file_path = "./data/faq.json"
    processed_data = process_data(faq_file_path)
    print(json.dumps(processed_data, indent=2))
