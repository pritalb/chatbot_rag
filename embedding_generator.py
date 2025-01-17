from transformers import AutoTokenizer, AutoModel
import torch
import json
from preprocess import process_data


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device}')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # first element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings

def generate_embeddings(text_data, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    all_embeddings = []

    for item in text_data:
        encoded_input = tokenizer(item["question"], padding=True, truncation=True, return_tensors='pt').to(device)

        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"]).cpu()

        all_embeddings.append({
            "question": item["question"],
            "answer": item["answer"],
            "embedding": sentence_embeddings.tolist()[0]
        })
    return all_embeddings

if __name__ == "__main__":
    faq_file_path = "./data/faq.json"
    text_data = process_data(faq_file_path)
    model_name = "sentence-transformers/all-mpnet-base-v2"
    embeddings = generate_embeddings(text_data, model_name)
    print(json.dumps(embeddings, indent=2))