from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "Transformers are amazing!"
encoded_input = tokenizer(text, return_tensors='pt')

model = BertModel.from_pretrained('bert-base-uncased')

with torch.no_grad():
    outputs = model(**encoded_input)

embeddings = outputs.last_hidden_state
print(embeddings)
