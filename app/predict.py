from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load your fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_ner")
model = AutoModelForTokenClassification.from_pretrained("./fine_tuned_ner")

# New test text
text = "Show me assets in Quantum Hub."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits
predictions = torch.argmax(logits, dim=2)[0].tolist()
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
labels = [model.config.id2label[p] for p in predictions]

# Print token-label pairs
print("Tokens and Labels:")
for token, label in zip(tokens, labels):
    print(f"{token:<15} {label}")