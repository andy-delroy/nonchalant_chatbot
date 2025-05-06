import json

with open("ner_fresh_balanced_dataset.json", "r") as f:
    data = json.load(f)

print(type(data))  # Should print: <class 'list'>
print(len(data))   # Should print: 100
print(data[0])     # Should print the first query dict