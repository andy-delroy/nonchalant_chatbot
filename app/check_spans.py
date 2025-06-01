import json

with open("mutli_label_dataset.json", "r") as f:
    data = json.load(f)

for item in data:
    text = item["text"]
    for ent in item["entities"]:
        extracted = text[ent["start"]:ent["end"]]
        if extracted != ent["text"]:
            print(f"Mismatch in '{text}'")
            print(f"Entity: '{ent['text']}', Label: {ent['label']}, Expected Span: {ent['start']}-{ent['end']}")
            print(f"Extracted: '{extracted}'")
            print("---")
        else:
            print("All fine")
