import json

with open("ner_fresh_balanced_dataset.json", "r") as f:
    data = json.load(f)

all_labels = set()
for item in data:
    for entity in item["entities"]:
        all_labels.add(entity["label"])

print("Unique labels in dataset:", sorted(all_labels))
expected_labels = ["LOCATION", "STATUS", "DEPARTMENT", "CONDITION", "USER", "DATE"]  # Base labels without B-/I-
print("Expected base labels:", expected_labels)
missing_labels = [label for label in all_labels if label not in expected_labels]
print("Unexpected labels:", missing_labels)