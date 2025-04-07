# test_fine_tune_raw.py
from transformers import pipeline
ner = pipeline(
    "ner",
    model="C:/Users/Adrian/_enhancement/app/fine_tuned_ner",
    tokenizer="C:/Users/Adrian/_enhancement/app/fine_tuned_ner"
)
result = ner("Give me all the assets from Branch Office")
for entity in result:
    print(f"Word: {entity['word']:<10} Entity: {entity['entity']:<10} Score: {entity['score']}")