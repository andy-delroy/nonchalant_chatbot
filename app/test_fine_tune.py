# test_fine_tune.py
from transformers import pipeline
ner = pipeline(
    "ner",
    model="C:/Users/Adrian/_enhancement/app/fine_tuned_ner",
    tokenizer="C:/Users/Adrian/_enhancement/app/fine_tuned_ner",
    aggregation_strategy="simple"  # Groups B- and I- into full entities
)
print(ner("Give me all the assets from Branch Office"))
print(ner("give me all the assets from branch office"))