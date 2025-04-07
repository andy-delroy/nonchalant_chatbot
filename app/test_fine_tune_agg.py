# test_fine_tune_agg.py
from transformers import pipeline
ner_agg = pipeline("ner", model="C:/Users/Adrian/_enhancement/app/fine_tuned_ner", tokenizer="C:/Users/Adrian/_enhancement/app/fine_tuned_ner", aggregation_strategy="simple")
print(ner_agg("Give me all the assets from Branch Office"))
print(ner_agg("give me all the assets from branch office"))