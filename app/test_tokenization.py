# test_tokenization.py
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("C:/Users/Adrian/_enhancement/app/fine_tuned_ner")
tokens = tokenizer("Give me all the assets from Branch Office", return_offsets_mapping=True)
for token, (start, end) in zip(tokenizer.convert_ids_to_tokens(tokens["input_ids"]), tokens["offset_mapping"]):
    print(f"Token: {token:<15} Start: {start:<3} End: {end}")