import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
import json

# Verify JSON
with open("ner_fresh_balanced_dataset.json", "r") as f:
    data = json.load(f)
    print(f"Loaded data type: {type(data)}, length: {len(data)}")
    print(f"First entry: {data[0]}")

# Load model and tokenizer
model_name = "dslim/bert-base-NER"
labels = ["O", "B-LOCATION", "I-LOCATION", "B-STATUS", "I-STATUS", "B-DEPARTMENT", "I-DEPARTMENT", 
          "B-CONDITION", "I-CONDITION", "B-USER", "I-USER", "B-DATE", "I-DATE"]
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(labels), ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.config.label2id = {label: idx for idx, label in enumerate(labels)}
model.config.id2label = {idx: label for idx, label in enumerate(labels)}
print(f"Model num_labels: {model.config.num_labels}")

# Load dataset without shuffling
dataset = load_dataset("json", data_files="ner_fresh_balanced_dataset.json", split="train")
dataset = dataset.train_test_split(test_size=0.2, shuffle=False)

# Tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        return_offsets_mapping=True,
        is_split_into_words=False
    )
    labels = []
    for i, entity_list in enumerate(examples["entities"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        offsets = tokenized_inputs["offset_mapping"][i]
        label_ids = [-100] * len(tokenized_inputs["input_ids"][i])

        for entity in entity_list:
            start, end = entity["start"], entity["end"]
            b_label = f"B-{entity['label']}"
            i_label = f"I-{entity['label']}"
            if b_label not in model.config.label2id:
                print(f"Warning: Invalid label {b_label} in {examples['text'][i]}")
                continue
            for token_idx, (token_start, token_end) in enumerate(offsets):
                if token_end <= start or token_start >= end:
                    continue
                if label_ids[token_idx] != -100:  # Skip if already labeled
                    continue
                if token_start <= start < token_end:
                    label_ids[token_idx] = model.config.label2id[b_label]
                elif start < token_start < end:
                    label_ids[token_idx] = model.config.label2id[i_label]

        for idx in range(len(label_ids)):
            if label_ids[idx] == -100 and tokenized_inputs["input_ids"][i][idx] not in (tokenizer.cls_token_id, tokenizer.sep_token_id):
                label_ids[idx] = model.config.label2id["O"]
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Debug first example
first_example = tokenized_dataset["train"][0]
tokens = tokenizer.convert_ids_to_tokens(first_example["input_ids"])
labels_out = [model.config.id2label.get(l, "PAD") if l != -100 else "PAD" for l in first_example["labels"]]
print("Tokens and Labels for first example:")
for token, label in zip(tokens, labels_out):
    print(f"{token:<15} {label}")

# Training args
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Train and save
trainer.train()
model.save_pretrained("./fine_tuned_balanced_with_v3")
tokenizer.save_pretrained("./fine_tuned_balanced_with_v3")