# fine_tune.py
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # For synchronous CUDA error reporting

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
import json

# Verify JSON file
with open("ner_dataset.json", "r") as f:
    data = json.load(f)
    print(f"Loaded data type: {type(data)}, length: {len(data)}")
    print(f"First entry: {data[0]}")

# Load model and tokenizer
model_name = "dslim/bert-base-NER"
labels = ["O", "B-LOCATION", "I-LOCATION", "B-STATUS", "I-STATUS", "B-DEPARTMENT", "I-DEPARTMENT", 
          "B-CONDITION", "I-CONDITION", "B-USER", "I-USER", "B-DATE", "I-DATE"]
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    ignore_mismatched_sizes=True  # Expected due to different number of labels
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.config.label2id = {label: idx for idx, label in enumerate(labels)}
model.config.id2label = {idx: label for idx, label in enumerate(labels)}
print(f"Model num_labels: {model.config.num_labels}")  # Should be 13

# Load dataset
dataset = load_dataset("json", data_files="ner_dataset_new.json")
dataset = dataset["train"].train_test_split(test_size=0.2)

# Tokenize and align labels
def tokenize_and_align_labels(examples):
    # Tokenize without the invalid 'return_word_ids' argument
    tokenized_inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        return_offsets_mapping=True,  # Useful for aligning labels
        is_split_into_words=False     # 'text' is a list of strings, not pre-tokenized words
    )
    labels = []
    for i, entity_list in enumerate(examples["entities"]):
        # Get word IDs for the current sequence in the batch
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        offsets = tokenized_inputs["offset_mapping"][i]
        label_ids = [-100] * len(tokenized_inputs["input_ids"][i])  # -100 for padding/special tokens
        
        # Align labels with tokens
        for word_idx in set(word_ids) - {None}:  # Exclude padding (None)
            token_indices = [idx for idx, wid in enumerate(word_ids) if wid == word_idx]
            if not token_indices:
                continue
            word_start, word_end = offsets[token_indices[0]][0], offsets[token_indices[-1]][1]
            assigned = False
            for entity in entity_list:
                start, end = entity["start"], entity["end"]
                if word_start >= start and word_end <= end:
                    b_label = f"B-{entity['label']}"
                    i_label = f"I-{entity['label']}"
                    if b_label not in model.config.label2id:
                        print(f"Warning: Invalid label {b_label} in {examples['text'][i]}")
                        continue
                    label_ids[token_indices[0]] = model.config.label2id[b_label]
                    for idx in token_indices[1:]:
                        label_ids[idx] = model.config.label2id[i_label]
                    assigned = True
                    break
            if not assigned:
                for idx in token_indices:
                    label_ids[idx] = model.config.label2id["O"]
        
        # Set special tokens ([CLS], [SEP]) to "O"
        for idx, token_id in enumerate(tokenized_inputs["input_ids"][i]):
            if token_id in (tokenizer.cls_token_id, tokenizer.sep_token_id):
                label_ids[idx] = model.config.label2id["O"]
        
        # Validate label IDs
        for idx, label_id in enumerate(label_ids):
            if label_id != -100 and (label_id < 0 or label_id >= model.config.num_labels):
                print(f"Invalid label {label_id} at index {idx} in {examples['text'][i]}")
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply tokenization and alignment
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Debugging: Print tokens and labels for the first example
first_example = tokenized_dataset["train"][0]
tokens = tokenizer.convert_ids_to_tokens(first_example["input_ids"])
labels_out = [model.config.id2label.get(l, "PAD") if l != -100 else "PAD" for l in first_example["labels"]]
print("Tokens and Labels for first example:")
for token, label in zip(tokens, labels_out):
    print(f"{token:<15} {label}")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Train and save the model
trainer.train()
model.save_pretrained("./fine_tuned_ner")
tokenizer.save_pretrained("./fine_tuned_ner")