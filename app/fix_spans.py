import json
import re

# Load the dataset
with open("ner_dataset_new.json", "r") as f:
    data = json.load(f)

# Function to find true span of entity text in full text
def find_true_span(full_text, entity_text, old_start, old_end):
    # Try exact match starting near the old_start
    search_start = max(0, old_start - 10)  # Look 10 chars before old_start
    search_end = min(len(full_text), old_end + 10)  # Look 10 chars after old_end
    search_text = full_text[search_start:search_end]
    
    # Find the entity text in the search window
    match_pos = search_text.find(entity_text)
    if match_pos != -1:
        true_start = search_start + match_pos
        true_end = true_start + len(entity_text)
        return true_start, true_end
    
    # Fallback: Search entire text if not found in window
    global_pos = full_text.find(entity_text)
    if global_pos != -1:
        return global_pos, global_pos + len(entity_text)
    
    # If no match, raise a flag
    raise ValueError(f"Entity '{entity_text}' not found in '{full_text}'")

# Process each entry and fix spans
fixed_data = []
mismatch_count = 0

for item in data:
    text = item["text"]
    entities = item["entities"]
    fixed_entities = []
    
    for ent in entities:
        old_start, old_end = ent["start"], ent["end"]
        entity_text = ent["text"]
        extracted = text[old_start:old_end]
        
        # Check if the span is already correct
        if extracted == entity_text:
            fixed_entities.append(ent)
            continue
        
        # Fix the span
        try:
            true_start, true_end = find_true_span(text, entity_text, old_start, old_end)
            if (true_start != old_start or true_end != old_end):
                mismatch_count += 1
                print(f"Fixed in '{text}'")
                print(f"Entity: '{entity_text}', Old: {old_start}-{old_end} ('{extracted}'), New: {true_start}-{true_end}")
                print("---")
            fixed_entities.append({
                "text": entity_text,
                "label": ent["label"],
                "start": true_start,
                "end": true_end
            })
        except ValueError as e:
            print(f"Error: {e}")
            fixed_entities.append(ent)  # Keep original if unfixable
    
    fixed_data.append({"text": text, "entities": fixed_entities})

# Save the fixed dataset
with open("ner_dataset_fixed.json", "w") as f:
    json.dump(fixed_data, f, indent=4)

print(f"Processed {len(data)} entries, fixed {mismatch_count} spans.")
print("Saved to 'ner_dataset_fixed.json'")