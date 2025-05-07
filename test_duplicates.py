import json

def load_pairs(file_path):
    pairs = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            query = item['query'].strip().lower()
            positive = item['positive'].strip().lower()
            pairs.add((query, positive))
    return pairs

# Paths to your files
train_path = r"C:\Users\Asus\PycharmProjects\Fine_tuning_Sentence_transformer\data\training_marco_sm.jsonl"
test_path = r"C:\Users\Asus\PycharmProjects\Fine_tuning_Sentence_transformer\data\testing_marco_sm.jsonl"

# Load pairs
train_pairs = load_pairs(train_path)
test_pairs = load_pairs(test_path)

# Find overlaps
common = train_pairs.intersection(test_pairs)


with open(train_path, 'r', encoding='utf-8') as f:
 lines = [line for line in f if line.strip()]
 total_lines=len(lines)



print(f"📄 Total entries in file : {total_lines}")


# Report
print(f"🧠 Train examples: {len(train_pairs)}")
print(f"🧪 Test examples: {len(test_pairs)}")
print(f"⚠️ Duplicate query-positive pairs found: {len(common)}")
print(f"Found internal duplicates in Training: {total_lines-len(train_pairs)}")

# Print duplicates if needed
for i, (q, p) in enumerate(common):
    print(f"{i+1}. Query: {q} \n   Answer: {p}\n")
