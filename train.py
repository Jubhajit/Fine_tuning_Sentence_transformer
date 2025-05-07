# train.py

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json
import os
import random


# Load your data (with negatives)
def load_data(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return []

    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} at line: {line}")
                continue  # Skip any malformed lines

            query = item.get('query')
            positive = item.get('positive')
            negatives = item.get('negatives', [])

            if not query or not positive or not negatives:
                continue

            # Select one negative randomly
            negative = random.choice(negatives)
            examples.append(InputExample(texts=[query, positive, negative]))
    return examples


# Fine-tune model
def train_model():
    model = SentenceTransformer('msmarco-distilbert-base-v4')

    # Load and prepare data
    data_path = r"C:\Users\Asus\PycharmProjects\Fine_tuning_Sentence_transformer\data\training_marco_sm.jsonl"
    print(f"Loading data from {data_path}...")
    train_examples = load_data(data_path)

    if not train_examples:
        print("No data loaded. Exiting...")
        return

    print(f"Loaded {len(train_examples)} training examples.")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.TripletLoss(model=model)

    # Train
    print("Training the model...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=200,
        output_path='models/output',
        show_progress_bar=True
    )

    print("✅ Fine-tuning complete! Model saved at models/output")


if __name__ == "__main__":
    train_model()
