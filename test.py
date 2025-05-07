# test.py

from sentence_transformers import SentenceTransformer, util
import json
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import os

def evaluate(model_path):
    model = SentenceTransformer(model_path)
    query = "How can I return a product?"
    pos = "Go to 'My Orders' and click 'Return'."
    neg = "Shipping may take 7 days."

    emb_query = model.encode(query, convert_to_tensor=True)
    emb_pos = model.encode(pos, convert_to_tensor=True)
    emb_neg = model.encode(neg, convert_to_tensor=True)

    sim_pos = util.cos_sim(emb_query, emb_pos)
    sim_neg = util.cos_sim(emb_query, emb_neg)

    print(f"Positive similarity: {sim_pos.item():.4f}")
    print(f"Negative similarity: {sim_neg.item():.4f}")

def evaluate_macro_file(model_path, file_path):
    model = SentenceTransformer(model_path)

    y_true = []
    y_pred = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            query = sample['query']
            pos = sample['positive']
            negatives = sample['negatives']

            # Encode all options
            emb_query = model.encode(query, convert_to_tensor=True)
            emb_pos = model.encode(pos, convert_to_tensor=True)
            emb_negs = model.encode(negatives, convert_to_tensor=True)

            # Similarities
            sim_pos = util.cos_sim(emb_query, emb_pos)
            sim_negs = util.cos_sim(emb_query, emb_negs)

            # Prediction logic
            if sim_pos > sim_negs.max():
                y_pred.append(1)  # model picked positive
            else:
                y_pred.append(0)  # model picked wrong

            y_true.append(1)  # because positive is always the true label

    # Evaluation
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    test_file = r"data/testing_marco_sm.jsonl"

    print("🚀 Evaluating base model:")
    evaluate('msmarco-distilbert-base-v4')
    evaluate_macro_file('msmarco-distilbert-base-v4', test_file)

    print("\n🔥 Evaluating fine-tuned model:")
    evaluate('models/output')
    evaluate_macro_file('models/output', test_file)
