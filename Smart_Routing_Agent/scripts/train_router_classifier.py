# Smart_Routing_Agent/scripts/train_router_classifier.py
import numpy as np
import os 
import pickle
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def extract_features(query: str) -> np.ndarray:
    words = query.split()
    word_count = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    question_marks = query.count('?')
    return np.array([word_count, avg_word_len, question_marks])

def load_mix_instruct_data(num_samples=1000):
    """
    Load a subset of the MixInstruct dataset from Hugging Face.
    For demonstration purposes, we use the "train" split.
    """
    ds = load_dataset("llm-blender/mix-instruct", split="train")
    # Assume the instruction field exists; otherwise, use "prompt"
    instructions = ds["instruction"] if "instruction" in ds.column_names else ds["prompt"]
    return instructions[:num_samples]

def create_training_data(instructions, threshold=20):
    """
    Create training examples.
    We'll assign a label 0 ("single") if word count < threshold, else 1 ("ensemble").
    """
    X, y = [], []
    for inst in instructions:
        features = extract_features(inst)
        label = 1 if features[0] >= threshold else 0
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)

def train_classifier():
    instructions = load_mix_instruct_data(num_samples=1000)
    X, y = create_training_data(instructions, threshold=20)
    
    # Split data for evaluation.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("Classifier accuracy:", acc)
    
    # Save the classifier.
    save_path = os.path.join("Smart_Routing_Agent", "models", "router_classifier.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Trained classifier saved to {save_path}")

if __name__ == "__main__":
    train_classifier()
