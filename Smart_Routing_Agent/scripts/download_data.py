# scripts/download_data.py
import os
from datasets import load_dataset
import json
from config import Config

def download_mixinstruct():
    config = Config()
    dataset_name = config.mixinstruct_dataset_name
    print(f"Downloading dataset {dataset_name}...")
    ds = load_dataset(dataset_name)
    
    # For example, save the train split
    save_dir = "data"
    os.makedirs(save_dir, exist_ok=True)
    train_file = os.path.join(save_dir, "mix_instruct_train.jsonl")
    with open(train_file, "w", encoding="utf-8") as fout:
        for example in ds["train"]:
            fout.write(json.dumps(example) + "\n")
    print(f"Train split saved to {train_file}")

if __name__ == "__main__":
    download_mixinstruct()
