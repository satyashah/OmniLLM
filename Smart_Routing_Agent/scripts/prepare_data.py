# scripts/prepare_data.py
import os
import json
from Smart_Routing_Agent.utils.data import load_jsonl
from Smart_Routing_Agent.config.config import Config

def process_dataset():
    config = Config()
    input_file = os.path.join("Smart_Routing_Agent/data", "mix_instruct_train.jsonl")
    output_file = os.path.join(config.processed_data_dir, "processed_train.jsonl")
    os.makedirs(config.processed_data_dir, exist_ok=True)
    data = load_jsonl(input_file)
    with open(output_file, "w", encoding="utf-8") as fout:
        for example in data:
            processed = {
                "prompt": example.get("instruction", "") + " " + example.get("input", ""),
                "reference": example.get("output", ""),
                "candidates": [cand.get("text", "") for cand in example.get("candidates", [])]
            }
            fout.write(json.dumps(processed) + "\n")
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    process_dataset()
