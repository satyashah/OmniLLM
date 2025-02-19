import os
import json
from Smart_Routing_Agent.utils.data import load_jsonl
from Smart_Routing_Agent.config.config import Config

def process_dataset():
    config = Config()
    # Compute the project root by going up one directory from this script's location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Now the input file is in the "data" folder at the project root
    input_file = os.path.join(base_dir, "data", "mix_instruct_train.jsonl")
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

