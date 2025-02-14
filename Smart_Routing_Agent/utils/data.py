# utils/data.py
import json

def load_jsonl(file_path: str) -> list:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data
