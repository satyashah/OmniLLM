import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from Smart_Routing_Agent.config.config import Config
from Smart_Routing_Agent.models.ranker import PairRanker

def prepare_training_data(dataset_name: str, split: str = "train"):
    # Load a dataset from Hugging Face (e.g., the mix-instruct dataset)
    dataset = load_dataset(dataset_name, split=split)
    # Assume the dataset has fields: "instruction", "input", "output" and possibly candidate pairs.
    # For demonstration, we create dummy pairwise examples.
    examples = []
    for item in dataset:
        prompt = item.get("instruction", "") + " " + item.get("input", "")
        reference = item.get("output", "")
        # In practice, you need pairs of candidate responses and a label indicating which is better.
        # Here we create two dummy candidates by splitting the reference in two parts.
        if reference:
            half = len(reference) // 2
            cand1 = reference[:half]
            cand2 = reference[half:]
            # Dummy label: assume candidate 1 is better if its length is greater
            label = 1 if len(cand1) >= len(cand2) else 0
            examples.append({"prompt": prompt, "candidate1": cand1, "candidate2": cand2, "label": label})
    return examples

def collate_fn(batch):
    # Simple collate function for a list of dicts
    return batch

def train_ranker(config: Config, train_data):
    ranker = PairRanker(config)
    tokenizer = ranker.tokenizer
    model = ranker.model
    device = config.device
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    num_epochs = 3
    dataloader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_fn)
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            # For each example, encode the pair (prompt, candidate1, candidate2)
            batch_losses = []
            for example in batch:
                prompt = example["prompt"]
                cand1 = example["candidate1"]
                cand2 = example["candidate2"]
                label = torch.tensor(example["label"], dtype=torch.float, device=device)
                text = f"Prompt: {prompt} [SEP] Candidate1: {cand1} [SEP] Candidate2: {cand2}"
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                # Assume output logits: positive means candidate1 is better
                logit = outputs.logits[0][0]
                loss = loss_fn(logit.unsqueeze(0), label.unsqueeze(0))
                batch_losses.append(loss)
            if batch_losses:
                batch_loss = torch.stack(batch_losses).mean()
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += batch_loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
    # Save fine-tuned ranker model
    output_dir = "Smart_Routing_Agent/models/fine_tuned_ranker"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned ranker saved to {output_dir}")

if __name__ == "__main__":
    config = Config()
    # For demonstration, we load the mix-instruct dataset.
    train_data = prepare_training_data(config.mixinstruct_dataset_name, split="train")
    train_ranker(config, train_data)
