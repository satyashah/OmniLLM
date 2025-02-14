# models/ranker.py

"""
The PairRanker is designed to compare candidate responses for a given prompt and assign a “quality score” to each candidate. 

Here's a detailed breakdown of its functionality and how a training routine (train_on_dataset) could be implemented:
Ranking Inference (score_candidate & rank functions)
    1) Input Preparation:
        When you call score_candidate, it takes a user prompt and one candidate answer. It concatenates these into a single string in the format:

        "Prompt: {prompt} [SEP] Candidate: {candidate}"

        This helps the model focus on the relationship between the prompt and the candidate response.

    2) Tokenization:
        The combined text is tokenized using a pretrained tokenizer (loaded from a model like a fine‑tuned BERT variant). The tokenizer converts the text into input IDs and attention masks that are fed into the classification model.

    3) Model Forward Pass:
        The tokenized inputs are passed to a pretrained sequence classification model (e.g., a BERT-based classifier). This model outputs raw logits – numbers that represent the unnormalized scores for each label. 
        In this context, we assume the classifier has two output logits (binary classification) where:

            Logit for class 0 might represent “candidate is not ideal”
            Logit for class 1 represents “candidate is ideal” (or simply a higher quality)
    4) Probability Computation:
        The logits are passed through a softmax function to get probabilities. The method then returns the probability corresponding to label 1 as the score for that candidate. A higher score indicates a better candidate response.

    5) Ranking:
        The rank method applies score_candidate to every candidate in a list and collects (candidate, score) pairs. It then sorts these pairs in descending order of the score. This sorted list provides a ranking of candidates from best to worst based on the model’s learned judgment.

Training the Ranker (train_on_dataset)

Dataset Format:
You would need a training dataset where each example contains:

prompt: The user query or instruction.
candidate1: A candidate response.
candidate2: Another candidate response.
label: A binary value (e.g., 1 if candidate1 is better than candidate2, and 0 if candidate2 is better).
Pairwise Input Creation:
For each training example, you would create an input string that concatenates the prompt with both candidate responses. One common approach is to create two inputs (or even a single input with a fixed format) and let the model output a score that represents the quality difference between candidate1 and candidate2.

Forward Pass & Loss Calculation:

The model processes the concatenated input and outputs logits.
You then use a loss function (commonly Binary Cross-Entropy with Logits Loss) comparing the model’s output (or the difference between two logits) against the ground truth label.
The loss will encourage the model to output higher scores for the candidate that is labeled as better.
Backpropagation:
The loss is backpropagated through the model, and the model parameters are updated using an optimizer (like AdamW).

Iteration:
This training loop is repeated over multiple epochs and batches of data, gradually fine‑tuning the ranker to predict which candidate is superior.

Evaluation & Saving:
After training, you can evaluate the ranker on a validation set to check how well it ranks candidate pairs. Finally, the fine‑tuned model is saved for later inference.

"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class PairRanker:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.ranker_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.ranker_model_name)
        self.device = config.device
        self.model.to(self.device)
    
    def score_candidate(self, prompt: str, candidate: str) -> float:
        text = f"Prompt: {prompt} [SEP] Candidate: {candidate}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        return probs[0][1].item()
    
    def rank(self, prompt: str, candidates: list) -> list:
        scored = [(cand, self.score_candidate(prompt, cand)) for cand in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def best_candidate(self, prompt: str, candidates: list) -> str:
        ranked = self.rank(prompt, candidates)
        return ranked[0][0]
