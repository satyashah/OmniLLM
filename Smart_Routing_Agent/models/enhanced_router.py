# Smart_Routing_Agent/models/enhanced_router.py
"""
EnhancedRouter (Classifier‑Based):

How It Works:
It extracts basic features from the query (like word count, average word length, number of question marks, etc.) and then feeds those features into a classifier (for example, a logistic regression model). This classifier is trained on a labeled dataset (even a small or dummy one) to predict a binary decision:
0 (“single”): Use a weak model (or a single model)
1 (“ensemble”): Use multiple (strong) models and then fuse the responses
Pros and Cons:
Pros:
Simple and fast to compute
Easy to train with small datasets or handcrafted labels
Cons:
The decision is binary (single vs. ensemble) without ranking individual models
The features are manually chosen and might not capture all the nuances of query complexity
May be less flexible if you want to select among a wide variety of models
"""
import os
import numpy as np
import pickle

class EnhancedRouter:
    """
    An enhanced router that uses a classifier (e.g. logistic regression)
    to decide whether to route the query to the "single" (weak model) or "ensemble" (strong model) branch.
    
    Mode 0 -> "single"
    Mode 1 -> "ensemble"
    """
    def __init__(self, config, classifier_path: str = None):
        self.config = config
        self.fallback_threshold = config.router_length_threshold  # fallback heuristic threshold
        self.classifier = None
        if classifier_path and os.path.exists(classifier_path):
            with open(classifier_path, "rb") as f:
                self.classifier = pickle.load(f)
        else:
            print("No trained router classifier found; using fallback heuristic.")
    
    def extract_features(self, query: str) -> np.ndarray:
        """
        Extract basic features from the query.
        Example features:
          - Word count
          - Average word length
          - Count of question marks
        """
        words = query.split()
        word_count = len(words)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        question_marks = query.count('?')
        # Additional features could be added here.
        return np.array([word_count, avg_word_len, question_marks]).reshape(1, -1)
    
    def route(self, query: str) -> dict:
        """
        Decide on the routing mode using the classifier if available;
        otherwise, use a fallback heuristic.
        """
        if self.classifier is not None:
            features = self.extract_features(query)
            pred = self.classifier.predict(features)[0]
            mode = "ensemble" if pred == 1 else "single"
        else:
            # Fallback: use word count threshold.
            mode = "single" if len(query.split()) < self.fallback_threshold else "ensemble"
        return {"mode": mode}
