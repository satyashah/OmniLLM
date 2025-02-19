# Smart_Routing_Agent/tests/test_mvp.py
import unittest
from Smart_Routing_Agent.config.config import Config
from Smart_Routing_Agent.models.router import Router
from Smart_Routing_Agent.models.ranker import PairRanker
from Smart_Routing_Agent.models.fuser import Fuser

class TestMVP(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.router = Router(self.config)
        self.ranker = PairRanker(self.config)
        self.fuser = Fuser(self.config)
    
    def test_router(self):
        query = "How do I reset my password?"
        decision = self.router.route(query)
        self.assertIn(decision["mode"], ["single", "ensemble"])
    
    def test_ranker_and_fuser(self):
        query = "What is the capital of France?"
        candidates = [
            "Paris is the capital of France.",
            "The capital of France is Lyon.",
            "I think it might be Marseille."
        ]
        ranked = self.ranker.rank(query, candidates)
        self.assertGreater(len(ranked), 0)
        top_candidates = [cand for cand, _ in ranked[:self.config.top_k_fusion]]
        fused = self.fuser.fuse(query, top_candidates, max_new_tokens=50)
        self.assertTrue(len(fused) > 0)

if __name__ == "__main__":
    unittest.main()

