# models/router.py
class Router:
    def __init__(self, config):
        self.threshold = config.router_length_threshold

    def route(self, query: str) -> dict:
        if len(query.split()) < self.threshold:
            return {"mode": "single"}
        else:
            return {"mode": "ensemble"}
