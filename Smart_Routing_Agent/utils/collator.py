# utils/collator.py
def simple_collate(batch: list) -> dict:
    collated = {}
    for d in batch:
        for k, v in d.items():
            collated.setdefault(k, []).append(v)
    return collated
