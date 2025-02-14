# MVP for Routing & Ensembling LLMs

This repository implements a minimal viable product (MVP) that integrates ideas from [LLM-Blender](https://github.com/yuchenlin/LLM-Blender) and [RouteLLM](https://github.com/lm-sys/RouteLLM). It features:
- A **Router** that decides whether a user query is “easy” (handled by a weak model) or “hard” (requiring an ensemble).
- A **Ranker** (inspired by PairRM) that scores candidate responses given a prompt.
- A **Fuser** (inspired by GenFuser) that fuses the top‑\(K\) candidate responses into a final answer.

The system also includes scripts to download and process the [MixInstruct dataset](https://huggingface.co/datasets/llm-blender/mix-instruct) so you can later fine‑tune your models if desired.

## Folder Structure

mvp/ ├── README.md # This file ├── requirements.txt # Python dependencies ├── config/ │ └── config.py # Configuration parameters (model names, thresholds, etc.) ├── data/ # Place to store downloaded & processed dataset files ├── models/ │ ├── init.py │ ├── router.py # The routing agent (decides single vs. ensemble) │ ├── ranker.py # Simplified pairwise ranker (scores candidate responses) │ └── fuser.py # Generative fusion module (fuses top candidates) ├── scripts/ │ ├── download_data.py # Downloads MixInstruct from Hugging Face │ ├── prepare_data.py # Processes the raw dataset into training format │ └── infer.py # Main inference script for testing the MVP ├── utils/ │ ├── data.py # Utility functions to load JSONL data │ └── collator.py # (Optional) Data collator for batching └── tests/ └── test_mvp.py # Simple tests for the router, ranker, and fuser