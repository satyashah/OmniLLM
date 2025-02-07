import asyncio
from typing import Dict, Any
from serverRouter.core.interfaces import ChatProvider
from serverRouter.core.datamodels import ChatCompletionRequest, ChatCompletionResponse
from serverRouter.core.exceptions import ProviderError
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class LlamaProvider(ChatProvider):
    """Llama provider for locally hosted Llama2-7B-chat model"""

    def __init__(self, model_path: str = "./llama2-7b-chat", device: str = "cpu"):
        try:
            # Load tokenizer and model from the local path.
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                device_map="auto" if device != "cpu" else None
            )
            # Create a text-generation pipeline.
            self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=0 if device != "cpu" else -1)
        except Exception as e:
            raise ProviderError(f"Failed to initialize Llama model: {str(e)}")

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """
        Generate a chat completion using the locally hosted Llama2-7B-chat model.
        We run the generation in a thread to avoid blocking the event loop.
        """
        try:
            prompt = request.messages[-1].content
            # Wrap synchronous generation in asyncio.to_thread
            generation = await asyncio.to_thread(
                self.generator,
                prompt,
                max_length=(len(prompt.split()) + (request.max_tokens or 50)),
                temperature=request.temperature,
                num_return_sequences=1
            )
            # extraction: pipeline returns a list of dictionaries with "generated_text"
            content = generation[0]["generated_text"]
            return ChatCompletionResponse(
                model=request.model,
                content=content,
                provider="llama",
                usage={}  # Local models typically do not return token usage.
            )
        except Exception as e:
            raise ProviderError(f"Llama model error: {str(e)}")
