import os
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

class APIClient:
    def __init__(self, api_key: str = None):

        load_dotenv()
        
        if api_key is None:
            # Try to get from environment variable
            api_key = os.getenv('OMNI_API_KEY')

        if not api_key:
            raise ValueError(
                "No API key provided. Pass it when initializing the client or "
                "set the OMNI_API_KEY environment variable."
            )
            
        self._api_key = api_key
        self._base_url =  "http://localhost:8000"
        

    def _make_request(self, endpoint: str, method: str = 'GET', **kwargs):
        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'application/json'
        }
        
        response = requests.request(
            method=method,
            url=f"{self._base_url}/{endpoint.lstrip('/')}",
            headers=headers,
            **kwargs
        )
        
        response.raise_for_status()
        return response.json()
        
    def chat(self, messages, model: str, temperature: float = 0.7, max_tokens: int = 100) -> dict:
        """
        Send a chat completion request.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            model (str): The model ID to use
            temperature (float, optional): Sampling temperature. Defaults to 0.7
            max_tokens (int, optional): Maximum tokens to generate. Defaults to 100
            
        Returns:
            dict: The API response containing the chat completion
        """
        request_data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        return self._make_request(
            endpoint="/v1/chat/completions",
            method="POST",
            json=request_data
        )

    def generate_image(self, prompt: str, model: str, n: int = 1, size: str = "1024x1024") -> dict:
        """
        Generate images from a text prompt.
        
        Args:
            prompt (str): The image generation prompt
            model (str): The model ID to use
            n (int, optional): Number of images to generate. Defaults to 1
            size (str, optional): Image size. Defaults to "1024x1024"
            
        Returns:
            dict: The API response containing image URLs
        """
        request_data = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size
        }
        
        return self._make_request(
            endpoint="/v1/images/generate",
            method="POST",
            json=request_data
        )

    def get_available_models(self, model_type: str = None) -> List[Dict[str, Any]]:
        """
        Get list of available models from the API
        
        Args:
            model_type (str, optional): Type of models to get ('chat' or 'image'). 
                                      If None, returns all models.
        
        Returns:
            List[Dict[str, Any]]: List of available models and their information
        """
        if model_type == 'chat':
            return self._make_request("v1/models/chat")["models"]
        elif model_type == 'image':
            return self._make_request("v1/models/image")["models"]
        else:
            # Get all models if no specific type is requested
            chat_models = self._make_request("v1/models/chat")["models"]
            image_models = self._make_request("v1/models/image")["models"]
            return chat_models + image_models

# Usage example:
# client = APIClient(api_key='your-api-key-here')
# Or using environment variable:
# client = APIClient()