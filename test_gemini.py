import os
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv(find_dotenv())

def test_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set in environment.")
        return

    # Instantiate the client with your API key and API version.
    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

    prompt = "What is the capital of France?"
    # Create a configuration using the supported type.
    config = types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=256
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt],
            config=config  # Use 'config', not 'generation_config'
        )
        print("Response from Gemini API:", response.text)
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    test_gemini()
