import os
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

import requests
import json

API_KEY = os.getenv("STABILITY_API_KEY")
if not API_KEY:
    raise Exception("Missing STABILITY_API_KEY environment variable")

# Choose the Stability AI endpoint for Stable Image Ultra.
# (You can change '/ultra' to '/core' or '/sd3' to test other models.)
endpoint = "https://api.stability.ai/v2beta/stable-image/generate/ultra"

# To force multipart/form-data encoding, include a dummy file field.
files = {"none": (None, "")}

# Build the data payload.
# The API typically requires at least the prompt, mode, and output_format.
data = {
    "prompt": "A beautiful sunset over a mountain range",
    "mode": "text-to-image",      # Explicitly specify the mode
    "output_format": "png"        # Request PNG format
}

# Set the headers.
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "accept": "application/json"  # Request JSON response with base64 image data
}

print(f"Sending request to: {endpoint}")
response = requests.post(endpoint, headers=headers, files=files, data=data)

# Check and print the response.
if response.status_code != 200:
    print("Error:", response.status_code)
    print("Response:", response.text)
else:
    result = response.json()
    # For debugging, you can pretty-print the result.
    print("Success! Full response:")
    print(json.dumps(result, indent=2))
    
    # Check if there are any artifacts (generated images)
    artifacts = result.get("artifacts") or result.get("images")
    if artifacts:
        print("Number of generated images:", len(artifacts))
    else:
        print("No images were returned. Check your prompt and API key.")
