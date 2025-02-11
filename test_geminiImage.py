from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_images(
    model='imagen-3.0-generate-002',
    prompt='A serene landscape at sunset',
    config=types.GenerateImagesConfig(
        number_of_images=1,
        aspect_ratio='16:9',
        person_generation='ALLOW_ADULT'
    )
)

for generated_image in response.generated_images:
    image = Image.open(BytesIO(generated_image.image.image_bytes))
    image.show()