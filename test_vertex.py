import os
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from google.cloud import aiplatform
from typing import List
import base64

# Check environment variables at the beginning
project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
location = os.getenv("GOOGLE_CLOUD_LOCATION")
model_name = "imagen-3.0-generate-002"
prompt = "Pink panda on a skateboard at maryland university chasing a smoking snoop dogg poodle"

if not project_id or not location:
    raise ValueError("GOOGLE_CLOUD_PROJECT_ID and GOOGLE_CLOUD_LOCATION environment variables must be set.")

def test_vertex_ai(project_id: str, location: str, model_name: str, prompt: str) -> None:
    """Calls Vertex AI's image generation model."""
    
    # Initialize the vertex with the environment variable, otherwise, crash
    vertexai.init(project=project_id, location=location)
    
    #Load model
    model = ImageGenerationModel.from_pretrained(model_name)
    
    #Generate Images
    try:
        images = model.generate_images(  # Use async
            prompt=prompt,
            number_of_images=1,  # number of images
        )
        print ("Successfully called generate images")

        #Base 64 encode it for easier transport
        base64_encoded_images = []  # array to store b64 encoded images
        for image in images:
            #buffered = io.BytesIO() #no need to save buffered
            #image.save(buffered, format="PNG") #error is happening here since Vertex AI has data
            img_str = base64.b64encode(image._image_bytes).decode('utf-8')  # encode #image.image_bytes is the right way
            print(f"Generated image with bytes {len(image._image_bytes)}")
            print("Success")

            #urls.append(f"data:image/png;base64,{img_str}")


    except Exception as e:
        print(f"There was an error {str(e)}") #to help you with debugging in detail

# Check and call the test
if __name__ == "__main__":
    try:
        test_vertex_ai(project_id = project_id, location = location, model_name = model_name, prompt = prompt)
    except Exception as e:
        print(f"Exception outside the function {str(e)}")