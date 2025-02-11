import base64
import os
import tempfile
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from typing import List

class VertexAiHelper:
    @staticmethod
    async def call(
        prompt: str,
        model_name: str,
        google_cloud_project_id: str,
        google_cloud_location: str,
        num_images: int = 1
    ) -> List[str]:
        """
        Calls Vertex AI's image generation model and returns a list of PNG data URLs
        for the generated images.
        """
        if not google_cloud_project_id or not google_cloud_location:
            raise ValueError(
                "Your Project ID and Project Location are required to call Vertex AI's generate image"
            )

        # Initialize Vertex AI with your project and location.
        vertexai.init(project=google_cloud_project_id, location=google_cloud_location)
        
        # Load the image generation model.
        model = ImageGenerationModel.from_pretrained(model_name)
        
        # Generate images using the provided prompt.
        images = model.generate_images(prompt=prompt, number_of_images=num_images)
        
        data_urls = []
        for image in images:
            # Create a temporary file with a .png suffix.
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                temp_filename = tmp_file.name
            try:
                # Save the generated image to the temporary file.
                # The Vertex AI SDK’s save() method expects a file path (a string)
                # and does not accept file-like objects (e.g. BytesIO).
                image.save(temp_filename, include_generation_parameters=False)
                
                # Read the saved image bytes.
                with open(temp_filename, "rb") as f:
                    img_bytes = f.read()
            finally:
                os.remove(temp_filename)
            
            # Encode the image bytes to base64.
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            # Construct the data URL for a PNG image.
            data_url = f"data:image/png;base64,{img_b64}"
            data_urls.append(data_url)
        
        return data_urls
