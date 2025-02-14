# serverRouter/helpers/vertex_ai_helper.py
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
import base64
from typing import List

class VertexAiHelper:
    @staticmethod
    def call_sync(
        prompt: str,
        model_name: str,
        google_cloud_project_id: str,
        google_cloud_location: str,
        num_images: int = 1
    ) -> List[str]:
        """Synchronously calls Vertex AI's image generation model."""
        if not google_cloud_project_id or not google_cloud_location:
            raise ValueError("Project ID and Location are required")
            
        # Initialize Vertex AI with your project/region.
        vertexai.init(project=google_cloud_project_id, location=google_cloud_location)
        
        # Load the image generation model.
        model = ImageGenerationModel.from_pretrained(model_name)
        
        # Generate images.
        images = model.generate_images(prompt=prompt, number_of_images=num_images)
        
        data_urls = []
        for image in images:
            # Use image.image_bytes if available; fallback to _image_bytes.
            img_bytes = getattr(image, "image_bytes", None) or getattr(image, "_image_bytes", None)
            if not img_bytes:
                raise ValueError("No image bytes found in the generated image.")
            img_str = base64.b64encode(img_bytes).decode("utf-8")
            data_urls.append(f"data:image/png;base64,{img_str}")
        return data_urls

    @staticmethod
    async def call(
        prompt: str,
        model_name: str,
        google_cloud_project_id: str,
        google_cloud_location: str,
        num_images: int = 1
    ) -> List[str]:
        """Runs the synchronous call in a thread so it can be awaited."""
        import asyncio
        return await asyncio.to_thread(
            VertexAiHelper.call_sync,
            prompt,
            model_name,
            google_cloud_project_id,
            google_cloud_location,
            num_images
        )
