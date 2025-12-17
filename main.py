import os
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient
from PIL import Image

# Initialize FastAPI app
app = FastAPI(
    title="HuggingFace Image Generator API",
    description="Generate images from text prompts using FLUX.1-schnell model",
    version="1.0.0"
)

# Initialize Hugging Face client
client = InferenceClient(
    provider="auto",
    api_key=os.environ.get("HF_TOKEN"),
)

# Request model with validation
class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to generate image from", min_length=1)
    width: int = Field(1024, description="Image width in pixels", ge=256, le=2048)
    height: int = Field(576, description="Image height in pixels (default 16:9 ratio)", ge=256, le=2048)
    guidance_scale: float = Field(7.5, description="Guidance scale for generation", ge=1.0, le=20.0)
    num_inference_steps: int = Field(4, description="Number of inference steps", ge=1, le=50)

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Astronaut riding a horse on Mars, cinematic lighting, 4k",
                "width": 1024,
                "height": 576,
                "guidance_scale": 7.5,
                "num_inference_steps": 4
            }
        }

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "HuggingFace Image Generator API",
        "docs": "/docs",
        "generate_endpoint": "/generate (POST)",
        "health_endpoint": "/health (GET)"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    hf_token = os.environ.get("HF_TOKEN")
    return {
        "status": "healthy",
        "hf_token_configured": bool(hf_token)
    }

@app.post("/generate")
async def generate_image(request: ImageGenerationRequest):
    """
    Generate an image from a text prompt using FLUX.1-schnell model
    
    Returns the image directly as a PNG file (16:9 aspect ratio by default)
    """
    try:
        # Check if HF_TOKEN is set
        if not os.environ.get("HF_TOKEN"):
            raise HTTPException(
                status_code=500, 
                detail="HF_TOKEN environment variable not set"
            )
        
        # Generate image using Hugging Face Inference API
        image = client.text_to_image(
            request.prompt,
            model="black-forest-labs/FLUX.1-schnell",
            width=request.width,
            height=request.height,
            # Note: FLUX.1-schnell is optimized for speed with fewer steps
            # Some parameters may not be supported by this specific model
        )
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Return image as streaming response
        return StreamingResponse(
            img_byte_arr, 
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename=generated_image.png"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}"
        )

# Optional: Add CORS middleware if you need to access from browser
# Uncomment the following lines if needed:
# from fastapi.middleware.cors import CORSMiddleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)