import os
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient
from PIL import Image

# Initialize FastAPI app
app = FastAPI(
    title="HuggingFace Image Generator API",
    description="Generate images from text prompts using FLUX.1-schnell model",
    version="1.0.0"
)

# Initialize Hugging Face clients
# Image generation uses auto provider
image_client = InferenceClient(
    provider="auto",
    api_key=os.environ.get("HF_TOKEN"),
)

# Video generation uses replicate provider
video_client = InferenceClient(
    provider="replicate",
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
                "prompt": "3 well adorned African priests riding horses in early Jerusalem, following the stars, cinematic lighting, 4k",
                "width": 1024,
                "height": 576,
                "guidance_scale": 7.5,
                "num_inference_steps": 4
            }
        }

# Video generation request model
class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to generate video from", min_length=1)
    model: str = Field(
        "Wan-AI/Wan2.2-T2V-A14B",
        description="Model to use for video generation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A young man walking on the street waving an Algerian flag, smiling",
                "model": "Wan-AI/Wan2.2-T2V-A14B"
            }
        }

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - Interactive HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>HuggingFace AI Generator</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            
            header {
                text-align: center;
                color: white;
                margin-bottom: 40px;
            }
            
            h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            
            .subtitle {
                font-size: 1.1rem;
                opacity: 0.9;
            }
            
            .cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 30px;
                margin-bottom: 40px;
            }
            
            .card {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            
            .card h2 {
                color: #667eea;
                margin-bottom: 20px;
                font-size: 1.8rem;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            label {
                display: block;
                margin-bottom: 8px;
                color: #333;
                font-weight: 600;
            }
            
            input[type="text"],
            input[type="number"],
            textarea {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                transition: border-color 0.3s;
            }
            
            input:focus,
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            
            textarea {
                resize: vertical;
                min-height: 80px;
            }
            
            .form-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
            }
            
            button {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            
            button:active {
                transform: translateY(0);
            }
            
            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
            }
            
            .result {
                margin-top: 25px;
                text-align: center;
            }
            
            .result img {
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.2);
            }
            
            .result video {
                max-width: 100%;
                border-radius: 10px;
                box-shadow: 0 5px 20px rgba(0,0,0,0.2);
            }
            
            .loading {
                display: inline-block;
                width: 50px;
                height: 50px;
                border: 5px solid #f3f3f3;
                border-top: 5px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .error {
                color: #d32f2f;
                background: #ffebee;
                padding: 15px;
                border-radius: 8px;
                margin-top: 15px;
            }
            
            .api-section {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                margin-top: 30px;
            }
            
            .api-section h2 {
                color: #667eea;
                margin-bottom: 25px;
                font-size: 1.8rem;
            }
            
            .api-section h3 {
                color: #444;
                margin-top: 25px;
                margin-bottom: 15px;
                font-size: 1.3rem;
            }
            
            .endpoint {
                background: #f8f9fa;
                border-left: 4px solid #667eea;
                padding: 20px;
                margin-bottom: 25px;
                border-radius: 8px;
            }
            
            .endpoint-header {
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 15px;
            }
            
            .method {
                display: inline-block;
                padding: 6px 12px;
                border-radius: 5px;
                font-weight: 700;
                font-size: 0.9rem;
                font-family: 'Courier New', monospace;
            }
            
            .method.post {
                background: #4caf50;
                color: white;
            }
            
            .method.get {
                background: #2196f3;
                color: white;
            }
            
            .endpoint-path {
                font-family: 'Courier New', monospace;
                font-size: 1.1rem;
                color: #333;
                font-weight: 600;
            }
            
            .endpoint-description {
                color: #666;
                margin-bottom: 15px;
                line-height: 1.6;
            }
            
            .code-block {
                background: #1e1e1e;
                color: #d4d4d4;
                padding: 20px;
                border-radius: 8px;
                overflow-x: auto;
                font-family: 'Courier New', monospace;
                font-size: 0.9rem;
                line-height: 1.5;
                margin-top: 10px;
            }
            
            .code-block .key {
                color: #9cdcfe;
            }
            
            .code-block .string {
                color: #ce9178;
            }
            
            .code-block .number {
                color: #b5cea8;
            }
            
            .quick-links {
                display: flex;
                gap: 15px;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }
            
            .quick-link {
                display: inline-block;
                padding: 12px 24px;
                background: #667eea;
                color: white;
                text-decoration: none;
                border-radius: 8px;
                font-weight: 600;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            .quick-link:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            
            .param-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 10px;
            }
            
            .param-table th,
            .param-table td {
                text-align: left;
                padding: 12px;
                border-bottom: 1px solid #e0e0e0;
            }
            
            .param-table th {
                background: #f5f5f5;
                font-weight: 600;
                color: #333;
            }
            
            .param-table code {
                background: #f5f5f5;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }
            
            @media (max-width: 768px) {
                .cards {
                    grid-template-columns: 1fr;
                }
                
                .form-row {
                    grid-template-columns: 1fr;
                }
                
                h1 {
                    font-size: 2rem;
                }
                
                .endpoint-header {
                    flex-direction: column;
                    align-items: flex-start;
                }
                
                .quick-links {
                    flex-direction: column;
                }
                
                .quick-link {
                    text-align: center;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <h1>🎨 HuggingFace AI Generator</h1>
                <p class="subtitle">Generate stunning images and videos with AI</p>
            </header>
            
            <div class="cards">
                <!-- Image Generation Card -->
                <div class="card">
                    <h2>🖼️ Image Generation</h2>
                    <form id="imageForm">
                        <div class="form-group">
                            <label for="imagePrompt">Prompt</label>
                            <textarea id="imagePrompt" required placeholder="Describe the image you want to generate...">3 well adorned African priests riding horses in early Jerusalem, following the stars, cinematic lighting, 4k</textarea>
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label for="width">Width (px)</label>
                                <input type="number" id="width" value="1024" min="256" max="2048" required>
                            </div>
                            <div class="form-group">
                                <label for="height">Height (px)</label>
                                <input type="number" id="height" value="576" min="256" max="2048" required>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label for="guidanceScale">Guidance Scale</label>
                                <input type="number" id="guidanceScale" value="7.5" min="1" max="20" step="0.5" required>
                            </div>
                            <div class="form-group">
                                <label for="steps">Inference Steps</label>
                                <input type="number" id="steps" value="4" min="1" max="50" required>
                            </div>
                        </div>
                        <button type="submit">Generate Image</button>
                    </form>
                    <div id="imageResult" class="result"></div>
                </div>
                
                <!-- Video Generation Card -->
                <div class="card">
                    <h2>🎥 Video Generation</h2>
                    <form id="videoForm">
                        <div class="form-group">
                            <label for="videoPrompt">Prompt</label>
                            <textarea id="videoPrompt" required placeholder="Describe the video you want to generate...">A young man walking on the street waving an Algerian flag, smiling</textarea>
                        </div>
                        <div class="form-group">
                            <label for="model">Model</label>
                            <input type="text" id="model" value="Wan-AI/Wan2.2-T2V-A14B" required>
                        </div>
                        <button type="submit">Generate Video</button>
                    </form>
                    <div id="videoResult" class="result"></div>
                </div>
            </div>
            
            <!-- API Documentation -->
            <div class="api-section">
                <h2>📡 API Documentation</h2>
                
                <div class="quick-links">
                    <a href="/docs" class="quick-link" target="_blank">📖 Swagger UI Docs</a>
                    <a href="/redoc" class="quick-link" target="_blank">📄 ReDoc Documentation</a>
                    <a href="/health" class="quick-link" target="_blank">💚 Health Check</a>
                </div>
                
                <h3>Available Endpoints</h3>
                
                <!-- Image Generation Endpoint -->
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method post">POST</span>
                        <span class="endpoint-path">/generate</span>
                    </div>
                    <p class="endpoint-description">
                        Generate an image from a text prompt using the FLUX.1-schnell model. Returns a PNG image file.
                    </p>
                    
                    <strong>Request Body:</strong>
                    <table class="param-table">
                        <thead>
                            <tr>
                                <th>Parameter</th>
                                <th>Type</th>
                                <th>Required</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><code>prompt</code></td>
                                <td>string</td>
                                <td>Yes</td>
                                <td>Text description of the image to generate</td>
                            </tr>
                            <tr>
                                <td><code>width</code></td>
                                <td>integer</td>
                                <td>No</td>
                                <td>Image width in pixels (256-2048, default: 1024)</td>
                            </tr>
                            <tr>
                                <td><code>height</code></td>
                                <td>integer</td>
                                <td>No</td>
                                <td>Image height in pixels (256-2048, default: 576)</td>
                            </tr>
                            <tr>
                                <td><code>guidance_scale</code></td>
                                <td>float</td>
                                <td>No</td>
                                <td>Guidance scale for generation (1.0-20.0, default: 7.5)</td>
                            </tr>
                            <tr>
                                <td><code>num_inference_steps</code></td>
                                <td>integer</td>
                                <td>No</td>
                                <td>Number of inference steps (1-50, default: 4)</td>
                            </tr>
                        </tbody>
                    </table>
                    
                    <strong>Example cURL Request:</strong>
                    <div class="code-block">curl -X POST "http://localhost:8000/generate" \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "3 well adorned African priests riding horses in early Jerusalem, following the stars, cinematic lighting, 4k",
    "width": 1024,
    "height": 576,
    "guidance_scale": 7.5,
    "num_inference_steps": 4
  }' \\
  --output generated_image.png</div>
                    
                    <strong>Example Python Request:</strong>
                    <div class="code-block">import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "3 well adorned African priests riding horses in early Jerusalem, following the stars, cinematic lighting, 4k",
        "width": 1024,
        "height": 576,
        "guidance_scale": 7.5,
        "num_inference_steps": 4
    }
)

with open("generated_image.png", "wb") as f:
    f.write(response.content)</div>
                </div>
                
                <!-- Video Generation Endpoint -->
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method post">POST</span>
                        <span class="endpoint-path">/generate-video</span>
                    </div>
                    <p class="endpoint-description">
                        Generate a video from a text prompt using text-to-video models. Returns an MP4 video file.
                    </p>
                    
                    <strong>Request Body:</strong>
                    <table class="param-table">
                        <thead>
                            <tr>
                                <th>Parameter</th>
                                <th>Type</th>
                                <th>Required</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><code>prompt</code></td>
                                <td>string</td>
                                <td>Yes</td>
                                <td>Text description of the video to generate</td>
                            </tr>
                            <tr>
                                <td><code>model</code></td>
                                <td>string</td>
                                <td>No</td>
                                <td>Model to use (default: "Wan-AI/Wan2.2-T2V-A14B")</td>
                            </tr>
                        </tbody>
                    </table>
                    
                    <strong>Example cURL Request:</strong>
                    <div class="code-block">curl -X POST "http://localhost:8000/generate-video" \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "A young man walking on the street waving an Algerian flag, smiling",
    "model": "Wan-AI/Wan2.2-T2V-A14B"
  }' \\
  --output generated_video.mp4</div>
                    
                    <strong>Example Python Request:</strong>
                    <div class="code-block">import requests

response = requests.post(
    "http://localhost:8000/generate-video",
    json={
        "prompt": "A young man walking on the street waving an Algerian flag, smiling",
        "model": "Wan-AI/Wan2.2-T2V-A14B"
    }
)

with open("generated_video.mp4", "wb") as f:
    f.write(response.content)</div>
                </div>
                
                <!-- Health Check Endpoint -->
                <div class="endpoint">
                    <div class="endpoint-header">
                        <span class="method get">GET</span>
                        <span class="endpoint-path">/health</span>
                    </div>
                    <p class="endpoint-description">
                        Check API health status and verify HuggingFace token configuration.
                    </p>
                    
                    <strong>Example cURL Request:</strong>
                    <div class="code-block">curl -X GET "http://localhost:8000/health"</div>
                    
                    <strong>Example Response:</strong>
                    <div class="code-block">{
  "status": "healthy",
  "hf_token_configured": true
}</div>
                </div>
            </div>
        </div>
        
        <script>
            // Image Generation
            document.getElementById('imageForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const resultDiv = document.getElementById('imageResult');
                const button = e.target.querySelector('button');
                
                button.disabled = true;
                resultDiv.innerHTML = '<div class="loading"></div>';
                
                try {
                    const response = await fetch('/generate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            prompt: document.getElementById('imagePrompt').value,
                            width: parseInt(document.getElementById('width').value),
                            height: parseInt(document.getElementById('height').value),
                            guidance_scale: parseFloat(document.getElementById('guidanceScale').value),
                            num_inference_steps: parseInt(document.getElementById('steps').value)
                        })
                    });
                    
                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Generation failed');
                    }
                    
                    const blob = await response.blob();
                    const imageUrl = URL.createObjectURL(blob);
                    
                    resultDiv.innerHTML = `
                        <img src="${imageUrl}" alt="Generated Image">
                        <p style="margin-top: 15px;">
                            <a href="${imageUrl}" download="generated_image.png" style="color: #667eea; text-decoration: none; font-weight: 600;">
                                📥 Download Image
                            </a>
                        </p>
                    `;
                } catch (error) {
                    resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                } finally {
                    button.disabled = false;
                }
            });
            
            // Video Generation
            document.getElementById('videoForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const resultDiv = document.getElementById('videoResult');
                const button = e.target.querySelector('button');
                
                button.disabled = true;
                resultDiv.innerHTML = '<div class="loading"></div><p style="margin-top: 15px;">Video generation may take a few minutes...</p>';
                
                try {
                    const response = await fetch('/generate-video', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            prompt: document.getElementById('videoPrompt').value,
                            model: document.getElementById('model').value
                        })
                    });
                    
                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Generation failed');
                    }
                    
                    const blob = await response.blob();
                    const videoUrl = URL.createObjectURL(blob);
                    
                    resultDiv.innerHTML = `
                        <video controls autoplay loop>
                            <source src="${videoUrl}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                        <p style="margin-top: 15px;">
                            <a href="${videoUrl}" download="generated_video.mp4" style="color: #667eea; text-decoration: none; font-weight: 600;">
                                📥 Download Video
                            </a>
                        </p>
                    `;
                } catch (error) {
                    resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                } finally {
                    button.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    hf_token = os.environ.get("HF_TOKEN")
    return {
        "status": "healthy",
        "hf_token_configured": bool(hf_token)
    }

@app.post("/generate", response_class=StreamingResponse)
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
        image = image_client.text_to_image(
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

@app.post("/generate-video", response_class=StreamingResponse)
async def generate_video(request: VideoGenerationRequest):
    """
    Generate a video from a text prompt using text-to-video models

    Returns the video directly as an MP4 file
    """
    try:
        # Check if HF_TOKEN is set
        if not os.environ.get("HF_TOKEN"):
            raise HTTPException(
                status_code=500,
                detail="HF_TOKEN environment variable not set"
            )

        # Generate video using Hugging Face Inference API
        # text_to_video returns bytes directly
        video = video_client.text_to_video(
            request.prompt,
            model=request.model,
        )

        # Convert bytes to BytesIO for streaming
        video_bytes = io.BytesIO(video)

        # Return video as streaming response
        return StreamingResponse(
            video_bytes,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"inline; filename=generated_video.mp4"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Video generation failed: {str(e)}"
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
