# HuggingFace Image Generator API

A simple REST API for generating images from text prompts using Hugging Face's FLUX.1-schnell model.

## Features

- 🚀 Fast image generation using FLUX.1-schnell model
- 📐 Customizable image dimensions (default 16:9 aspect ratio)
- 🎨 Adjustable generation parameters (guidance scale, inference steps)
- 📖 Automatic API documentation with Swagger UI
- 🐳 Docker support for easy deployment
- ☁️ Ready for Coolify deployment

## API Endpoints

### `GET /`
Root endpoint with API information

### `GET /health`
Health check endpoint - verifies API is running and HF_TOKEN is configured

### `POST /generate`
Generate an image from a text prompt

**Request Body:**
```json
{
  "prompt": "Astronaut riding a horse on Mars, cinematic lighting, 4k",
  "width": 1024,
  "height": 576,
  "guidance_scale": 7.5,
  "num_inference_steps": 4
}
```

**Parameters:**
- `prompt` (required): Text description of the image to generate
- `width` (optional, default: 1024): Image width in pixels (256-2048)
- `height` (optional, default: 576): Image height in pixels (256-2048)
- `guidance_scale` (optional, default: 7.5): How closely to follow the prompt (1.0-20.0)
- `num_inference_steps` (optional, default: 4): Number of denoising steps (1-50)

**Response:**
Returns the generated image directly as a PNG file

## Setup

### Prerequisites

- Python 3.11+
- Hugging Face account and API token

### Get Your Hugging Face Token

1. Go to https://huggingface.co/settings/tokens
2. Create a new token with "Read" access
3. Copy the token

### Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```bash
cp .env.example .env
```

5. Edit `.env` and add your Hugging Face token:
```
HF_TOKEN=hf_your_token_here
```

6. Run the application:
```bash
# Using Python directly
python main.py

# OR using Uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

7. Access the API:
- API Documentation: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

## Docker Deployment

### Build and Run Locally

```bash
# Build the image
docker build -t hf-image-generator .

# Run the container
docker run -d \
  -p 8000:8000 \
  -e HF_TOKEN=your_token_here \
  --name hf-image-api \
  hf-image-generator
```

### Using Docker Compose

Create a `docker-compose.yml`:
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HF_TOKEN=${HF_TOKEN}
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## Coolify Deployment

1. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. **In Coolify:**
   - Create a new application
   - Select "Docker" as the build pack
   - Connect your GitHub repository
   - Set the environment variable: `HF_TOKEN=your_token_here`
   - Deploy!

3. **Access your API:**
   - Your API will be available at your Coolify-provided URL
   - Access docs at: `https://your-app.coolify.io/docs`

## Usage Examples

### Using cURL

```bash
# Generate an image
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene landscape with mountains and a lake at sunset",
    "width": 1024,
    "height": 576
  }' \
  --output generated_image.png
```

### Using Python requests

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "A cute cat wearing a wizard hat",
        "width": 1024,
        "height": 576,
        "guidance_scale": 7.5
    }
)

if response.status_code == 200:
    with open("cat_wizard.png", "wb") as f:
        f.write(response.content)
    print("Image saved!")
else:
    print(f"Error: {response.json()}")
```

### Using JavaScript/Fetch

```javascript
fetch('http://localhost:8000/generate', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    prompt: 'A futuristic city with flying cars',
    width: 1024,
    height: 576
  })
})
.then(response => response.blob())
.then(blob => {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'generated_image.png';
  a.click();
});
```

## Testing the API

Once running, visit http://localhost:8000/docs to access the interactive Swagger UI where you can:
- See all available endpoints
- Test the API directly from your browser
- View request/response schemas
- Download generated images

## Image Specifications

### Default Dimensions
- **Width**: 1024 pixels
- **Height**: 576 pixels
- **Aspect Ratio**: 16:9 (optimized for widescreen)

### Supported Dimensions
- Minimum: 256x256 pixels
- Maximum: 2048x2048 pixels
- You can use any dimension within this range

### Popular Aspect Ratios
- **16:9 (Widescreen)**: 1024x576, 1920x1080
- **4:3 (Standard)**: 1024x768
- **1:1 (Square)**: 1024x1024
- **9:16 (Portrait)**: 576x1024

## Model Information

This API uses the **FLUX.1-schnell** model from Black Forest Labs:
- Optimized for speed (schnell = fast in German)
- High-quality image generation
- Default 4 inference steps for quick generation
- Hosted on Hugging Face Inference API

## Troubleshooting

### "HF_TOKEN environment variable not set"
Make sure you've set the `HF_TOKEN` environment variable with your Hugging Face API token.

### Slow generation
The first request may be slower as the model loads. Subsequent requests should be faster. Consider increasing `num_inference_steps` for higher quality (but slower generation).

### Out of memory errors
Try reducing the image dimensions (`width` and `height` parameters).

## License

MIT License - feel free to use this in your projects!

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
