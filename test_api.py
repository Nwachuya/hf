#!/usr/bin/env python3
"""
Simple test script for the HuggingFace Image Generator API
Usage: python test_api.py
"""

import requests
import sys

# Configuration
API_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_generate(prompt="A beautiful sunset over the ocean", output_file="test_generated.png"):
    """Test the image generation endpoint"""
    print(f"\nTesting /generate endpoint with prompt: '{prompt}'...")
    try:
        response = requests.post(
            f"{API_URL}/generate",
            json={
                "prompt": prompt,
                "width": 1024,
                "height": 576,
                "guidance_scale": 7.5,
                "num_inference_steps": 4
            }
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"✓ Image saved to: {output_file}")
            return True
        else:
            print(f"✗ Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("HuggingFace Image Generator API - Test Script")
    print("=" * 60)
    
    # Test health endpoint
    health_ok = test_health()
    
    if not health_ok:
        print("\n✗ Health check failed. Is the server running?")
        print("  Start the server with: python main.py")
        sys.exit(1)
    
    print("\n✓ Health check passed!")
    
    # Test image generation
    print("\n" + "=" * 60)
    generate_ok = test_generate()
    
    print("\n" + "=" * 60)
    if health_ok and generate_ok:
        print("✓ All tests passed!")
        print("\nYou can now:")
        print(f"  - View API docs at: {API_URL}/docs")
        print(f"  - Check the generated image: test_generated.png")
    else:
        print("✗ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
