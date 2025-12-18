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

def test_generate_video(prompt="A young man walking on the street", output_file="test_generated.mp4"):
    """Test the video generation endpoint"""
    print(f"\nTesting /generate-video endpoint with prompt: '{prompt}'...")
    print("⚠️  Note: Video generation may take several minutes...")
    try:
        response = requests.post(
            f"{API_URL}/generate-video",
            json={
                "prompt": prompt,
                "model": "Wan-AI/Wan2.2-T2V-A14B"
            },
            timeout=300  # 5 minute timeout for video generation
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            with open(output_file, "wb") as f:
                f.write(response.content)
            print(f"✓ Video saved to: {output_file}")
            return True
        else:
            print(f"✗ Error: {response.json()}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("HuggingFace Image & Video Generator API - Test Script")
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
    print("IMAGE GENERATION TEST")
    print("=" * 60)
    generate_ok = test_generate()

    # Ask if user wants to test video generation (it takes longer)
    print("\n" + "=" * 60)
    print("VIDEO GENERATION TEST (Optional - takes several minutes)")
    print("=" * 60)
    test_video = input("Do you want to test video generation? (y/n): ").lower().strip()

    video_ok = True  # Default to True if not tested
    if test_video == 'y':
        video_ok = test_generate_video()
    else:
        print("Skipping video generation test.")

    print("\n" + "=" * 60)
    if health_ok and generate_ok and video_ok:
        print("✓ All tests passed!")
        print("\nYou can now:")
        print(f"  - View API docs at: {API_URL}/docs")
        print(f"  - Check the generated image: test_generated.png")
        if test_video == 'y':
            print(f"  - Check the generated video: test_generated.mp4")
    else:
        print("✗ Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
