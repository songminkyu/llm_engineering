"""
Week 3 Day 1 - Image and Audio Generation
Refactored version for local environment

This script demonstrates:
1. Image generation using FLUX and Stable Diffusion models
2. Text-to-speech generation using Microsoft's SpeechT5

Requirements:
- Set your HF_TOKEN in config.py or as environment variable
- GPU recommended for best performance
"""

import torch
from diffusers import FluxPipeline, AutoPipelineForText2Image
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
from PIL import Image
import os

from config import HF_TOKEN, DEFAULT_DEVICE
from utils import authenticate_huggingface, check_gpu_availability, cleanup_memory


def generate_image_flux(prompt, output_filename="surreal.png"):
    """
    Generate image using FLUX model (requires powerful GPU like A100).
    
    Args:
        prompt (str): Text prompt for image generation
        output_filename (str): Output filename for the generated image
    
    Returns:
        PIL.Image: Generated image
    """
    print("Loading FLUX model...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell", 
        torch_dtype=torch.bfloat16
    ).to(DEFAULT_DEVICE)
    
    generator = torch.Generator(device=DEFAULT_DEVICE).manual_seed(0)
    
    print("Generating image...")
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
        generator=generator
    ).images[0]
    
    image.save(output_filename)
    print(f"Image saved as {output_filename}")
    
    # Clean up
    cleanup_memory(pipe, generator)
    
    return image


def generate_image_stable_diffusion(prompt, output_filename="stable_diffusion.png"):
    """
    Generate image using Stable Diffusion (works on T4 GPU).
    
    Args:
        prompt (str): Text prompt for image generation
        output_filename (str): Output filename for the generated image
    
    Returns:
        PIL.Image: Generated image
    """
    print("Loading Stable Diffusion model...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo", 
        torch_dtype=torch.float16, 
        variant="fp16"
    )
    pipe.to(DEFAULT_DEVICE)
    
    print("Generating image...")
    image = pipe(
        prompt=prompt, 
        num_inference_steps=1, 
        guidance_scale=0.0
    ).images[0]
    
    image.save(output_filename)
    print(f"Image saved as {output_filename}")
    
    # Clean up
    cleanup_memory(pipe)
    
    return image


def generate_speech(text, output_filename="speech.wav"):
    """
    Generate speech from text using Microsoft's SpeechT5.
    
    Args:
        text (str): Text to convert to speech
        output_filename (str): Output filename for the generated audio
    
    Returns:
        dict: Audio data and sampling rate
    """
    print("Loading SpeechT5 model...")
    synthesiser = pipeline(
        "text-to-speech", 
        "microsoft/speecht5_tts", 
        device=DEFAULT_DEVICE
    )
    
    print("Loading speaker embeddings...")
    embeddings_dataset = load_dataset(
        "matthijs/cmu-arctic-xvectors", 
        split="validation", 
        trust_remote_code=True
    )
    
    speaker_embedding = torch.tensor(
        embeddings_dataset[7306]["xvector"]
    ).unsqueeze(0)
    
    print("Generating speech...")
    speech = synthesiser(
        text, 
        forward_params={"speaker_embeddings": speaker_embedding}
    )
    
    sf.write(output_filename, speech["audio"], samplerate=speech["sampling_rate"])
    print(f"Audio saved as {output_filename}")
    
    # Clean up
    cleanup_memory(synthesiser, speaker_embedding)
    
    return speech


def main():
    """Main function to run the examples."""
    print("Week 3 Day 1 - Image and Audio Generation")
    print("=" * 50)
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    # Authenticate with HuggingFace
    try:
        authenticate_huggingface()
    except ValueError as e:
        print(f"Authentication error: {e}")
        return
    
    # Image generation prompts
    image_prompt = "A futuristic class full of students learning AI coding in the surreal style of Salvador Dali"
    speech_text = "Hi to an artificial intelligence engineer on the way to mastery!"
    
    print("\n1. Image Generation Examples")
    print("-" * 30)
    
    # Try FLUX first (requires powerful GPU)
    if gpu_available:
        try:
            print("Attempting FLUX image generation (requires powerful GPU)...")
            flux_image = generate_image_flux(image_prompt)
            print("FLUX generation successful!")
        except Exception as e:
            print(f"FLUX generation failed: {e}")
            print("Falling back to Stable Diffusion...")
    
    # Fallback to Stable Diffusion
    try:
        print("Generating image with Stable Diffusion...")
        sd_image = generate_image_stable_diffusion(image_prompt)
        print("Stable Diffusion generation successful!")
    except Exception as e:
        print(f"Stable Diffusion generation failed: {e}")
    
    print("\n2. Audio Generation Example")
    print("-" * 30)
    
    try:
        speech_data = generate_speech(speech_text)
        print("Speech generation successful!")
        print(f"Audio file created: speech.wav")
    except Exception as e:
        print(f"Speech generation failed: {e}")
    
    print("\nAll examples completed!")


if __name__ == "__main__":
    main()
