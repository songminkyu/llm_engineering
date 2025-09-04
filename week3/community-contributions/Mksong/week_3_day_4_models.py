"""
Week 3 Day 4 - Models
Refactored version for local environment

This script demonstrates:
1. Loading and using different language models
2. Model quantization for memory efficiency
3. Text generation with streaming
4. Model architecture exploration

Requirements:
- Set your HF_TOKEN in config.py or as environment variable
- GPU recommended for model inference
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import gc

from config import (
    HF_TOKEN, LLAMA_MODEL, PHI_MODEL, GEMMA_MODEL,
    QWEN_MODEL, MIXTRAL_MODEL, DEFAULT_DEVICE
)
from utils import (
    authenticate_huggingface, check_gpu_availability, 
    get_quantization_config, load_model_and_tokenizer,
    generate_text, cleanup_memory, get_model_memory_footprint
)


class ModelDemo:
    """Class to demonstrate various model functionalities."""
    
    def __init__(self):
        self.device = DEFAULT_DEVICE
        self._check_setup()
        self.models = {}
    
    def _check_setup(self):
        """Check GPU availability and authenticate."""
        check_gpu_availability()
        try:
            authenticate_huggingface()
        except ValueError as e:
            print(f"Authentication error: {e}")
            raise
    
    def load_model(self, model_name, use_quantization=True):
        """Load a model with optional quantization."""
        print(f"Loading {model_name}...")
        
        try:
            model, tokenizer = load_model_and_tokenizer(
                model_name, 
                use_quantization=use_quantization
            )
            
            memory_footprint = get_model_memory_footprint(model)
            print(f"✓ Model loaded successfully")
            print(f"  Memory footprint: {memory_footprint:,.1f} MB")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {e}")
            return None, None
    
    def explore_model_architecture(self, model):
        """Explore and display model architecture."""
        print("\nModel Architecture:")
        print("=" * 50)
        print(model)
        
        print(f"\nModel Configuration:")
        print(f"  Device: {next(model.parameters()).device}")
        print(f"  Dtype: {next(model.parameters()).dtype}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def generate_with_model(self, model_name, messages, max_new_tokens=80, use_streaming=True):
        """Generate text using a specific model."""
        print(f"\nGenerating text with {model_name}:")
        print("-" * 40)
        
        try:
            response = generate_text(
                model_name, 
                messages, 
                max_new_tokens=max_new_tokens,
                use_streaming=use_streaming
            )
            return response
        except Exception as e:
            print(f"Generation failed: {e}")
            return None
    
    def compare_models(self, messages, model_list):
        """Compare text generation across different models."""
        print(f"\nModel Comparison:")
        print("=" * 30)
        print(f"Prompt: {messages}")
        print()
        
        results = {}
        
        for model_name in model_list:
            print(f"Testing {model_name}...")
            try:
                response = self.generate_with_model(
                    model_name, 
                    messages, 
                    max_new_tokens=50,
                    use_streaming=False
                )
                results[model_name] = response
                print(f"✓ {model_name} completed")
            except Exception as e:
                print(f"✗ {model_name} failed: {e}")
                results[model_name] = None
        
        return results
    
    def demonstrate_quantization(self, model_name):
        """Demonstrate the effect of quantization on memory usage."""
        print(f"\nQuantization Comparison for {model_name}:")
        print("=" * 45)
        
        # Load without quantization
        print("Loading without quantization...")
        model_no_quant, tokenizer = self.load_model(model_name, use_quantization=False)
        if model_no_quant is not None:
            memory_no_quant = get_model_memory_footprint(model_no_quant)
            cleanup_memory(model_no_quant, tokenizer)
        
        # Load with quantization
        print("Loading with quantization...")
        model_quant, tokenizer = self.load_model(model_name, use_quantization=True)
        if model_quant is not None:
            memory_quant = get_model_memory_footprint(model_quant)
            cleanup_memory(model_quant, tokenizer)
        
        if model_no_quant is not None and model_quant is not None:
            reduction = ((memory_no_quant - memory_quant) / memory_no_quant) * 100
            print(f"\nMemory Usage Comparison:")
            print(f"  Without quantization: {memory_no_quant:,.1f} MB")
            print(f"  With quantization:    {memory_quant:,.1f} MB")
            print(f"  Memory reduction:     {reduction:.1f}%")


def main():
    """Main function to run model examples."""
    print("Week 3 Day 4 - Models Demo")
    print("=" * 30)
    
    try:
        demo = ModelDemo()
    except ValueError:
        return
    
    # Example messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
    ]
    
    # Available models (in order of preference based on size/compatibility)
    available_models = [
        PHI_MODEL,      # Smallest, most compatible
        GEMMA_MODEL,    # Small Google model
        QWEN_MODEL,     # Medium size
        LLAMA_MODEL,     # Larger model
    ]
    
    print("\n1. Model Architecture Exploration")
    print("-" * 35)
    
    # Load and explore the first available model
    for model_name in available_models:
        model, tokenizer = demo.load_model(model_name)
        if model is not None:
            demo.explore_model_architecture(model)
            cleanup_memory(model, tokenizer)
            break
    
    print("\n2. Quantization Demonstration")
    print("-" * 30)
    
    # Demonstrate quantization with a smaller model
    demo.demonstrate_quantization(PHI_MODEL)
    
    print("\n3. Text Generation Examples")
    print("-" * 28)
    
    # Generate with different models
    for model_name in available_models[:2]:  # Test first 2 models
        demo.generate_with_model(model_name, messages, max_new_tokens=80)
    
    print("\n4. Model Comparison")
    print("-" * 20)
    
    # Compare models
    comparison_results = demo.compare_models(messages, available_models[:2])
    
    print("\nAll model examples completed!")


if __name__ == "__main__":
    main()
