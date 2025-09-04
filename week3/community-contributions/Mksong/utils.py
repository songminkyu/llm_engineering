"""
Utility functions for Mksong's Week 3 examples.
"""

import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from huggingface_hub import login
from config import HF_TOKEN, DEFAULT_DEVICE, USE_QUANTIZATION, QUANTIZATION_TYPE, QUANTIZATION_BITS


def authenticate_huggingface():
    """Authenticate with HuggingFace Hub."""
    if HF_TOKEN == 'your_huggingface_token_here':
        raise ValueError("Please set your HF_TOKEN in config.py or environment variables")
    login(HF_TOKEN, add_to_git_credential=True)
    print("Successfully authenticated with HuggingFace Hub")


def get_quantization_config():
    """Get quantization configuration for memory-efficient model loading."""
    if not USE_QUANTIZATION:
        return None
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=QUANTIZATION_TYPE
    )


def load_model_and_tokenizer(model_name, use_quantization=True):
    """Load model and tokenizer with proper configuration."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    quant_config = get_quantization_config() if use_quantization else None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        quantization_config=quant_config
    )
    
    return model, tokenizer


def generate_text(model_name, messages, max_new_tokens=80, use_streaming=True):
    """Generate text using a model with optional streaming."""
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    inputs = tokenizer.apply_chat_template(
        messages, 
        return_tensors="pt", 
        add_generation_prompt=True
    ).to(DEFAULT_DEVICE)
    
    if use_streaming:
        streamer = TextStreamer(tokenizer)
        outputs = model.generate(
            inputs, 
            max_new_tokens=max_new_tokens, 
            streamer=streamer
        )
    else:
        outputs = model.generate(inputs, max_new_tokens=max_new_tokens)
    
    response = tokenizer.decode(outputs[0])
    
    # Clean up memory
    cleanup_memory(model, inputs, tokenizer, outputs)
    
    return response


def cleanup_memory(*objects):
    """Clean up GPU memory by deleting objects and running garbage collection."""
    for obj in objects:
        if obj is not None:
            del obj
    gc.collect()
    torch.cuda.empty_cache()


def check_gpu_availability():
    """Check if GPU is available and print memory info."""
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("GPU not available, using CPU")
        return False


def get_model_memory_footprint(model):
    """Get model memory footprint in MB."""
    if hasattr(model, 'get_memory_footprint'):
        return model.get_memory_footprint() / 1e6
    return 0
