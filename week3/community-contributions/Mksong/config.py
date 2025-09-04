"""
Configuration file for Mksong's Week 3 examples.
Replace the placeholder values with your actual API keys and preferences.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# HuggingFace Configuration
HF_TOKEN = os.getenv('HF_TOKEN', 'your_huggingface_token_here')

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your_openai_api_key_here')

# Model Configuration
DEFAULT_DEVICE = os.getenv('DEFAULT_DEVICE', 'cuda')
DEFAULT_TORCH_DTYPE = os.getenv('DEFAULT_TORCH_DTYPE', 'torch.float16')

# Audio Configuration
AUDIO_MODEL_OPENAI = os.getenv('AUDIO_MODEL_OPENAI', 'whisper-1')
AUDIO_MODEL_HF = os.getenv('AUDIO_MODEL_HF', 'openai/whisper-medium')

# Model Names
LLAMA_MODEL = os.getenv('LLAMA_MODEL', 'meta-llama/Meta-Llama-3.1-8B-Instruct')
PHI_MODEL = os.getenv('PHI_MODEL', 'microsoft/Phi-3-mini-4k-instruct')
GEMMA_MODEL = os.getenv('GEMMA_MODEL', 'google/gemma-2-2b-it')
QWEN_MODEL = os.getenv('QWEN_MODEL', 'Qwen/Qwen2-7B-Instruct')
MIXTRAL_MODEL = os.getenv('MIXTRAL_MODEL', 'mistralai/Mixtral-8x7B-Instruct-v0.1')
STARCODER_MODEL = os.getenv('STARCODER_MODEL', 'bigcode/starcoder2-3b')

# Quantization Configuration
USE_QUANTIZATION = os.getenv('USE_QUANTIZATION', 'true').lower() == 'true'
QUANTIZATION_TYPE = os.getenv('QUANTIZATION_TYPE', 'nf4')
QUANTIZATION_BITS = int(os.getenv('QUANTIZATION_BITS', '4'))

# File paths
AUDIO_FILENAME = os.getenv('AUDIO_FILENAME', 'denver_extract.mp3')
