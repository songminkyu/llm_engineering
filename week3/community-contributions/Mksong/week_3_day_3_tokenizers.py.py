"""
Week 3 Day 3 - Tokenizers
Refactored version for local environment

This script demonstrates:
1. Working with different tokenizers (Llama, Phi3, Qwen2, StarCoder2)
2. Token encoding and decoding
3. Chat template formatting
4. Model comparison

Requirements:
- Set your HF_TOKEN in config.py or as environment variable
- No GPU required for tokenizer operations
"""

from transformers import AutoTokenizer
from config import (
    HF_TOKEN, LLAMA_MODEL, PHI_MODEL, QWEN_MODEL, STARCODER_MODEL
)
from utils import authenticate_huggingface


class TokenizerDemo:
    """Class to demonstrate various tokenizer functionalities."""
    
    def __init__(self):
        self._check_setup()
        self.tokenizers = {}
        self._load_tokenizers()
    
    def _check_setup(self):
        """Check authentication."""
        try:
            authenticate_huggingface()
        except ValueError as e:
            print(f"Authentication error: {e}")
            raise
    
    def _load_tokenizers(self):
        """Load all tokenizers."""
        print("Loading tokenizers...")
        
        # Load Llama tokenizer
        try:
            self.tokenizers['llama'] = AutoTokenizer.from_pretrained(
                LLAMA_MODEL, 
                trust_remote_code=True
            )
            print(f"✓ Loaded Llama tokenizer")
        except Exception as e:
            print(f"✗ Failed to load Llama tokenizer: {e}")
        
        # Load Phi3 tokenizer
        try:
            self.tokenizers['phi3'] = AutoTokenizer.from_pretrained(PHI_MODEL)
            print(f"✓ Loaded Phi3 tokenizer")
        except Exception as e:
            print(f"✗ Failed to load Phi3 tokenizer: {e}")
        
        # Load Qwen2 tokenizer
        try:
            self.tokenizers['qwen2'] = AutoTokenizer.from_pretrained(QWEN_MODEL)
            print(f"✓ Loaded Qwen2 tokenizer")
        except Exception as e:
            print(f"✗ Failed to load Qwen2 tokenizer: {e}")
        
        # Load StarCoder2 tokenizer
        try:
            self.tokenizers['starcoder2'] = AutoTokenizer.from_pretrained(
                STARCODER_MODEL,
                trust_remote_code=True
            )
            print(f"✓ Loaded StarCoder2 tokenizer")
        except Exception as e:
            print(f"✗ Failed to load StarCoder2 tokenizer: {e}")
    
    def analyze_text(self, text, tokenizer_name='llama'):
        """Analyze text with a specific tokenizer."""
        if tokenizer_name not in self.tokenizers:
            print(f"Tokenizer {tokenizer_name} not available")
            return
        
        tokenizer = self.tokenizers[tokenizer_name]
        
        print(f"\nAnalyzing text with {tokenizer_name.upper()} tokenizer:")
        print(f"Text: {text}")
        
        # Encode text to tokens
        tokens = tokenizer.encode(text)
        print(f"Tokens: {tokens}")
        print(f"Number of tokens: {len(tokens)}")
        
        # Decode tokens back to text
        decoded = tokenizer.decode(tokens)
        print(f"Decoded: {decoded}")
        
        # Batch decode
        batch_decoded = tokenizer.batch_decode(tokens)
        print(f"Batch decoded: {batch_decoded}")
        
        # Get vocabulary info
        vocab_size = len(tokenizer.get_vocab())
        added_vocab = tokenizer.get_added_vocab()
        print(f"Vocabulary size: {vocab_size}")
        print(f"Added vocabulary: {len(added_vocab)} tokens")
        
        return {
            'tokens': tokens,
            'decoded': decoded,
            'vocab_size': vocab_size,
            'added_vocab_size': len(added_vocab)
        }
    
    def compare_tokenizers(self, text):
        """Compare how different tokenizers handle the same text."""
        print(f"\nComparing tokenizers on text: '{text}'")
        print("=" * 60)
        
        results = {}
        
        for name, tokenizer in self.tokenizers.items():
            try:
                tokens = tokenizer.encode(text)
                results[name] = {
                    'tokens': tokens,
                    'count': len(tokens),
                    'vocab_size': len(tokenizer.get_vocab())
                }
                print(f"{name.upper():12} | Tokens: {len(tokens):3} | Vocab: {len(tokenizer.get_vocab()):6,}")
            except Exception as e:
                print(f"{name.upper():12} | Error: {e}")
        
        return results
    
    def demonstrate_chat_templates(self, messages):
        """Demonstrate chat template formatting for different models."""
        print(f"\nChat Template Comparison:")
        print("=" * 40)
        print(f"Messages: {messages}")
        print()
        
        for name, tokenizer in self.tokenizers.items():
            try:
                if hasattr(tokenizer, 'apply_chat_template'):
                    template = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    print(f"{name.upper()} Template:")
                    print(f"{'─' * 20}")
                    print(template)
                    print()
                else:
                    print(f"{name.upper()}: No chat template support")
            except Exception as e:
                print(f"{name.upper()}: Error - {e}")
    
    def analyze_code_tokenization(self, code):
        """Analyze how code is tokenized (useful for StarCoder2)."""
        print(f"\nCode Tokenization Analysis:")
        print("=" * 30)
        print(f"Code:\n{code}")
        
        if 'starcoder2' in self.tokenizers:
            tokenizer = self.tokenizers['starcoder2']
            tokens = tokenizer.encode(code)
            
            print(f"\nToken breakdown:")
            for i, token in enumerate(tokens):
                decoded = tokenizer.decode(token)
                print(f"Token {i:2}: {token:6} = '{decoded}'")
        else:
            print("StarCoder2 tokenizer not available")


def main():
    """Main function to run tokenizer examples."""
    print("Week 3 Day 3 - Tokenizers Demo")
    print("=" * 40)
    
    try:
        demo = TokenizerDemo()
    except ValueError:
        return
    
    # Example texts
    sample_text = "I am excited to show Tokenizers in action to my LLM engineers"
    
    # Chat messages for template demonstration
    chat_messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
    ]
    
    # Code sample for StarCoder2
    code_sample = """
def hello_world(person):
  print("Hello", person)
"""
    
    print("\n1. Basic Tokenizer Analysis")
    print("-" * 30)
    demo.analyze_text(sample_text, 'llama')
    
    print("\n2. Tokenizer Comparison")
    print("-" * 25)
    demo.compare_tokenizers(sample_text)
    
    print("\n3. Chat Template Demonstration")
    print("-" * 32)
    demo.demonstrate_chat_templates(chat_messages)
    
    print("\n4. Code Tokenization Analysis")
    print("-" * 30)
    demo.analyze_code_tokenization(code_sample)
    
    print("\nAll tokenizer examples completed!")


if __name__ == "__main__":
    main()
