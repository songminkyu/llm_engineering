"""
Week 3 Day 5 - Meeting Minutes Product
Refactored version for local environment

This script demonstrates:
1. Audio transcription using OpenAI Whisper or HuggingFace models
2. Meeting minutes generation using LLM
3. Integration of audio processing and text generation

Requirements:
- Set your HF_TOKEN and OPENAI_API_KEY in config.py or as environment variables
- Audio file for transcription
- GPU recommended for best performance
"""

import os
import requests
from openai import OpenAI
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TextStreamer, 
    BitsAndBytesConfig, AutoModelForSpeechSeq2Seq, AutoProcessor,
    pipeline
)
import torch
from config import (
    HF_TOKEN, OPENAI_API_KEY, LLAMA_MODEL, AUDIO_MODEL_OPENAI, 
    AUDIO_MODEL_HF, AUDIO_FILENAME, DEFAULT_DEVICE
)
from utils import (
    authenticate_huggingface, check_gpu_availability, 
    get_quantization_config, cleanup_memory
)


class MeetingMinutesGenerator:
    """Class to generate meeting minutes from audio files."""
    
    def __init__(self):
        self.device = DEFAULT_DEVICE
        self._check_setup()
        self._setup_clients()
    
    def _check_setup(self):
        """Check GPU availability and authenticate."""
        check_gpu_availability()
        try:
            authenticate_huggingface()
        except ValueError as e:
            print(f"Authentication error: {e}")
            raise
    
    def _setup_clients(self):
        """Setup OpenAI and other clients."""
        if OPENAI_API_KEY == 'your_openai_api_key_here':
            print("Warning: OpenAI API key not set. OpenAI Whisper will not be available.")
            self.openai_client = None
        else:
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            print("✓ OpenAI client initialized")
    
    def transcribe_with_openai(self, audio_file_path):
        """Transcribe audio using OpenAI Whisper."""
        if self.openai_client is None:
            raise ValueError("OpenAI client not initialized. Please set OPENAI_API_KEY.")
        
        print("Transcribing with OpenAI Whisper...")
        
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model=AUDIO_MODEL_OPENAI,
                    file=audio_file,
                    response_format="text"
                )
            print("✓ Transcription completed")
            return transcription
        except Exception as e:
            print(f"✗ OpenAI transcription failed: {e}")
            raise
    
    def transcribe_with_huggingface(self, audio_file_path):
        """Transcribe audio using HuggingFace Whisper model."""
        print("Transcribing with HuggingFace Whisper...")
        
        try:
            # Load model and processor
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                AUDIO_MODEL_HF,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            model.to(self.device)
            
            processor = AutoProcessor.from_pretrained(AUDIO_MODEL_HF)
            
            # Create pipeline
            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=torch.float16,
                device=self.device,
            )
            
            # Transcribe
            result = pipe(audio_file_path)
            transcription = result["text"]
            
            print("✓ Transcription completed")
            
            # Clean up
            cleanup_memory(model, processor, pipe)
            
            return transcription
            
        except Exception as e:
            print(f"✗ HuggingFace transcription failed: {e}")
            raise
    
    def generate_meeting_minutes(self, transcription, use_streaming=True):
        """Generate meeting minutes from transcription using LLM."""
        print("Generating meeting minutes...")
        
        # Prepare messages
        system_message = (
            "You are an assistant that produces minutes of meetings from transcripts, "
            "with summary, key discussion points, takeaways and action items with owners, in markdown."
        )
        
        user_prompt = (
            f"Below is an extract transcript of a Denver council meeting. "
            f"Please write minutes in markdown, including a summary with attendees, "
            f"location and date; discussion points; takeaways; and action items with owners.\n"
            f"{transcription}"
        )
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
        tokenizer.pad_token = tokenizer.eos_token
        
        quant_config = get_quantization_config()
        model = AutoModelForCausalLM.from_pretrained(
            LLAMA_MODEL, 
            device_map="auto", 
            quantization_config=quant_config
        )
        
        # Prepare inputs
        inputs = tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt"
        ).to(self.device)
        
        # Generate with optional streaming
        if use_streaming:
            streamer = TextStreamer(tokenizer)
            outputs = model.generate(
                inputs, 
                max_new_tokens=2000, 
                streamer=streamer
            )
        else:
            outputs = model.generate(inputs, max_new_tokens=2000)
        
        response = tokenizer.decode(outputs[0])
        
        # Clean up
        cleanup_memory(model, inputs, tokenizer, outputs)
        
        return response
    
    def process_meeting(self, audio_file_path, use_openai=True, use_streaming=True):
        """Complete meeting minutes generation pipeline."""
        print("Meeting Minutes Generation Pipeline")
        print("=" * 40)
        
        # Check if audio file exists
        if not os.path.exists(audio_file_path):
            print(f"Error: Audio file not found at {audio_file_path}")
            print("Please provide a valid audio file path.")
            return None
        
        try:
            # Step 1: Transcribe audio
            if use_openai and self.openai_client is not None:
                transcription = self.transcribe_with_openai(audio_file_path)
            else:
                transcription = self.transcribe_with_huggingface(audio_file_path)
            
            print(f"\nTranscription preview: {transcription[:200]}...")
            
            # Step 2: Generate meeting minutes
            minutes = self.generate_meeting_minutes(transcription, use_streaming)
            
            return {
                'transcription': transcription,
                'minutes': minutes
            }
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            return None
    
    def save_results(self, results, output_dir="output"):
        """Save transcription and minutes to files."""
        if results is None:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save transcription
        transcription_file = os.path.join(output_dir, "transcription.txt")
        with open(transcription_file, "w", encoding="utf-8") as f:
            f.write(results['transcription'])
        print(f"✓ Transcription saved to {transcription_file}")
        
        # Save minutes
        minutes_file = os.path.join(output_dir, "meeting_minutes.md")
        with open(minutes_file, "w", encoding="utf-8") as f:
            f.write(results['minutes'])
        print(f"✓ Meeting minutes saved to {minutes_file}")


def download_sample_audio():
    """Download sample audio file if not present."""
    audio_file = AUDIO_FILENAME
    
    if os.path.exists(audio_file):
        print(f"Audio file already exists: {audio_file}")
        return audio_file
    
    print("Sample audio file not found.")
    print("Please download the Denver meeting extract from:")
    print("https://drive.google.com/file/d/1N_kpSojRR5RYzupz6nqM8hMSoEF_R7pU/view?usp=sharing")
    print(f"Or provide your own audio file and update AUDIO_FILENAME in config.py")
    
    return None


def main():
    """Main function to run the meeting minutes generator."""
    print("Week 3 Day 5 - Meeting Minutes Generator")
    print("=" * 45)
    
    try:
        generator = MeetingMinutesGenerator()
    except ValueError:
        return
    
    # Check for audio file
    audio_file = download_sample_audio()
    if audio_file is None:
        print("\nPlease provide an audio file to continue.")
        return
    
    print(f"\nUsing audio file: {audio_file}")
    
    # Process meeting
    results = generator.process_meeting(
        audio_file, 
        use_openai=True,  # Set to False to use HuggingFace Whisper
        use_streaming=True
    )
    
    if results:
        # Save results
        generator.save_results(results)
        
        print("\nMeeting minutes generation completed!")
        print("Check the 'output' directory for results.")
    else:
        print("Meeting minutes generation failed.")


if __name__ == "__main__":
    main()
