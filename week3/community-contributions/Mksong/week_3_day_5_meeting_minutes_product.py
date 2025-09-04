# -*- coding: utf-8 -*-
"""
Meeting Minutes Generator
- Transcribes an audio file of a meeting.
- Generates meeting minutes in markdown: summary, discussion points, takeaways, action items.
"""

import os
import torch
from openai import OpenAI
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    BitsAndBytesConfig,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq,
    pipeline,
)

AUDIO_MODEL_OPENAI = "whisper-1"
LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
AUDIO_MODEL_HF = "openai/whisper-medium"
audio_filename = "denver_extract.mp3"

openai_api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=openai_api_key)

def transcribe_audio_openai(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcription = openai.audio.transcriptions.create(
            model=AUDIO_MODEL_OPENAI,
            file=audio_file,
            response_format="text"
        )
    return transcription

def generate_minutes(transcript, model_name=LLAMA_MODEL):
    system_message = (
        "You are an assistant that produces minutes of meetings from transcripts, "
        "with summary, key discussion points, takeaways and action items with owners, in markdown."
    )
    user_prompt = (
        f"Below is an extract transcript of a Denver council meeting. "
        f"Please write minutes in markdown, including a summary with attendees, location and date; "
        f"discussion points; takeaways; and action items with owners.\n{transcript}"
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quant_config)
    streamer = TextStreamer(tokenizer)
    outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)
    response = tokenizer.decode(outputs[0])
    return response

def transcribe_audio_hf(audio_path, model_name=AUDIO_MODEL_HF):
    speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    speech_model.to('cuda')
    processor = AutoProcessor.from_pretrained(model_name)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=speech_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device='cuda',
    )
    result = pipe(audio_path)
    return result["text"]

if __name__ == "__main__":
    transcript = transcribe_audio_openai(audio_filename)
    print("Transcript:\n", transcript)
    meeting_minutes = generate_minutes(transcript)
    print("\nMeeting Minutes:\n", meeting_minutes)