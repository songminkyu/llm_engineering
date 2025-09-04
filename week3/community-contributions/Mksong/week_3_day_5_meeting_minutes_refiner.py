# -*- coding: utf-8 -*-
"""
Meeting Minutes Refiner
- Refines and summarizes meeting minutes from a transcript.
"""

import os
import torch
from openai import OpenAI
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
openai_api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=openai_api_key)

def refine_minutes(transcript, model_name=LLAMA_MODEL):
    system_message = (
        "You are an assistant that refines meeting minutes from transcripts, "
        "providing a clear summary, discussion points, takeaways and actionable items in markdown."
    )
    user_prompt = (
        f"Please refine the following meeting transcript into structured meeting minutes in markdown:\n{transcript}"
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
    outputs = model.generate(inputs, max_new_tokens=1500)
    response = tokenizer.decode(outputs[0])
    return response

if __name__ == "__main__":
    sample_transcript = "Your meeting transcript text here."
    refined_minutes = refine_minutes(sample_transcript)
    print(refined_minutes)