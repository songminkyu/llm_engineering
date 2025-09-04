"""
Week 3 Day 2 - HuggingFace Pipelines
Refactored version for local environment

This script demonstrates various HuggingFace pipelines for NLP tasks:
- Sentiment Analysis
- Named Entity Recognition
- Question Answering
- Text Summarization
- Translation
- Zero-shot Classification
- Text Generation
- Image Generation
- Audio Generation

Requirements:
- Set your HF_TOKEN in config.py or as environment variable
- GPU recommended for best performance
"""

import torch
from transformers import pipeline
from diffusers import DiffusionPipeline
from datasets import load_dataset
import soundfile as sf
from PIL import Image

from config import HF_TOKEN, DEFAULT_DEVICE
from utils import authenticate_huggingface, check_gpu_availability, cleanup_memory


class PipelineDemo:
    """Class to demonstrate various HuggingFace pipelines."""
    
    def __init__(self):
        self.device = DEFAULT_DEVICE
        self._check_setup()
    
    def _check_setup(self):
        """Check GPU availability and authenticate."""
        check_gpu_availability()
        try:
            authenticate_huggingface()
        except ValueError as e:
            print(f"Authentication error: {e}")
            raise
    
    def sentiment_analysis(self, text):
        """Perform sentiment analysis on text."""
        print("Running sentiment analysis...")
        classifier = pipeline("sentiment-analysis", device=self.device)
        result = classifier(text)
        print(f"Text: {text}")
        print(f"Result: {result}")
        cleanup_memory(classifier)
        return result
    
    def named_entity_recognition(self, text):
        """Extract named entities from text."""
        print("Running named entity recognition...")
        ner = pipeline("ner", grouped_entities=True, device=self.device)
        result = ner(text)
        print(f"Text: {text}")
        print(f"Entities: {result}")
        cleanup_memory(ner)
        return result
    
    def question_answering(self, question, context):
        """Answer questions based on context."""
        print("Running question answering...")
        question_answerer = pipeline("question-answering", device=self.device)
        result = question_answerer(question=question, context=context)
        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"Answer: {result}")
        cleanup_memory(question_answerer)
        return result
    
    def text_summarization(self, text, max_length=50, min_length=25):
        """Summarize long text."""
        print("Running text summarization...")
        summarizer = pipeline("summarization", device=self.device)
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        print(f"Original text: {text[:100]}...")
        print(f"Summary: {summary[0]['summary_text']}")
        cleanup_memory(summarizer)
        return summary
    
    def translation_en_to_fr(self, text):
        """Translate English to French."""
        print("Running English to French translation...")
        translator = pipeline("translation_en_to_fr", device=self.device)
        result = translator(text)
        print(f"English: {text}")
        print(f"French: {result[0]['translation_text']}")
        cleanup_memory(translator)
        return result
    
    def translation_en_to_es(self, text, model="Helsinki-NLP/opus-mt-en-es"):
        """Translate English to Spanish with specific model."""
        print("Running English to Spanish translation...")
        translator = pipeline("translation_en_to_es", model=model, device=self.device)
        result = translator(text)
        print(f"English: {text}")
        print(f"Spanish: {result[0]['translation_text']}")
        cleanup_memory(translator)
        return result
    
    def zero_shot_classification(self, text, candidate_labels):
        """Classify text into predefined categories."""
        print("Running zero-shot classification...")
        classifier = pipeline("zero-shot-classification", device=self.device)
        result = classifier(text, candidate_labels=candidate_labels)
        print(f"Text: {text}")
        print(f"Candidate labels: {candidate_labels}")
        print(f"Classification: {result}")
        cleanup_memory(classifier)
        return result
    
    def text_generation(self, prompt, max_length=50):
        """Generate text from a prompt."""
        print("Running text generation...")
        generator = pipeline("text-generation", device=self.device)
        result = generator(prompt, max_length=max_length)
        print(f"Prompt: {prompt}")
        print(f"Generated: {result[0]['generated_text']}")
        cleanup_memory(generator)
        return result
    
    def image_generation(self, prompt, output_filename="generated_image.png"):
        """Generate image from text prompt."""
        print("Running image generation...")
        image_gen = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2",
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to(self.device)
        
        image = image_gen(prompt=prompt).images[0]
        image.save(output_filename)
        print(f"Image saved as {output_filename}")
        cleanup_memory(image_gen)
        return image
    
    def audio_generation(self, text, output_filename="speech.wav"):
        """Generate speech from text."""
        print("Running audio generation...")
        synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device=self.device)
        
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        
        speech = synthesiser(
            text, 
            forward_params={"speaker_embeddings": speaker_embedding}
        )
        
        sf.write(output_filename, speech["audio"], samplerate=speech["sampling_rate"])
        print(f"Audio saved as {output_filename}")
        cleanup_memory(synthesiser, speaker_embedding)
        return speech


def main():
    """Main function to run all pipeline examples."""
    print("Week 3 Day 2 - HuggingFace Pipelines Demo")
    print("=" * 50)
    
    try:
        demo = PipelineDemo()
    except ValueError:
        return
    
    # Example texts and prompts
    sentiment_text = "I'm super excited to be on the way to LLM mastery!"
    ner_text = "Barack Obama was the 44th president of the United States."
    qa_question = "Who was the 44th president of the United States?"
    qa_context = "Barack Obama was the 44th president of the United States."
    
    summary_text = """The Hugging Face transformers library is an incredibly versatile and powerful tool for natural language processing (NLP).
    It allows users to perform a wide range of tasks such as text classification, named entity recognition, and question answering, among others.
    It's an extremely popular library that's widely used by the open-source data science community.
    It lowers the barrier to entry into the field by providing Data Scientists with a productive, convenient way to work with transformer models."""
    
    translation_text = "The Data Scientists were truly amazed by the power and simplicity of the HuggingFace pipeline API."
    classification_text = "Hugging Face's Transformers library is amazing!"
    classification_labels = ["technology", "sports", "politics"]
    generation_prompt = "If there's one thing I want you to remember about using HuggingFace pipelines, it's"
    image_prompt = "A class of Data Scientists learning about AI, in the surreal style of Salvador Dali"
    audio_text = "Hi to an artificial intelligence engineer, on the way to mastery!"
    
    print("\n1. Sentiment Analysis")
    print("-" * 20)
    demo.sentiment_analysis(sentiment_text)
    
    print("\n2. Named Entity Recognition")
    print("-" * 30)
    demo.named_entity_recognition(ner_text)
    
    print("\n3. Question Answering")
    print("-" * 20)
    demo.question_answering(qa_question, qa_context)
    
    print("\n4. Text Summarization")
    print("-" * 22)
    demo.text_summarization(summary_text)
    
    print("\n5. Translation (English to French)")
    print("-" * 35)
    demo.translation_en_to_fr(translation_text)
    
    print("\n6. Translation (English to Spanish)")
    print("-" * 36)
    demo.translation_en_to_es(translation_text)
    
    print("\n7. Zero-shot Classification")
    print("-" * 25)
    demo.zero_shot_classification(classification_text, classification_labels)
    
    print("\n8. Text Generation")
    print("-" * 18)
    demo.text_generation(generation_prompt)
    
    print("\n9. Image Generation")
    print("-" * 19)
    demo.image_generation(image_prompt)
    
    print("\n10. Audio Generation")
    print("-" * 19)
    demo.audio_generation(audio_text)
    
    print("\nAll pipeline examples completed!")


if __name__ == "__main__":
    main()
