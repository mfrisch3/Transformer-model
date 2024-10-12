import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from PositionalEncoding import PositionalEncoding
from MultiHeadAttention import MultiHeadAttention
from FeedForwardNetwork import FeedForwardNetwork



# Importing necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pretrained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set model to evaluation mode
#model.eval()
# Fix the padding issue by using eos_token as pad_token
tokenizer.pad_token = tokenizer.eos_token
# Loop to keep taking prompts and generating responses
while True:
    # Take a prompt input from the terminal
    prompt = input("Enter your prompt: ")

    # Break the loop if user types "exit"
    if prompt.lower() == "exit":
        break

     # Tokenize the user input and set attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

   # Generate output with sampling techniques to avoid repetition
    outputs = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask'], 
        max_length=100, 
        do_sample=True, 
        top_k=50, 
        top_p=0.95, 
        temperature=0.7
    )

    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    print("Generated Text: ", generated_text)