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
model.eval()

# Loop to keep taking prompts and generating responses
while True:
    # Take a prompt input from the terminal
    prompt = input("Enter your prompt: ")

    # Break the loop if user types "exit"
    if prompt.lower() == "exit":
        break

    # Tokenize the input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate output using the model (set a max output length)
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1)

    # Decode the output to human-readable text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print the generated text
    print("Generated Text: ", generated_text)
