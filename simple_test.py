#!/usr/bin/env python3
"""
Simple test to verify basic model generation works
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Set pad token if not exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move to MPS if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
model.eval()

print(f"Model loaded on device: {device}")

# Test generation
prompt = "Hello, how are you?"
print(f"Testing generation with prompt: '{prompt}'")

inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"Input tokens: {inputs['input_ids']}")
print(f"Input shape: {inputs['input_ids'].shape}")

with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs.get('attention_mask'),
        max_new_tokens=5,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True
    )

print(f"Output tokens: {outputs}")
print(f"Output shape: {outputs.shape}")

# Decode output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated text: '{generated_text}'")

# Extract new tokens
new_tokens = outputs[0][inputs['input_ids'].size(-1):]
print(f"New tokens: {new_tokens}")
new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(f"New text only: '{new_text}'")

print("✅ Basic generation test completed successfully!")
