import torch
from custom_GPT.params import device
from custom_GPT.BigramLM import m
from data.process_data import decode

context = torch.zeros((1,1), dtype=torch.long, device=device)
generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_chars)