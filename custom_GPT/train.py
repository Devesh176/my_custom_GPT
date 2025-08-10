import pandas as pd
import numpy as np
import torch
from custom_GPT.params import batch_size, block_size, n_embd, n_layer, n_head, dropout, device, eval_iters, max_iters, learning_rate
from custom_GPT.GPT_model import model
from utils.utils import get_batch, estimate_loss
from data.process_data import preprocess_data


# create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
df = pd.read_parquet("hf://datasets/KisanVaani/agriculture-qa-english-only/data/train-00000-of-00001.parquet")
data, vocab_size, encode, decode = preprocess_data(df)

for iter in range(max_iters):
  if iter % eval_iters == 0:
    losses = estimate_loss()
    print(f"step: {iter}, train_loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}")

  # sample of batch of data
  xb, yb = get_batch('train',data=data)

  #evaluate the loss
  logits, loss = model.forward_pass(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

print(loss.item())