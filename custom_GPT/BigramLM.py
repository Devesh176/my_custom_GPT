import torch
import torch.nn as nn
from torch.nn import functional as F

from  custom_GPT.params import batch_size, block_size, n_embd, n_layer, n_head, dropout,device
# TODO: Import vocab_size from the data preprocessing module if needed
# from data.process_data import vocab_size, encode, decode

# Define the BigramLanguageModel class
class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward_pass(self, index, targets=None):
    logits = self.token_embedding_table(index)

    if targets is not None:
      B, T, C = logits.shape #Batch, Time, Channel(vocab size)
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    else:
      loss = None

    return logits, loss

  def generate(self, index, max_new_tokens):
    #index is (B,T) array of indices in the current context

    for _ in range(max_new_tokens):  # To get the prdictions based on the token size
      logits ,loss = self.forward_pass(index) # focuss on the last time stamp
      logits = logits[:, -1, :] # becomes (B,C)
      probs = F.softmax(logits, dim=-1) # note that dims=-1
      index_next = torch.multinomial(probs, num_samples=1) # sample from the distribution
      index = torch.cat((index, index_next), dim=1) # append the sampled ata to the running sequence
    return index

# model = BigramLanguageModel(vocab_size)
# m = model.to(device)  # to use gpu if availbale

# context = torch.zeros((1,1), dtype=torch.long, device=device)
# generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
# print(generated_chars)
