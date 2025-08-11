import torch
import torch.nn as nn
from torch.nn import functional as F

from custom_GPT.params import batch_size, block_size, n_embd, n_layer, n_head, dropout, device
#TODO: Import vocab_size from the data preprocessing module if needed
# from data.process_data import vocab_size

# Define the Head, MultiHeadAttention, FeedForward, Block, and GPTLanguageModel classes
class Head(nn.Module):
  """ one head of self attention """

  def __init__(self, head_size):
    super().__init__()
    #keys are what other tokens look for
    self.key = nn.Linear(n_embd, head_size, bias=False)  # a linear transformation only, here all heads try to extract different info
    #queries are what a token uses to find the other relevent tokens
    self.query = nn.Linear(n_embd, head_size, bias=False) #same goes here
    #values contain the actual information that gets aggregated
    self.value = nn.Linear(n_embd, head_size, bias=False)
    #casual masking to ensure a token can only attend to previous tokens in the sequence, not the future ones
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    #initialize the dropout layer to prevent the overfitting
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape # B: Batch size, T: Sequence length, C: Embedding dimension
    k = self.key(x)   # (B,T,head size)
    q = self.query(x) # (B,T,head size)
    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T) , further scaled by square root of head_size
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T), This ensures that the softmax later will result in a probability of zero for these positions.
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei)  #apply dropout with dropout=0.2
    # perform the weighted aggregation of the values
    v = self.value(x) # (B,T,hs)
    out = wei @ v # (B, T, T) @ (B, T, hs)hs->headsize
    return out

class MultiHeadAttention(nn.Module):
  """multiple heads of self attention in parallel"""
  def __init__(self, num_heads, head_size):
    #Calling the parent class constructor (nn.Module).
    super().__init__()
    # list(container) that contains all the heads
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    # final linear layer to project the concatenated output of all heads back to original dimension (n_embd)
    self.proj = nn.Linear(n_embd, n_embd)
    # dropout layer for the output
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    # concatenate the outputs of all heads along the last dimension(dim=-1). Resulting tensor has shape (B, T, num_heads*head_size) -> (B, T, num_embd)
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    # apply linear projection and dropout
    out = self.dropout(self.proj(out))
    return out


class FeedFoward(nn.Module):
  """ a simple linear layer followed by a non-linearity """

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),  # Expand
      nn.ReLU(),                      # Introduce non linearity
      nn.Linear(4 * n_embd, n_embd),  # Contract / scale down
      nn.Dropout(dropout),            # Dropout
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  """ Transformer block: communication followed by computation """

  def __init__(self, n_embd, n_head):
    # n_embd: embedding dimension, n_head: the number of heads we'd like
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedFoward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)  # First layer of normalization (To normalize the inputs across the embedding dimensions for each sample in batch)
    self.ln2 = nn.LayerNorm(n_embd)  # second layer of noramalization

  def forward(self, x):
    # y = self.sa(self.ln1(x))
    # x = x + y
    # y = self.ffwd(self.ln2(x))
    # x = x + y
    # return x
    y = self.sa(x)
    x = self.ln1(x + y)  # Residual connection (output of attention block is added to the original input)
    y = self.ffwd(x)
    x = self.ln2(x + y)
    return x


class GPTLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    # Create an embedding table that maps each token ID in vocabulary to a vector of size n_embd
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    # Create an embedding table taht maps each position in the sequence of vector size n_embd
    self.position_embedding_table = nn.Embedding(block_size, n_embd)

    #decoder blocks
    # Create a sequence of Block modules
    self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    # The Layer to project final embedding of each token to the size of vocabulary.
    self.lm_head = nn.Linear(n_embd, vocab_size) # The output is logits (raw prediction scores) for next token.
    # Initialize all the weights
    self.apply(self._init_weights)

  def _init_weights(self, module):
    """ Helper function to initialize the weights """

    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward_pass(self, index, targets=None):
    """ The forward pass of entire module """

    B, T = index.shape #Batch, Time
    tok_em = self.token_embedding_table(index) # (B,T,C), looks up the embeddings for the input token ID's
    pos_em = self.position_embedding_table(torch.arange(T, device=device)) # (T,C) , looks up the positional embeddings for the sequence of length T.
    x = tok_em + pos_em # (B,T,C)
    x = self.blocks(x) # (B,T,C)
    x = self.ln_f(x) # (B,T,C)
    logits = self.lm_head(x) # (B,T,vocab_size)

    if targets is not None: # Targets will be not None during training.
      B, T, C = logits.shape #Batch, Time, Channel(vocab size)
      # Reshape logits and targets tensors to 2D and 1D shape for easy loss calculation.
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    else:
      loss = None

    return logits, loss

  def generate(self, index, max_new_tokens):
    #index is (B,T) array of indices in the current context

    for _ in range(max_new_tokens):  # To get the prdictions based on the token size
      # if the context is too long, crop it
      index_cond = index[:, -block_size:]
      # get the predictions
      logits ,loss = self.forward_pass(index_cond) # focuss on the last time stamp
      # pluck the last time step because only the last token's output is relevant to predict next token
      logits = logits[:, -1, :] # becomes (B,C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # note that dims=-1
      # sample from the distribution
      index_next = torch.multinomial(probs, num_samples=1) # sample from the distribution
      # append sampled index to the running sequence
      index = torch.cat((index, index_next), dim=1) # append the sampled ata to the running sequence
    return index

# model = GPTLanguageModel(vocab_size)
# m = model.to(device)  # to use gpu if availbale

# context = torch.zeros((1,1), dtype=torch.long, device=device)
# generated_chars = decode(m.generate(context, max_new_tokens=500)[0].tolist())
# print(generated_chars)