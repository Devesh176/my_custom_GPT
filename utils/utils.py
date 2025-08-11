import torch
from custom_GPT.params import batch_size, block_size, n_embd, n_layer, n_head, dropout, device, eval_iters, max_iters
# from custom_GPT.GPT_model import model

def get_batch(split, data):
    n = int(0.8*len(data))
    train_data = data[:n]
    val_data = data[n:]
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device) # to use cuda if gpu is available
    return x, y


@torch.no_grad() #pytorch doesnt use gradient at all in this cell

def estimate_loss(model,data):
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split, data)  # Assuming data is passed or available globally
      logits, loss = model.forward_pass(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out