import torch

batch_size = 128  # hyper parameters
block_size = 64  # hyper parameters
max_iters = 500
eval_iters = 100
learning_rate = 3e-4
n_embd = 384 # how long the embedding vector will be
n_layer = 8
n_head = 8
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'