import torch
import torch.nn as nn


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device))
            * (x + 0.044715 * x.pow(3))
        ))


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.gelu = GELU()
        self.fc2  = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.eps = eps
        # BUG FIX: no device= argument; .to(device) on the parent module handles placement
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.bias  = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = x.var(-1, keepdim=True, unbiased=False)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.scale * x + self.bias