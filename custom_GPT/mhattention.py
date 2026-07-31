import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1, context_len=1024, qkv_bias=True):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.num_heads  = num_heads
        self.head_size  = input_dim // num_heads
        self.dropout_p  = dropout   # passed to F.scaled_dot_product_attention at runtime

        self.W_query    = nn.Linear(input_dim, input_dim, bias=qkv_bias)
        self.W_key      = nn.Linear(input_dim, input_dim, bias=qkv_bias)
        self.W_value    = nn.Linear(input_dim, input_dim, bias=qkv_bias)
        self.linear_out = nn.Linear(input_dim, input_dim)

        # NOTE: No manual mask buffer needed.
        # F.scaled_dot_product_attention handles causal masking internally via
        # Flash Attention on CUDA — never allocates the O(N²) score matrix.

    def forward(self, x):
        batch, num_tokens, input_dim = x.shape

        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        # Reshape to (batch, num_heads, num_tokens, head_size)
        queries = Q.view(batch, num_tokens, self.num_heads, self.head_size).transpose(1, 2)
        keys    = K.view(batch, num_tokens, self.num_heads, self.head_size).transpose(1, 2)
        values  = V.view(batch, num_tokens, self.num_heads, self.head_size).transpose(1, 2)

        # Flash Attention — memory-efficient, O(N) instead of O(N²).
        # is_causal=True applies the causal mask automatically.
        # dropout_p is only active during training.
        context = F.scaled_dot_product_attention(
            queries, keys, values,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )

        # Merge heads back: (batch, num_tokens, input_dim)
        context = context.transpose(1, 2).contiguous().view(batch, num_tokens, input_dim)
        return self.linear_out(context)