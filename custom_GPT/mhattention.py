import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1, context_len=1024, qkv_bias=True):
        super().__init__()
        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_size = input_dim // num_heads
        self.context_len = context_len

        self.W_query = nn.Linear(input_dim, input_dim, bias=qkv_bias)
        self.W_key   = nn.Linear(input_dim, input_dim, bias=qkv_bias)
        self.W_value = nn.Linear(input_dim, input_dim, bias=qkv_bias)
        self.linear_out = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

        # BUG FIX: register_buffer belongs in __init__, not forward()
        # Shape: (1, 1, context_len, context_len) — ready for 4-D masked_fill
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_len, context_len))
                 .view(1, 1, context_len, context_len)
        )

    def forward(self, x):
        batch, num_tokens, input_dim = x.shape

        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        # Split into heads: (batch, num_heads, num_tokens, head_size)
        queries = Q.view(batch, num_tokens, self.num_heads, self.head_size).transpose(1, 2)
        keys    = K.view(batch, num_tokens, self.num_heads, self.head_size).transpose(1, 2)
        values  = V.view(batch, num_tokens, self.num_heads, self.head_size).transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)  # (batch, heads, T, T)

        # BUG FIX: slice all 4 dims correctly
        mask_bool = self.mask.bool()[:, :, :num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill(~mask_bool, -torch.inf)
        attention_probs  = torch.softmax(attention_scores / (self.head_size ** 0.5), dim=-1)
        attention_probs  = self.dropout(attention_probs)

        context = attention_probs @ values  # (batch, heads, T, head_size)
        context = context.transpose(1, 2).contiguous().view(batch, num_tokens, input_dim)
        return self.linear_out(context)