import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, head_size, dropout=0.1, context_len=1024, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        assert head_size * num_heads == input_dim, "input_dim must be equal to head_size * num_heads"
        self.head_size = head_size
        self.context_len = context_len
        self.W_key = nn.Linear(input_dim, num_heads * head_size, bias=qkv_bias)
        self.W_query = nn.Linear(input_dim, num_heads * head_size, bias=qkv_bias)
        self.W_value = nn.Linear(input_dim, num_heads * head_size, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        self.linear_out = nn.Linear(num_heads * head_size, num_heads * head_size)  # Final linear layer to project the concatenated output of all heads back to original dimension (n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(context_len, context_len)).view(1, 1, context_len, context_len))  # Lower triangular mask for causal attention

    def forward(self, x):
        batch, num_tokens, input_dim = x.shape
        # Compute queries, keys, and values
        Q = self.W_query(x)
        K = self.W_key(x)
        V = self.W_value(x)

        # Reshape for multi-head attention
        queries = Q.view(batch, num_tokens, self.num_heads, self.head_size).transpose(1, 2)  # (b, num_tokens, num_heads, head_size) -> (b, num_heads, num_tokens, head_size)
        keys = K.view(batch, num_tokens, self.num_heads, self.head_size).transpose
        values = V.view(batch, num_tokens, self.num_heads, self.head_size).transpose(1, 2)  

        attention_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attention_scores = attention_scores.masked_fill(~mask_bool, -torch.inf)
        attention_probs = torch.softmax(attention_scores / (self.head_size ** 0.5), dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = attention_probs @ values
        context = context.transpose(1, 2).contiguous().view(batch, num_tokens, self.num_heads * self.head_size) # Concatenate the heads and reshape back to (batch, num_tokens, num_heads * head_size)
        output = self.linear_out(context)
        return output 