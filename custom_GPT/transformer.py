import torch.nn as nn
from mhattention import MultiHeadAttention
from utils import FeedForward, LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config["emb_dim"]
        # head_dim = config["head_dim"]
        n_heads = config["n_heads"]
        ctx_length = config["ctx_length"]
        dropout = config["dropout"]
        qkv_bias = config["qkv_bias"]

        self.attention = MultiHeadAttention(input_dim = input_dim, num_heads=n_heads, dropout=dropout, context_len=ctx_length, qkv_bias=qkv_bias)
        self.ff = FeedForward(input_dim, config["ff_hidden_dim"], input_dim)
        self.ln1 = LayerNorm(input_dim)
        self.ln2 = LayerNorm(input_dim)
        self.drop_shortcut = nn.Dropout(dropout)
    
    def forward(self, x):
        # first layer norm
        x_norm = self.ln1(x)
        # multi-head attention
        attn_out = self.attention(x_norm)
        # dropout and residual connection
        x = x + self.drop_shortcut(attn_out)

        # second layer norm
        x_norm = self.ln2(x)
        # feed forward network
        ff_out = self.ff(x_norm)
        # dropout and residual connection
        x = x + self.drop_shortcut(ff_out)

        return x

