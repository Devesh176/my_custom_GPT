import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformer import TransformerBlock
from embeddings import token_embedding_layer, positional_embedding_layer


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        emb_dim     = config['GPT_CONFIG']['emb_dim']
        vocab_size  = config['tokenizer']['vocab_size']
        ctx_length  = config['GPT_CONFIG']['ctx_length']

        self.token_embedding      = token_embedding_layer(vocab_size, emb_dim)
        # Learnable positional embeddings — ctx_length positions, each of size emb_dim
        self.positional_embedding = positional_embedding_layer(ctx_length, emb_dim)
        self.dropout_embedding    = nn.Dropout(p=config['GPT_CONFIG']['dropout'])

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config=config['GPT_CONFIG'])
            for _ in range(config['GPT_CONFIG']['n_layers'])
        ])
        # Gradient checkpointing: recompute activations on backward instead of
        # storing them — cuts peak activation memory by ~70% at ~20% speed cost.
        self.use_gradient_checkpointing = config['GPT_CONFIG'].get('gradient_checkpointing', False)

        self.final_layer_norm = nn.LayerNorm(emb_dim)
        self.output_layer     = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, x):
        # x: (batch, seq_len)
        batch, seq_len = x.shape
        token_emb = self.token_embedding(x)   # (batch, seq_len, emb_dim)

        # positions: (1, seq_len) — shared across the batch
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb   = self.positional_embedding(positions)  # (1, seq_len, emb_dim)

        embeddings = self.dropout_embedding(token_emb + pos_emb)

        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                # use_reentrant=False is required for autocast + gradient checkpointing
                embeddings = checkpoint(block, embeddings, use_reentrant=False)
            else:
                embeddings = block(embeddings)

        embeddings = self.final_layer_norm(embeddings)
        logits     = self.output_layer(embeddings)   # (batch, seq_len, vocab_size)
        return logits
