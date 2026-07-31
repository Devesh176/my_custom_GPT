"""
BigramLanguageModel — kept as a standalone educational reference only.
This file is NOT used in the main GPT training pipeline.

NOTE: The original broken import `from custom_GPT.params import ...` has been
removed because `params.py` does not exist in this project.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    """Simple character-level bigram model (educational baseline)."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, index, targets=None):
        logits = self.token_embedding_table(index)  # (B, T, vocab_size)
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        else:
            loss = None
        return logits, loss

    def generate(self, index, max_new_tokens: int):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(index)
            logits     = logits[:, -1, :]                     # (B, C)
            probs      = F.softmax(logits, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index      = torch.cat((index, index_next), dim=1)
        return index
