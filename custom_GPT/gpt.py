import torch
import torch.nn as nn
from transformer import TransformerBlock
from embeddings import token_embedding_layer, sinusoidal_positional_embedding

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = token_embedding_layer(vocab_size=config["tokenizer"]['vocab_size'], embedding_dim=config['tokenizer']['embedding_dim'], device=config['GPT_CONFIG']['device'])
        self.positional_embedding = sinusoidal_positional_embedding(token_sequence_size=config['tokenizer']['token_sequence_size'], token_embedding_dim=config['tokenizer']['embedding_dim'], device=config['GPT_CONFIG']['device'])
        self.dropout_embedding = nn.Dropout(p=config['GPT_CONFIG']['dropout'])

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config=config['GPT_CONFIG']
            ) for _ in range(config['GPT_CONFIG']['n_layers'])
        ])

        self.final_layer_norm = nn.LayerNorm(config['tokenizer']['embedding_dim'])
        self.output_layer = nn.Linear(config['tokenizer']['embedding_dim'], config["tokenizer"]['vocab_size'], bias=False)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        token_embeddings = self.token_embedding(x)  # shape: (batch_size, sequence_length, embedding_dim)
        """# Note: the sequence lenght should be less than or equal to the token_sequence_size defined in config.yaml"""
        print(f"positional_embedding shape: {self.positional_embedding.shape}")
        print(f"x shape: {x.shape[0]}")
        positional_embeddings = self.positional_embedding[:x.shape[0], :].unsqueeze(0) # shape: (1, sequence_length, embedding_dim) 
        positional_embeddings = positional_embeddings.to(token_embeddings.device)    

        embeddings = token_embeddings + positional_embeddings  # shape: (batch_size, sequence_length, embedding_dim)
        embeddings = self.dropout_embedding(embeddings)

        for block in self.transformer_blocks:
            embeddings = block(embeddings)

        embeddings = self.final_layer_norm(embeddings)
        logits = self.output_layer(embeddings)  # shape: (batch_size, sequence_length, vocab_size)
        return logits


