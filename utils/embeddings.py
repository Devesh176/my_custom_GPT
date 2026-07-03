import yaml
import torch
import torch.nn as nn

config = yaml.safe_load(open('config.yaml', 'r'))

def token_embedding_layer(vocab_size: int = config["tokenizer"]['vocab_size'], embedding_dim: int = config['tokenizer']['embedding_dim']) -> nn.Embedding:
    """
    Create an embedding layer for token embeddings.
    """
    return nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)


def sinusoidal_positional_embedding(token_sequence_size: int = config['tokenizer']['token_sequence_size'], token_embedding_dim:int = config['tokenizer']['embedding_dim'], n=10000.0):
    # the dim should be even because it pairs a sine wave for the even index (2i) and a cosine wave for the odd index (2i+1) to form 2D rotational vectors for each frequency
    if token_embedding_dim % 2 != 0: 
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

    T = token_sequence_size
    d = token_embedding_dim #d_model=head_num*d_k, not d_q, d_k, d_v

    # using cuda to speed up the computation of positional embeddings
    if torch.cuda.is_available():
        positions = torch.arange(0, T, device='cuda').unsqueeze_(1)
        embeddings = torch.zeros(T, d, device='cuda')
    else:
        positions = torch.arange(0, T).unsqueeze_(1)
        embeddings = torch.zeros(T, d)
    
    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings