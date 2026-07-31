import torch


def softmax_with_temperature(logits, temperature):
    return torch.softmax(logits / temperature, dim=-1)


def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, device='cpu'):
    model.eval()
    # BUG FIX: get context window size from the learnable positional embedding
    context_size = model.positional_embedding.num_embeddings

    input_ids    = torch.tensor(tokenizer.tokenize(prompt)).unsqueeze(0).to(device)
    generated_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_length):
            # BUG FIX: truncate to context window to avoid index-out-of-range
            idx_cond = generated_ids[:, -context_size:]
            outputs  = model(idx_cond)
            next_token_logits = outputs[:, -1, :]
            next_token_probs  = softmax_with_temperature(next_token_logits, temperature)
            next_token_id     = torch.multinomial(next_token_probs, num_samples=1)
            generated_ids     = torch.cat((generated_ids, next_token_id), dim=1)

    return tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)