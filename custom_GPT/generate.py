import torch

def softmax_with_temperature(logits, temperature):
    """
    Apply softmax to logits with temperature scaling.

    Args:
        logits (torch.Tensor): The input logits.
        temperature (float): The temperature value for scaling.

    Returns:
        torch.Tensor: The probabilities after applying softmax with temperature.
    """
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=-1)

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, device='cpu'):
    model.eval()
    input_ids = torch.tensor(tokenizer.tokenize(prompt)).unsqueeze(0).to(device)
    
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(generated_ids)
            next_token_logits = outputs[:, -1, :]
            next_token_probs = softmax_with_temperature(next_token_logits, temperature)
            next_token_id = torch.multinomial(next_token_probs, num_samples=1)
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
    
    # Convert tensor to list before decoding
    generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    return generated_text