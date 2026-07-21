import yaml
from gpt import GPT
from generate import generate_text
from tokenizer import Tokenizer
import torch
import matplotlib.pyplot as plt
import os
import requests
from dataloader import CustomDataset, dataloader_v1

# with open('config.yaml', 'r') as file:
#     config = yaml.safe_load(file)
# # config = 

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = GPT(config).to(device)  # Move model to device
# total_params = sum(p.numel() for p in model.parameters())
# print(f"Total number of parameters: {total_params:,}")
# tokenizer = Tokenizer("openai")
# start_context = "Once upon a time"
# token_ids = generate_text(
#     model=model,
#     tokenizer=tokenizer,
#     prompt=start_context,
#     max_length=10,
#     temperature=1.0,
#     device=config['GPT_CONFIG']['device']
# )

# print("Generated text:", token_ids)
# print("Generated text:", tokenizer.decode(token_ids, skip_special_tokens=True))

def calculate_loss(model, data_loader, device, num_batches):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()
    average_loss = total_loss / len(data_loader)
    return average_loss

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss(model, train_loader, device, num_batches=eval_iter)
        val_loss = calculate_loss(model, val_loader, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.positional_embedding.shape[0]
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0).to(device)  # add batch dimension and move to device
    with torch.no_grad():
        token_ids = generate_text(
            model=model, idx=encoded_tensor,
            max_length=50, context_size=context_size
        )
        decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def train_model(model, train_loader, val_loader, device, optimizer, num_epochs, eval_iter, tokenizer, start_context, eval_freq):
    
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Clear gradients
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            if (epoch + 1) % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                train_losses.append(train_loss)
                val_losses.append(val_loss)

        # Evaluate the model after each epoch
        # train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
        # print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        generate_sample(model, tokenizer, device, start_context)
    
    return train_losses, val_losses


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def main(config):
    torch.manual_seed(123)
    device = config['GPT_CONFIG']['device']

    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    

    model = GPT(config).to(device)  # Move model to device
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    tokenizer = Tokenizer("openai")
    start_context = "Every effort moves you"
    # print("length of text_data:", len(text_data))
    train_ratio = config['TRAINING_CONFIG']['train_ratio']
    train_size = int(len(text_data) * train_ratio)
    train_text = text_data[:train_size]
    val_text = text_data[train_size:]
    # print(type(train_text))

    #convert train_text and val_text to Dataset 
    train_text = CustomDataset([train_text], tokenizer, block_size=config['data_load']['block_size'], stride=config['data_load']['stride'], max_length=config['data_load']['max_length'])
    val_text = CustomDataset([val_text], tokenizer, block_size=config['data_load']['block_size'], stride=config['data_load']['stride'], max_length=config['data_load']['max_length'])
    train_loader = dataloader_v1(data=train_text, batch_size=config['TRAINING_CONFIG']['batch_size'], shuffle=True, num_workers=config['TRAINING_CONFIG']['num_workers'], mode='openai')
    val_loader = dataloader_v1(data=val_text, batch_size=config['TRAINING_CONFIG']['batch_size'], shuffle=False, num_workers=config['TRAINING_CONFIG']['num_workers'], mode='openai')
    
    # Assuming train_loader and val_loader are defined elsewhere
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=torch.optim.AdamW(model.parameters(), lr=float(config['TRAINING_CONFIG']['learning_rate'])),
        num_epochs=config['TRAINING_CONFIG']['num_epochs'],
        eval_iter=config['TRAINING_CONFIG']['eval_iter'],
        tokenizer=tokenizer,
        start_context=start_context,
        eval_freq=config['TRAINING_CONFIG']['eval_freq']
    )

    return train_losses, val_losses, model, tokenizer

if __name__ == "__main__":
    torch.cuda.empty_cache()
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    train_lossess, val_lossess, model, tokenizer = main(config)

    plot_loss(train_lossess, val_lossess)

    torch.save(model.state_dict(), "gpt_model.pth")
    model.load_state_dict(torch.load("gpt_model.pth"), map_location=config['GPT_CONFIG']['device'], strict=False)

