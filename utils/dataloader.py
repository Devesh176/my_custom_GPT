import torch
import yaml
from torch.utils.data import Dataset, DataLoader  
from .tokenizer import Tokenizer
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class CustomDataset(Dataset):
        def __init__(self, data, tokenizer, block_size, stride, max_length=config['data_load']['block_size'] + config['data_load']['stride']):
            self.data = data
            self.tokenizer = tokenizer
            self.block_size = block_size
            self.stride = stride
            self.max_length = max_length  # Ensure max_length is at least block_size + stride
            self.input_blocks = []
            self.target_blocks = []

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            text = self.data[idx]
            tokens = self.tokenizer.tokenize(text)
            assert len(tokens) > self.max_length, "Number of tokenized inputs must at least be equal to max_length+1"
            
            # Pad or truncate to block_size
            for i in range(0, len(tokens)-self.max_length+1, self.stride):
                input_token_block = tokens[i:i+self.max_length]
                target_token_block = tokens[i+1:i+self.max_length+1]

                if len(input_token_block) < self.block_size:
                    input_token_block += [0] * (self.block_size - len(input_token_block))  # Assuming 0 is the padding token
                    target_token_block += [0] * (self.block_size - len(target_token_block))  # Assuming 0 is the padding token
                else:
                    input_token_block = input_token_block[:self.block_size]
                    target_token_block = target_token_block[:self.block_size]
                # yield torch.tensor(input_token_block, dtype=torch.long), torch.tensor(target_token_block, dtype

                self.input_blocks.append(torch.tensor(input_token_block, dtype=torch.long))
                self.target_blocks.append(torch.tensor(target_token_block, dtype=torch.long))
            
            return torch.tensor(tokens, dtype=torch.long)

    

def create_dataset(data, tokenizer, block_size: int = config['data_load']['block_size'], stride: int = config['data_load']['stride'], max_length: int = config['data_load']['max_length']) -> Dataset:
    """
    Create a dataset from the given data and tokenizer.
    """
    return CustomDataset(data, tokenizer, block_size, stride, max_length)  
    

def dataloader_v1(data: Dataset, batch_size: int = config['data_load']['batch_size'], shuffle: bool = config['data_load']['shuffle'], num_workers: int = config['data_load']['num_workers']) -> DataLoader:
    """
    Create a DataLoader for the given dataset.
    """
    tokenizer = Tokenizer(mode='gpt2')
    dataset = CustomDataset(data, tokenizer, block_size=config['data_load']['block_size'], stride=config['data_load']['stride'], max_length=config['data_load']['max_length'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
