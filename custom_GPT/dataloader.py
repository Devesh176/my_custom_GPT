import torch
import yaml
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer

# BUG FIX: resolve config.yaml relative to this file, not the CWD
_ROOT = Path(__file__).resolve().parent.parent
with open(_ROOT / 'config.yaml', 'r') as file:
    config = yaml.safe_load(file)


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, block_size, stride, max_length=None):
        self.input_blocks  = []
        self.target_blocks = []
        self.pad_token_id  = 0
        effective_max = block_size if max_length is None else max(max_length, block_size)

        texts = data if isinstance(data, (list, tuple)) else [data]
        for item in texts:
            if isinstance(item, bytes):
                item = item.decode("utf-8", "ignore")
            if item is None:
                raise ValueError("Dataset item is None")
            if not isinstance(item, str):
                item = str(item)

            tokens = tokenizer.tokenize(item)
            if len(tokens) == 0:
                continue

            # Pad only if the whole document is shorter than one window
            if len(tokens) < block_size + 1:
                tokens = tokens + [self.pad_token_id] * (block_size + 1 - len(tokens))

            for i in range(0, len(tokens) - block_size, stride):
                self.input_blocks.append(torch.tensor(tokens[i:i + block_size],     dtype=torch.long))
                self.target_blocks.append(torch.tensor(tokens[i + 1:i + block_size + 1], dtype=torch.long))

        if len(self.input_blocks) != len(self.target_blocks):
            raise ValueError(f"Input/target mismatch: {len(self.input_blocks)} vs {len(self.target_blocks)}")

    def __len__(self):
        return len(self.input_blocks)

    def __getitem__(self, idx):
        return self.input_blocks[idx], self.target_blocks[idx]


def create_dataset(data, tokenizer,
                   block_size: int = config['data_load']['block_size'],
                   stride:     int = config['data_load']['stride'],
                   max_length: int = config['data_load']['max_length']) -> Dataset:
    return CustomDataset(data, tokenizer, block_size, stride, max_length)


def dataloader_v1(data, batch_size: int = config['data_load']['batch_size'],
                  shuffle: bool = config['data_load']['shuffle'],
                  num_workers: int = config['data_load']['num_workers'],
                  mode: str = 'openai', **kwargs) -> DataLoader:
    if isinstance(data, Dataset):
        dataset = data
    else:
        tokenizer = Tokenizer(mode=mode)
        dataset   = CustomDataset(data, tokenizer,
                                  block_size=config['data_load']['block_size'],
                                  stride=config['data_load']['stride'],
                                  max_length=config['data_load']['max_length'])
    if len(dataset) == 0:
        raise ValueError("No training samples found. Check input data, block_size and stride.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                     num_workers=num_workers, **kwargs)
