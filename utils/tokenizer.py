import tiktoken
from .bpe import BPETokenizerSimple
from pathlib import Path
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
tokenizer_vocab_path = Path(config["tokenizer_vocab_path"])
tokenizer_merges_path = Path(config["tokenizer_merges_path"])

class Tokenizer:
    def __init__(self, mode: str = "gpt2"):
        """Initialize the tokenizer with the specified mode"""
        self.mode = mode
        if mode == "bpe":
            self.tokenizer = BPETokenizerSimple()
        elif mode == "openai":
            self.enc = tiktoken.get_encoding("gpt2")
        elif mode == "cl100k_base":
            self.enc = tiktoken.get_encoding("cl100k_base")
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
    def tokenize(self, text: str) -> list:
        """Tokenize the sequence"""

        if self.mode != "bpe":
            tokens = self.enc.encode(text)
        else:
            if not tokenizer_vocab_path.exists():
                self.tokenizer.train(text, vocab_size=10000)  # Example vocab size
                self.tokenizer.save(tokenizer_vocab_path, tokenizer_merges_path)

            # load the tokenizer from the saved file
            self.tokenizer.load_vocab_and_merges(tokenizer_vocab_path, tokenizer_merges_path)

            tokens = self.tokenizer.encode(text)
        return tokens
        