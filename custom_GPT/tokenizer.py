import tiktoken
from bpe import BPETokenizerSimple
from pathlib import Path
import yaml

# BUG FIX: resolve config.yaml relative to this file, not the CWD
_ROOT = Path(__file__).resolve().parent.parent
with open(_ROOT / 'config.yaml', 'r') as file:
    config = yaml.safe_load(file)

tokenizer_vocab_path   = _ROOT / config["tokenizer"]["tokenizer_vocab_path"]
tokenizer_merges_path  = _ROOT / config["tokenizer"]["tokenizer_merges_path"]


class Tokenizer:
    def __init__(self, mode: str = "openai"):
        """Initialize the tokenizer. mode: 'openai' | 'cl100k_base' | 'bpe'"""
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
        if self.mode != "bpe":
            return self.enc.encode(text)
        else:
            if not tokenizer_vocab_path.exists():
                self.tokenizer.train(text, vocab_size=config['tokenizer']['vocab_size'])
                self.tokenizer.save(tokenizer_vocab_path, tokenizer_merges_path)
            self.tokenizer.load_vocab_and_merges(tokenizer_vocab_path, tokenizer_merges_path)
            return self.tokenizer.encode(text)

    def decode(self, token_ids: list, skip_special_tokens: bool = True) -> str:
        if self.mode != "bpe":
            return self.enc.decode(token_ids)
        return self.tokenizer.decode(token_ids)