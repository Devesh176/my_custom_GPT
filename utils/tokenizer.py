import tiktoken

class Tokenizer:
    def __init__(self, mode: str = "openai"):
        """Initialize the tokenizer with the specified mode"""
        self.mode = mode
        if mode == "openai":
            self.enc = tiktoken.encoding_for_model("gpt-2")
        elif mode == "cl100k_base":
            self.enc = tiktoken.get_encoding("cl100k_base")
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
    def tokenize(self, text: str) -> list:
        """Tokenize the sequence"""
        tokens = self.enc.encode(text)
        return tokens
        