import pandas as pd
import numpy as np
import regex as re

# The data set used
df = pd.read_parquet("hf://datasets/KisanVaani/agriculture-qa-english-only/data/train-00000-of-00001.parquet")

def preprocess_data(df):
    # text = df.to_string()
    text = re.sub(r"\s+", " ", df.to_string()) # Replace one or more whitespace characters with a single space
    # text = re.sub(r"([0-9]{1,6})","", text)
    chars = sorted(set(text.split(" ")))
    vocab_size = len(chars)
    print(len(chars))

    #character tokenizer
    #there can be word level and sentence , subword tokenizer
    string_to_int = {ch:i for i,ch in enumerate(chars)}
    int_to_string = {i:ch for i,ch in enumerate(chars)}
    encode = lambda s: [string_to_int[c] for c in s.split(" ") if c in string_to_int]
    decode = lambda l: " ".join([int_to_string[i] for i in l])

    # ec = encode("hello")
    # print(decode(ec))
    data = torch.tensor(encode(text), dtype=torch.long)
    # print(data[1000:1100])
    return data, vocab_size, encode, decode