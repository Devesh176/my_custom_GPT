# my_custom_GPT
## The GPT Architecture is shown in figure below.
![alt text](images/GPT_architecture.png)


# Custom GPT just for Learning !!

This project focuses on building andGPT model from scratch focusing on learning and hands on experience.

## Project Structure
```
├── cmd.txt
├── config.yaml
├── custom_GPT
│   ├── BigramLM.py
│   ├── bpe.py
│   ├── dataloader.py
│   ├── embeddings.py
│   ├── generate.py
│   ├── gpt.py
│   ├── mhattention.py
│   ├── __pycache__
│   │   ├── bpe.cpython-312.pyc
│   │   ├── dataloader.cpython-312.pyc
│   │   ├── embeddings.cpython-312.pyc
│   │   ├── generate.cpython-312.pyc
│   │   ├── gpt.cpython-312.pyc
│   │   ├── mhattention.cpython-312.pyc
│   │   ├── tokenizer.cpython-312.pyc
│   │   ├── transformer.cpython-312.pyc
│   │   └── utils.cpython-312.pyc
│   ├── tokenizer.py
│   ├── train.py
│   ├── transformer.py
│   └── utils.py
├── custom_llm.ipynb
├── data
│   ├── process_data.py
│   └── __pycache__
│       └── process_data.cpython-39.pyc
├── finetune_gpt2.ipynb
├── images
│   └── GPT_architecture.png
├── kaggle_train.ipynb
├── LICENSE
├── model
│   └── gpt_model.pth
├── README.md
├── requirements.txt
└── the-verdict.txt
```

## Getting Started

1.  **Clone the repository:** 
2.  **Install dependencies:**
  ```bash
     pip install pandas numpy regex torch torchtext transformers sentencepiece tqdm datasets
  ```

## Data

The project uses the "WikiText" datset.

## Models

-   **Custom Bigram Language Model:** A basic language model implemented from scratch to understand the fundamentals of language modeling.
-   **Custom GPT-like Transformer:** 163M-parameter GPT-2-scale Transformer from scratch in PyTorch


## Training and Evaluation

The notebook ["kaggle_train.py"](https://github.com/Devesh176/my_custom_GPT/blob/main/kaggle_train.ipynb) includes code for training the model on kaggle session.

## Results
Trained on WikiText-103 (538M chars, 7,190 batches/epoch) across 2×T4 GPUs (29 GB VRAM) using nn.DataParallel; achieved Train Loss: 3.29 | Val Loss: 3.35 by epoch 2 with no sign of overfitting.
