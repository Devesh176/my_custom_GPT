# my_custom_GPT
## The GPT Architecture is shown in figure below.
![alt text](images/GPT_architecture.png)


# Custom GPT for Agricultural QA

This project focuses on building and fine-tuning a custom GPT model for question answering within the agricultural domain. The model is trained on a dataset of agricultural questions and answers.

## Project Structure

- Several code cells within the notebook demonstrate the steps for:
    - Loading and preprocessing the agricultural QA dataset.
    - Implementing a custom Bigram Language Model.
    - Implementing a GPT-like Transformer model from scratch.
    - Training and evaluating the custom models.

## Getting Started

1.  **Clone the repository (if applicable):** If the code is in a separate repository, provide instructions on how to clone it.
2.  **Install dependencies:**
  ```bash
     pip install pandas numpy regex torch torchtext transformers sentencepiece tqdm datasets
  ```
3.  **Run the notebook:** Execute the cells in the notebook sequentially to load the data, define and train the models, and fine-tune the pre-trained model.

## Data

The project uses the "KisanVaani/agriculture-qa-english-only" dataset from Hugging Face. The data is loaded using `pandas.read_parquet`.

## Models

-   **Custom Bigram Language Model:** A basic language model implemented from scratch to understand the fundamentals of language modeling.
-   **Custom GPT-like Transformer:** A more advanced model built with Transformer blocks, including multi-head attention and feed-forward networks.


## Training and Evaluation

The notebook includes code for:

-   Splitting the data into training and validation sets.
-   Defining loss functions and optimizers.
-   Training loops for both custom and fine-tuned models.
-   Evaluating the models using loss metrics.

## Results

The training and validation loss are printed during the training process to monitor the model's performance.

## Suggestions

Please provide your suggestions to make the project more successful.

