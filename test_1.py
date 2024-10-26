import torch
from gpt.gpt_model import GPTModel
from utils.test_model_generation import test_model_generation

if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "embedding_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "dropout_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    print("####### Test 1 : model generation #######")
    test_model_generation(
        model=model,
        input_text="Hi, I am a large language model",
        tokenizer_encoding="gpt2",
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )
